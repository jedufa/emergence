#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace {

constexpr int32_t FOOD = 1;

__host__ __device__ __forceinline__ int64_t max_i64(const int64_t left, const int64_t right) {
    return left > right ? left : right;
}

__host__ __device__ __forceinline__ int64_t min_i64(const int64_t left, const int64_t right) {
    return left < right ? left : right;
}

__device__ __forceinline__ float clamp01_cuda(const float value) {
    if (value < 0.0f) {
        return 0.0f;
    }
    if (value > 1.0f) {
        return 1.0f;
    }
    return value;
}

__device__ __forceinline__ float safe_norm_cuda(const float value, const float denom) {
    if (denom <= 0.0f) {
        return 0.0f;
    }
    return clamp01_cuda(value / denom);
}

__device__ __forceinline__ int64_t bounded_active_count_cuda(
    const int64_t* __restrict__ active_count,
    const int64_t capacity
) {
    if (active_count == nullptr) {
        return capacity;
    }
    const int64_t count = active_count[0];
    if (count <= 0) {
        return 0;
    }
    return count < capacity ? count : capacity;
}

__global__ void density_counts_kernel(
    const int32_t* __restrict__ agent_grid,
    float* __restrict__ density,
    const int64_t height,
    const int64_t width,
    const int64_t radius,
    const bool include_self
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = height * width;
    if (idx >= total) {
        return;
    }
    if (agent_grid[idx] == -1) {
        return;
    }
    const int64_t y = idx / width;
    const int64_t x = idx - y * width;
    const int64_t y0 = max_i64(0, y - radius);
    const int64_t y1 = min_i64(height, y + radius + 1);
    const int64_t x0 = max_i64(0, x - radius);
    const int64_t x1 = min_i64(width, x + radius + 1);
    for (int64_t yy = y0; yy < y1; ++yy) {
        for (int64_t xx = x0; xx < x1; ++xx) {
            atomicAdd(density + yy * width + xx, 1.0f);
        }
    }
    if (!include_self) {
        atomicAdd(density + idx, -1.0f);
    }
}

__global__ void patch_observation_kernel(
    const int32_t* __restrict__ agent_ids,
    const int32_t* __restrict__ slots,
    const int32_t* __restrict__ position_indices,
    const int32_t* __restrict__ agent_pos_y,
    const int32_t* __restrict__ agent_pos_x,
    const int32_t* __restrict__ agent_grid,
    const int32_t* __restrict__ food_grid,
    const float* __restrict__ energy_by_slot,
    const float* __restrict__ health_by_slot,
    const float* __restrict__ patch_food_amount,
    const float* __restrict__ patch_cover,
    const float* __restrict__ patch_shelter,
    const float* __restrict__ patch_scent,
    const float* __restrict__ patch_waste,
    const float* __restrict__ patch_movement_cost,
    const float* __restrict__ density,
    float* __restrict__ out,
    const int64_t capacity,
    const int64_t* __restrict__ active_count,
    const int64_t height,
    const int64_t width,
    const int64_t view_radius,
    const float max_energy,
    const float start_health,
    const float patch_food_amount_max,
    const float patch_scent_max,
    const float patch_waste_max,
    const float density_denom,
    const bool patch_density_normalize,
    const float patch_movement_cost_min,
    const float patch_movement_cost_max
) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= bounded_active_count_cuda(active_count, capacity)) {
        return;
    }
    const int64_t k = 2 * view_radius + 1;
    const int64_t local_size = k * k;
    const int64_t channel_count = 8;
    const int32_t slot = slots[row];
    const int32_t position_index = position_indices[row];
    const int32_t cy = agent_pos_y[position_index];
    const int32_t cx = agent_pos_x[position_index];
    float* row_out = out + row * (channel_count * local_size + 2);
    int64_t t = 0;
    const float move_span = patch_movement_cost_max - patch_movement_cost_min;
    for (int64_t dy = -view_radius; dy <= view_radius; ++dy) {
        for (int64_t dx = -view_radius; dx <= view_radius; ++dx) {
            const int64_t y = static_cast<int64_t>(cy) + dy;
            const int64_t x = static_cast<int64_t>(cx) + dx;
            float legacy = 0.0f;
            float food_norm = 0.0f;
            float cover = 0.0f;
            float shelter = 0.0f;
            float scent_norm = 0.0f;
            float waste_norm = 0.0f;
            float density_norm = 0.0f;
            float movement_norm = 0.0f;
            if (y >= 0 && y < height && x >= 0 && x < width) {
                const int64_t idx = y * width + x;
                if (agent_grid[idx] != -1) {
                    legacy = -1.0f;
                } else if (food_grid[idx] == FOOD) {
                    legacy = 1.0f;
                }
                food_norm = safe_norm_cuda(patch_food_amount[idx], patch_food_amount_max);
                cover = clamp01_cuda(patch_cover[idx]);
                shelter = clamp01_cuda(patch_shelter[idx]);
                scent_norm = safe_norm_cuda(patch_scent[idx], patch_scent_max);
                waste_norm = safe_norm_cuda(patch_waste[idx], patch_waste_max);
                density_norm = patch_density_normalize ? clamp01_cuda(density[idx]) : safe_norm_cuda(density[idx], density_denom);
                if (move_span > 0.0f) {
                    movement_norm = clamp01_cuda((patch_movement_cost[idx] - patch_movement_cost_min) / move_span);
                }
            }
            row_out[t] = legacy;
            row_out[local_size + t] = food_norm;
            row_out[2 * local_size + t] = cover;
            row_out[3 * local_size + t] = shelter;
            row_out[4 * local_size + t] = scent_norm;
            row_out[5 * local_size + t] = waste_norm;
            row_out[6 * local_size + t] = density_norm;
            row_out[7 * local_size + t] = movement_norm;
            ++t;
        }
    }
    row_out[channel_count * local_size] = energy_by_slot[slot] / max_energy;
    row_out[channel_count * local_size + 1] = health_by_slot[slot] / start_health;
}

__global__ void flat_observation_kernel(
    const int32_t* __restrict__ slots,
    const int32_t* __restrict__ position_indices,
    const int32_t* __restrict__ agent_pos_y,
    const int32_t* __restrict__ agent_pos_x,
    const int32_t* __restrict__ agent_grid,
    const int32_t* __restrict__ food_grid,
    const float* __restrict__ energy_by_slot,
    const float* __restrict__ health_by_slot,
    float* __restrict__ out,
    const int64_t capacity,
    const int64_t* __restrict__ active_count,
    const int64_t height,
    const int64_t width,
    const int64_t view_radius,
    const float max_energy,
    const float start_health
) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= bounded_active_count_cuda(active_count, capacity)) {
        return;
    }
    const int32_t slot = slots[row];
    const int32_t position_index = position_indices[row];
    const int32_t cy = agent_pos_y[position_index];
    const int32_t cx = agent_pos_x[position_index];
    const int64_t side = 2 * view_radius + 1;
    const int64_t local_size = side * side;
    float* row_out = out + row * (local_size + 2);
    int64_t offset = 0;
    for (int64_t dy = -view_radius; dy <= view_radius; ++dy) {
        for (int64_t dx = -view_radius; dx <= view_radius; ++dx) {
            const int64_t y = static_cast<int64_t>(cy) + dy;
            const int64_t x = static_cast<int64_t>(cx) + dx;
            float value = 0.0f;
            if (y >= 0 && y < height && x >= 0 && x < width) {
                const int64_t cell = y * width + x;
                if (agent_grid[cell] != -1) {
                    value = -1.0f;
                } else if (food_grid[cell] == FOOD) {
                    value = 1.0f;
                }
            }
            row_out[offset++] = value;
        }
    }
    row_out[local_size] = energy_by_slot[slot] / max_energy;
    row_out[local_size + 1] = health_by_slot[slot] / start_health;
}

__global__ void density_counts_batched_kernel(
    const int32_t* __restrict__ agent_grid,
    float* __restrict__ density,
    const int64_t cell_count,
    const int64_t height,
    const int64_t width,
    const int64_t radius,
    const bool include_self
) {
    const int64_t flat_index = static_cast<int64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    const int64_t grid_cells = height * width;
    if (flat_index >= cell_count * grid_cells) {
        return;
    }
    const int64_t cell = flat_index / grid_cells;
    const int64_t index = flat_index - cell * grid_cells;
    const int64_t cell_offset = cell * grid_cells;
    if (agent_grid[cell_offset + index] == -1) {
        return;
    }
    const int64_t y = index / width;
    const int64_t x = index - y * width;
    const int64_t y0 = max_i64(0, y - radius);
    const int64_t y1 = min_i64(height, y + radius + 1);
    const int64_t x0 = max_i64(0, x - radius);
    const int64_t x1 = min_i64(width, x + radius + 1);
    for (int64_t yy = y0; yy < y1; ++yy) {
        for (int64_t xx = x0; xx < x1; ++xx) {
            atomicAdd(density + cell_offset + yy * width + xx, 1.0f);
        }
    }
    if (!include_self) {
        atomicAdd(density + cell_offset + index, -1.0f);
    }
}

__global__ void flat_observation_batched_kernel(
    const int32_t* __restrict__ slots,
    const int32_t* __restrict__ agent_pos_y,
    const int32_t* __restrict__ agent_pos_x,
    const int32_t* __restrict__ agent_grid,
    const int32_t* __restrict__ food_grid,
    const float* __restrict__ energy_by_slot,
    const float* __restrict__ health_by_slot,
    const int64_t* __restrict__ active_count,
    float* __restrict__ out,
    const int64_t cell_count,
    const int64_t capacity,
    const int64_t height,
    const int64_t width,
    const int64_t view_radius,
    const float max_energy,
    const float start_health
) {
    const int64_t flat_row = static_cast<int64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (flat_row >= cell_count * capacity) {
        return;
    }
    const int64_t cell = flat_row / capacity;
    const int64_t row = flat_row - cell * capacity;
    if (row >= bounded_active_count_cuda(active_count + cell, capacity)) {
        return;
    }
    const int32_t slot = slots[flat_row];
    const int64_t slot_index = cell * capacity + static_cast<int64_t>(slot);
    const int32_t cy = agent_pos_y[slot_index];
    const int32_t cx = agent_pos_x[slot_index];
    const int64_t grid_cells = height * width;
    const int64_t grid_offset = cell * grid_cells;
    const int64_t side = 2 * view_radius + 1;
    const int64_t local_size = side * side;
    float* row_out = out + flat_row * (local_size + 2);
    int64_t offset = 0;
    for (int64_t dy = -view_radius; dy <= view_radius; ++dy) {
        for (int64_t dx = -view_radius; dx <= view_radius; ++dx) {
            const int64_t y = static_cast<int64_t>(cy) + dy;
            const int64_t x = static_cast<int64_t>(cx) + dx;
            float value = 0.0f;
            if (y >= 0 && y < height && x >= 0 && x < width) {
                const int64_t world_index = grid_offset + y * width + x;
                if (agent_grid[world_index] != -1) {
                    value = -1.0f;
                } else if (food_grid[world_index] == FOOD) {
                    value = 1.0f;
                }
            }
            row_out[offset++] = value;
        }
    }
    row_out[local_size] = energy_by_slot[slot_index] / max_energy;
    row_out[local_size + 1] = health_by_slot[slot_index] / start_health;
}

__global__ void patch_observation_batched_kernel(
    const int32_t* __restrict__ agent_ids,
    const int32_t* __restrict__ slots,
    const int32_t* __restrict__ agent_pos_y,
    const int32_t* __restrict__ agent_pos_x,
    const int32_t* __restrict__ agent_grid,
    const int32_t* __restrict__ food_grid,
    const float* __restrict__ energy_by_slot,
    const float* __restrict__ health_by_slot,
    const float* __restrict__ patch_food_amount,
    const float* __restrict__ patch_cover,
    const float* __restrict__ patch_shelter,
    const float* __restrict__ patch_scent,
    const float* __restrict__ patch_waste,
    const float* __restrict__ patch_movement_cost,
    const float* __restrict__ density,
    const int64_t* __restrict__ active_count,
    float* __restrict__ out,
    const int64_t cell_count,
    const int64_t capacity,
    const int64_t height,
    const int64_t width,
    const int64_t view_radius,
    const float max_energy,
    const float start_health,
    const float patch_food_amount_max,
    const float patch_scent_max,
    const float patch_waste_max,
    const float density_denom,
    const bool patch_density_normalize,
    const float patch_movement_cost_min,
    const float patch_movement_cost_max
) {
    const int64_t flat_row = static_cast<int64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (flat_row >= cell_count * capacity) {
        return;
    }
    const int64_t cell = flat_row / capacity;
    const int64_t row = flat_row - cell * capacity;
    if (row >= bounded_active_count_cuda(active_count + cell, capacity)) {
        return;
    }
    const int32_t slot = slots[flat_row];
    const int64_t slot_index = cell * capacity + static_cast<int64_t>(slot);
    const int32_t cy = agent_pos_y[slot_index];
    const int32_t cx = agent_pos_x[slot_index];
    const int64_t grid_cells = height * width;
    const int64_t grid_offset = cell * grid_cells;
    const int64_t side = 2 * view_radius + 1;
    const int64_t local_size = side * side;
    constexpr int64_t channel_count = 8;
    float* row_out = out + flat_row * (channel_count * local_size + 2);
    int64_t offset = 0;
    const float move_span = patch_movement_cost_max - patch_movement_cost_min;
    for (int64_t dy = -view_radius; dy <= view_radius; ++dy) {
        for (int64_t dx = -view_radius; dx <= view_radius; ++dx) {
            const int64_t y = static_cast<int64_t>(cy) + dy;
            const int64_t x = static_cast<int64_t>(cx) + dx;
            float legacy = 0.0f;
            float food_norm = 0.0f;
            float cover = 0.0f;
            float shelter = 0.0f;
            float scent_norm = 0.0f;
            float waste_norm = 0.0f;
            float density_norm = 0.0f;
            float movement_norm = 0.0f;
            if (y >= 0 && y < height && x >= 0 && x < width) {
                const int64_t index = grid_offset + y * width + x;
                if (agent_grid[index] != -1) {
                    legacy = -1.0f;
                } else if (food_grid[index] == FOOD) {
                    legacy = 1.0f;
                }
                food_norm = safe_norm_cuda(
                    patch_food_amount[index],
                    patch_food_amount_max
                );
                cover = clamp01_cuda(patch_cover[index]);
                shelter = clamp01_cuda(patch_shelter[index]);
                scent_norm = safe_norm_cuda(patch_scent[index], patch_scent_max);
                waste_norm = safe_norm_cuda(patch_waste[index], patch_waste_max);
                density_norm = patch_density_normalize
                    ? clamp01_cuda(density[index])
                    : safe_norm_cuda(density[index], density_denom);
                if (move_span > 0.0f) {
                    movement_norm = clamp01_cuda(
                        (patch_movement_cost[index] - patch_movement_cost_min)
                            / move_span
                    );
                }
            }
            row_out[offset] = legacy;
            row_out[local_size + offset] = food_norm;
            row_out[2 * local_size + offset] = cover;
            row_out[3 * local_size + offset] = shelter;
            row_out[4 * local_size + offset] = scent_norm;
            row_out[5 * local_size + offset] = waste_norm;
            row_out[6 * local_size + offset] = density_norm;
            row_out[7 * local_size + offset] = movement_norm;
            ++offset;
        }
    }
    row_out[channel_count * local_size] = energy_by_slot[slot_index] / max_energy;
    row_out[channel_count * local_size + 1] = health_by_slot[slot_index] / start_health;
    (void)agent_ids;
}

void check_cuda_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

torch::Tensor cxxbackbone8r_patch_observation_frame_cuda_v1(
    torch::Tensor agent_ids,
    torch::Tensor slots,
    torch::Tensor agent_pos_y,
    torch::Tensor agent_pos_x,
    torch::Tensor agent_grid,
    torch::Tensor food_grid,
    torch::Tensor energy_by_slot,
    torch::Tensor health_by_slot,
    torch::Tensor patch_food_amount,
    torch::Tensor patch_cover,
    torch::Tensor patch_shelter,
    torch::Tensor patch_scent,
    torch::Tensor patch_waste,
    torch::Tensor patch_movement_cost,
    const int64_t view_radius,
    const double max_energy,
    const double start_health,
    const double patch_food_amount_max,
    const double patch_scent_max,
    const double patch_waste_max,
    const int64_t patch_density_radius,
    const bool patch_density_include_self,
    const bool patch_density_normalize,
    const double patch_movement_cost_min,
    const double patch_movement_cost_max
) {
    check_cuda_tensor(agent_ids, "agent_ids");
    check_cuda_tensor(slots, "slots");
    check_cuda_tensor(agent_pos_y, "agent_pos_y");
    check_cuda_tensor(agent_pos_x, "agent_pos_x");
    check_cuda_tensor(agent_grid, "agent_grid");
    check_cuda_tensor(food_grid, "food_grid");
    check_cuda_tensor(energy_by_slot, "energy_by_slot");
    check_cuda_tensor(health_by_slot, "health_by_slot");
    check_cuda_tensor(patch_food_amount, "patch_food_amount");
    check_cuda_tensor(patch_cover, "patch_cover");
    check_cuda_tensor(patch_shelter, "patch_shelter");
    check_cuda_tensor(patch_scent, "patch_scent");
    check_cuda_tensor(patch_waste, "patch_waste");
    check_cuda_tensor(patch_movement_cost, "patch_movement_cost");
    TORCH_CHECK(agent_ids.scalar_type() == torch::kInt32, "agent_ids must be int32");
    TORCH_CHECK(slots.scalar_type() == torch::kInt32, "slots must be int32");
    TORCH_CHECK(agent_pos_y.scalar_type() == torch::kInt32, "agent_pos_y must be int32");
    TORCH_CHECK(agent_pos_x.scalar_type() == torch::kInt32, "agent_pos_x must be int32");
    TORCH_CHECK(agent_grid.scalar_type() == torch::kInt32, "agent_grid must be int32");
    TORCH_CHECK(food_grid.scalar_type() == torch::kInt32, "food_grid must be int32");
    TORCH_CHECK(energy_by_slot.scalar_type() == torch::kFloat32, "energy_by_slot must be float32");
    TORCH_CHECK(health_by_slot.scalar_type() == torch::kFloat32, "health_by_slot must be float32");
    TORCH_CHECK(patch_food_amount.scalar_type() == torch::kFloat32, "patch_food_amount must be float32");
    TORCH_CHECK(patch_cover.scalar_type() == torch::kFloat32, "patch_cover must be float32");
    TORCH_CHECK(patch_shelter.scalar_type() == torch::kFloat32, "patch_shelter must be float32");
    TORCH_CHECK(patch_scent.scalar_type() == torch::kFloat32, "patch_scent must be float32");
    TORCH_CHECK(patch_waste.scalar_type() == torch::kFloat32, "patch_waste must be float32");
    TORCH_CHECK(patch_movement_cost.scalar_type() == torch::kFloat32, "patch_movement_cost must be float32");
    TORCH_CHECK(agent_grid.dim() == 2 && food_grid.sizes() == agent_grid.sizes(), "grids must be aligned 2D tensors");
    TORCH_CHECK(patch_food_amount.sizes() == agent_grid.sizes(), "patch_food_amount must align to grid");
    TORCH_CHECK(patch_cover.sizes() == agent_grid.sizes(), "patch_cover must align to grid");
    TORCH_CHECK(patch_shelter.sizes() == agent_grid.sizes(), "patch_shelter must align to grid");
    TORCH_CHECK(patch_scent.sizes() == agent_grid.sizes(), "patch_scent must align to grid");
    TORCH_CHECK(patch_waste.sizes() == agent_grid.sizes(), "patch_waste must align to grid");
    TORCH_CHECK(patch_movement_cost.sizes() == agent_grid.sizes(), "patch_movement_cost must align to grid");

    const int64_t n = agent_ids.size(0);
    const int64_t radius = max_i64(0, view_radius);
    const int64_t k = 2 * radius + 1;
    const int64_t local_size = k * k;
    const int64_t channel_count = 8;
    const int64_t height = agent_grid.size(0);
    const int64_t width = agent_grid.size(1);
    auto out = torch::empty({n, channel_count * local_size + 2}, agent_ids.options().dtype(torch::kFloat32));
    auto density = torch::zeros({height, width}, agent_ids.options().dtype(torch::kFloat32));
    auto position_indices = agent_ids.clone();

    const int threads = 128;
    const int64_t cells = height * width;
    const int density_blocks = static_cast<int>((cells + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    density_counts_kernel<<<density_blocks, threads, 0, stream>>>(
        agent_grid.data_ptr<int32_t>(),
        density.data_ptr<float>(),
        height,
        width,
        max_i64(0, patch_density_radius),
        patch_density_include_self
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int64_t density_radius = max_i64(0, patch_density_radius);
    float density_denom = static_cast<float>((2 * density_radius + 1) * (2 * density_radius + 1));
    if (!patch_density_include_self) {
        density_denom -= 1.0f;
    }
    if (density_denom <= 0.0f) {
        density_denom = 1.0f;
    }
    const int obs_blocks = static_cast<int>((n + threads - 1) / threads);
    patch_observation_kernel<<<obs_blocks, threads, 0, stream>>>(
        agent_ids.data_ptr<int32_t>(),
        slots.data_ptr<int32_t>(),
        position_indices.data_ptr<int32_t>(),
        agent_pos_y.data_ptr<int32_t>(),
        agent_pos_x.data_ptr<int32_t>(),
        agent_grid.data_ptr<int32_t>(),
        food_grid.data_ptr<int32_t>(),
        energy_by_slot.data_ptr<float>(),
        health_by_slot.data_ptr<float>(),
        patch_food_amount.data_ptr<float>(),
        patch_cover.data_ptr<float>(),
        patch_shelter.data_ptr<float>(),
        patch_scent.data_ptr<float>(),
        patch_waste.data_ptr<float>(),
        patch_movement_cost.data_ptr<float>(),
        density.data_ptr<float>(),
        out.data_ptr<float>(),
        n,
        nullptr,
        height,
        width,
        radius,
        static_cast<float>(max_energy == 0.0 ? 1.0 : max_energy),
        static_cast<float>(start_health == 0.0 ? 1.0 : start_health),
        static_cast<float>(patch_food_amount_max),
        static_cast<float>(patch_scent_max),
        static_cast<float>(patch_waste_max),
        density_denom,
        patch_density_normalize,
        static_cast<float>(patch_movement_cost_min),
        static_cast<float>(patch_movement_cost_max)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor eiel_cuda_owner_observation_frame_v1(
    torch::Tensor agent_ids_by_alive_order,
    torch::Tensor agent_id_to_slot,
    torch::Tensor pos_y,
    torch::Tensor pos_x,
    torch::Tensor agent_grid,
    torch::Tensor food_grid,
    torch::Tensor energy,
    torch::Tensor health,
    torch::Tensor patch_food_amount,
    torch::Tensor patch_cover,
    torch::Tensor patch_shelter,
    torch::Tensor patch_scent,
    torch::Tensor patch_waste,
    torch::Tensor patch_movement_cost,
    const bool patch_enabled,
    const int64_t view_radius,
    const double max_energy,
    const double start_health,
    const double patch_food_amount_max,
    const double patch_scent_max,
    const double patch_waste_max,
    const int64_t patch_density_radius,
    const bool patch_density_include_self,
    const bool patch_density_normalize,
    const double patch_movement_cost_min,
    const double patch_movement_cost_max
) {
    for (const auto& named : std::vector<std::pair<const char*, torch::Tensor>>{
             {"agent_ids_by_alive_order", agent_ids_by_alive_order},
             {"agent_id_to_slot", agent_id_to_slot},
             {"pos_y", pos_y},
             {"pos_x", pos_x},
             {"agent_grid", agent_grid},
             {"food_grid", food_grid},
             {"energy", energy},
             {"health", health},
             {"patch_food_amount", patch_food_amount},
             {"patch_cover", patch_cover},
             {"patch_shelter", patch_shelter},
             {"patch_scent", patch_scent},
             {"patch_waste", patch_waste},
             {"patch_movement_cost", patch_movement_cost},
         }) {
        check_cuda_tensor(named.second, named.first);
        TORCH_CHECK(
            named.second.device() == agent_ids_by_alive_order.device(),
            named.first,
            " must share the owner CUDA device"
        );
    }
    TORCH_CHECK(
        agent_ids_by_alive_order.scalar_type() == torch::kInt64 && agent_ids_by_alive_order.dim() == 1,
        "agent_ids_by_alive_order must be int64 [alive]"
    );
    TORCH_CHECK(agent_id_to_slot.scalar_type() == torch::kInt32 && agent_id_to_slot.dim() == 1, "agent_id_to_slot must be int32 [agent_ids]");
    TORCH_CHECK(pos_y.scalar_type() == torch::kInt32 && pos_x.scalar_type() == torch::kInt32, "positions must be int32");
    TORCH_CHECK(energy.scalar_type() == torch::kFloat32 && health.scalar_type() == torch::kFloat32, "energy and health must be float32");
    TORCH_CHECK(pos_y.sizes() == energy.sizes() && pos_x.sizes() == energy.sizes(), "position and energy slot banks must align");
    TORCH_CHECK(energy.sizes() == health.sizes(), "energy and health slot banks must align");
    TORCH_CHECK(agent_grid.scalar_type() == torch::kInt32 && food_grid.scalar_type() == torch::kInt32, "grids must be int32");
    TORCH_CHECK(agent_grid.dim() == 2 && food_grid.sizes() == agent_grid.sizes(), "grids must be aligned 2D tensors");
    for (const auto& tensor : {patch_food_amount, patch_cover, patch_shelter, patch_scent, patch_waste, patch_movement_cost}) {
        TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, "Patch observation grids must be float32");
        TORCH_CHECK(tensor.sizes() == agent_grid.sizes(), "Patch observation grids must align to agent_grid");
    }

    auto slots = agent_id_to_slot.index_select(0, agent_ids_by_alive_order).contiguous();
    auto agent_ids = agent_ids_by_alive_order.to(torch::kInt32).contiguous();
    const int64_t n = agent_ids.size(0);
    const int64_t radius = max_i64(0, view_radius);
    const int64_t side = 2 * radius + 1;
    const int64_t local_size = side * side;
    const int64_t height = agent_grid.size(0);
    const int64_t width = agent_grid.size(1);
    const int64_t channels = patch_enabled ? 8 : 1;
    auto out = torch::empty({n, channels * local_size + 2}, energy.options());
    if (n == 0) {
        return out;
    }

    constexpr int threads = 128;
    const int obs_blocks = static_cast<int>((n + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (!patch_enabled) {
        flat_observation_kernel<<<obs_blocks, threads, 0, stream>>>(
            slots.data_ptr<int32_t>(),
            slots.data_ptr<int32_t>(),
            pos_y.data_ptr<int32_t>(),
            pos_x.data_ptr<int32_t>(),
            agent_grid.data_ptr<int32_t>(),
            food_grid.data_ptr<int32_t>(),
            energy.data_ptr<float>(),
            health.data_ptr<float>(),
            out.data_ptr<float>(),
            n,
            nullptr,
            height,
            width,
            radius,
            static_cast<float>(max_energy == 0.0 ? 1.0 : max_energy),
            static_cast<float>(start_health == 0.0 ? 1.0 : start_health)
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }

    auto density = torch::zeros({height, width}, energy.options());
    const int64_t cells = height * width;
    const int density_blocks = static_cast<int>((cells + threads - 1) / threads);
    density_counts_kernel<<<density_blocks, threads, 0, stream>>>(
        agent_grid.data_ptr<int32_t>(),
        density.data_ptr<float>(),
        height,
        width,
        max_i64(0, patch_density_radius),
        patch_density_include_self
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int64_t density_radius = max_i64(0, patch_density_radius);
    float density_denom = static_cast<float>((2 * density_radius + 1) * (2 * density_radius + 1));
    if (!patch_density_include_self) {
        density_denom -= 1.0f;
    }
    if (density_denom <= 0.0f) {
        density_denom = 1.0f;
    }
    patch_observation_kernel<<<obs_blocks, threads, 0, stream>>>(
        agent_ids.data_ptr<int32_t>(),
        slots.data_ptr<int32_t>(),
        slots.data_ptr<int32_t>(),
        pos_y.data_ptr<int32_t>(),
        pos_x.data_ptr<int32_t>(),
        agent_grid.data_ptr<int32_t>(),
        food_grid.data_ptr<int32_t>(),
        energy.data_ptr<float>(),
        health.data_ptr<float>(),
        patch_food_amount.data_ptr<float>(),
        patch_cover.data_ptr<float>(),
        patch_shelter.data_ptr<float>(),
        patch_scent.data_ptr<float>(),
        patch_waste.data_ptr<float>(),
        patch_movement_cost.data_ptr<float>(),
        density.data_ptr<float>(),
        out.data_ptr<float>(),
        n,
        nullptr,
        height,
        width,
        radius,
        static_cast<float>(max_energy == 0.0 ? 1.0 : max_energy),
        static_cast<float>(start_health == 0.0 ? 1.0 : start_health),
        static_cast<float>(patch_food_amount_max),
        static_cast<float>(patch_scent_max),
        static_cast<float>(patch_waste_max),
        density_denom,
        patch_density_normalize,
        static_cast<float>(patch_movement_cost_min),
        static_cast<float>(patch_movement_cost_max)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor eiel_cuda_owner_observation_frame_counted_v1(
    torch::Tensor agent_ids_by_alive_order,
    torch::Tensor active_count,
    torch::Tensor agent_id_to_slot,
    torch::Tensor pos_y,
    torch::Tensor pos_x,
    torch::Tensor agent_grid,
    torch::Tensor food_grid,
    torch::Tensor energy,
    torch::Tensor health,
    torch::Tensor patch_food_amount,
    torch::Tensor patch_cover,
    torch::Tensor patch_shelter,
    torch::Tensor patch_scent,
    torch::Tensor patch_waste,
    torch::Tensor patch_movement_cost,
    const bool patch_enabled,
    const int64_t view_radius,
    const double max_energy,
    const double start_health,
    const double patch_food_amount_max,
    const double patch_scent_max,
    const double patch_waste_max,
    const int64_t patch_density_radius,
    const bool patch_density_include_self,
    const bool patch_density_normalize,
    const double patch_movement_cost_min,
    const double patch_movement_cost_max
) {
    for (const auto& named : std::vector<std::pair<const char*, torch::Tensor>>{
             {"agent_ids_by_alive_order", agent_ids_by_alive_order},
             {"active_count", active_count},
             {"agent_id_to_slot", agent_id_to_slot},
             {"pos_y", pos_y},
             {"pos_x", pos_x},
             {"agent_grid", agent_grid},
             {"food_grid", food_grid},
             {"energy", energy},
             {"health", health},
             {"patch_food_amount", patch_food_amount},
             {"patch_cover", patch_cover},
             {"patch_shelter", patch_shelter},
             {"patch_scent", patch_scent},
             {"patch_waste", patch_waste},
             {"patch_movement_cost", patch_movement_cost},
         }) {
        check_cuda_tensor(named.second, named.first);
        TORCH_CHECK(
            named.second.device() == agent_ids_by_alive_order.device(),
            named.first,
            " must share the owner CUDA device"
        );
    }
    TORCH_CHECK(
        agent_ids_by_alive_order.scalar_type() == torch::kInt64 && agent_ids_by_alive_order.dim() == 1,
        "agent_ids_by_alive_order must be fixed-capacity int64 [capacity]"
    );
    TORCH_CHECK(
        active_count.scalar_type() == torch::kInt64 && active_count.dim() == 1 && active_count.numel() == 1,
        "active_count must be CUDA int64[1]"
    );
    TORCH_CHECK(agent_id_to_slot.scalar_type() == torch::kInt32 && agent_id_to_slot.dim() == 1, "agent_id_to_slot must be int32 [agent_ids]");
    TORCH_CHECK(pos_y.scalar_type() == torch::kInt32 && pos_x.scalar_type() == torch::kInt32, "positions must be int32");
    TORCH_CHECK(energy.scalar_type() == torch::kFloat32 && health.scalar_type() == torch::kFloat32, "energy and health must be float32");
    TORCH_CHECK(pos_y.sizes() == energy.sizes() && pos_x.sizes() == energy.sizes(), "position and energy slot banks must align");
    TORCH_CHECK(energy.sizes() == health.sizes(), "energy and health slot banks must align");
    TORCH_CHECK(agent_grid.scalar_type() == torch::kInt32 && food_grid.scalar_type() == torch::kInt32, "grids must be int32");
    TORCH_CHECK(agent_grid.dim() == 2 && food_grid.sizes() == agent_grid.sizes(), "grids must be aligned 2D tensors");
    for (const auto& tensor : {patch_food_amount, patch_cover, patch_shelter, patch_scent, patch_waste, patch_movement_cost}) {
        TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, "Patch observation grids must be float32");
        TORCH_CHECK(tensor.sizes() == agent_grid.sizes(), "Patch observation grids must align to agent_grid");
    }

    const int64_t capacity = agent_ids_by_alive_order.size(0);
    auto row_indices = torch::arange(capacity, agent_ids_by_alive_order.options());
    auto active_mask = row_indices < torch::clamp(active_count, 0, capacity);
    auto safe_agent_ids = torch::where(
        active_mask,
        agent_ids_by_alive_order,
        torch::zeros_like(agent_ids_by_alive_order)
    ).contiguous();
    auto slots = agent_id_to_slot.index_select(0, safe_agent_ids).contiguous();
    slots = torch::where(active_mask, slots, torch::zeros_like(slots)).contiguous();
    auto agent_ids = safe_agent_ids.to(torch::kInt32).contiguous();
    const int64_t radius = max_i64(0, view_radius);
    const int64_t side = 2 * radius + 1;
    const int64_t local_size = side * side;
    const int64_t height = agent_grid.size(0);
    const int64_t width = agent_grid.size(1);
    const int64_t channels = patch_enabled ? 8 : 1;
    auto out = torch::zeros({capacity, channels * local_size + 2}, energy.options());
    if (capacity == 0) {
        return out;
    }

    constexpr int threads = 128;
    const int obs_blocks = static_cast<int>((capacity + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (!patch_enabled) {
        flat_observation_kernel<<<obs_blocks, threads, 0, stream>>>(
            slots.data_ptr<int32_t>(),
            slots.data_ptr<int32_t>(),
            pos_y.data_ptr<int32_t>(),
            pos_x.data_ptr<int32_t>(),
            agent_grid.data_ptr<int32_t>(),
            food_grid.data_ptr<int32_t>(),
            energy.data_ptr<float>(),
            health.data_ptr<float>(),
            out.data_ptr<float>(),
            capacity,
            active_count.data_ptr<int64_t>(),
            height,
            width,
            radius,
            static_cast<float>(max_energy == 0.0 ? 1.0 : max_energy),
            static_cast<float>(start_health == 0.0 ? 1.0 : start_health)
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }

    auto density = torch::zeros({height, width}, energy.options());
    const int64_t cells = height * width;
    const int density_blocks = static_cast<int>((cells + threads - 1) / threads);
    density_counts_kernel<<<density_blocks, threads, 0, stream>>>(
        agent_grid.data_ptr<int32_t>(),
        density.data_ptr<float>(),
        height,
        width,
        max_i64(0, patch_density_radius),
        patch_density_include_self
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int64_t density_radius = max_i64(0, patch_density_radius);
    float density_denom = static_cast<float>((2 * density_radius + 1) * (2 * density_radius + 1));
    if (!patch_density_include_self) {
        density_denom -= 1.0f;
    }
    if (density_denom <= 0.0f) {
        density_denom = 1.0f;
    }
    patch_observation_kernel<<<obs_blocks, threads, 0, stream>>>(
        agent_ids.data_ptr<int32_t>(),
        slots.data_ptr<int32_t>(),
        slots.data_ptr<int32_t>(),
        pos_y.data_ptr<int32_t>(),
        pos_x.data_ptr<int32_t>(),
        agent_grid.data_ptr<int32_t>(),
        food_grid.data_ptr<int32_t>(),
        energy.data_ptr<float>(),
        health.data_ptr<float>(),
        patch_food_amount.data_ptr<float>(),
        patch_cover.data_ptr<float>(),
        patch_shelter.data_ptr<float>(),
        patch_scent.data_ptr<float>(),
        patch_waste.data_ptr<float>(),
        patch_movement_cost.data_ptr<float>(),
        density.data_ptr<float>(),
        out.data_ptr<float>(),
        capacity,
        active_count.data_ptr<int64_t>(),
        height,
        width,
        radius,
        static_cast<float>(max_energy == 0.0 ? 1.0 : max_energy),
        static_cast<float>(start_health == 0.0 ? 1.0 : start_health),
        static_cast<float>(patch_food_amount_max),
        static_cast<float>(patch_scent_max),
        static_cast<float>(patch_waste_max),
        density_denom,
        patch_density_normalize,
        static_cast<float>(patch_movement_cost_min),
        static_cast<float>(patch_movement_cost_max)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor eiel_cuda_owner_observation_frame_slotted_v1(
    torch::Tensor agent_ids_by_alive_order,
    torch::Tensor active_slots_by_alive_order,
    torch::Tensor active_count,
    torch::Tensor pos_y,
    torch::Tensor pos_x,
    torch::Tensor agent_grid,
    torch::Tensor food_grid,
    torch::Tensor energy,
    torch::Tensor health,
    torch::Tensor patch_food_amount,
    torch::Tensor patch_cover,
    torch::Tensor patch_shelter,
    torch::Tensor patch_scent,
    torch::Tensor patch_waste,
    torch::Tensor patch_movement_cost,
    const bool patch_enabled,
    const int64_t view_radius,
    const double max_energy,
    const double start_health,
    const double patch_food_amount_max,
    const double patch_scent_max,
    const double patch_waste_max,
    const int64_t patch_density_radius,
    const bool patch_density_include_self,
    const bool patch_density_normalize,
    const double patch_movement_cost_min,
    const double patch_movement_cost_max
) {
    for (const auto& named : std::vector<std::pair<const char*, torch::Tensor>>{
             {"agent_ids_by_alive_order", agent_ids_by_alive_order},
             {"active_slots_by_alive_order", active_slots_by_alive_order},
             {"active_count", active_count},
             {"pos_y", pos_y},
             {"pos_x", pos_x},
             {"agent_grid", agent_grid},
             {"food_grid", food_grid},
             {"energy", energy},
             {"health", health},
             {"patch_food_amount", patch_food_amount},
             {"patch_cover", patch_cover},
             {"patch_shelter", patch_shelter},
             {"patch_scent", patch_scent},
             {"patch_waste", patch_waste},
             {"patch_movement_cost", patch_movement_cost},
         }) {
        check_cuda_tensor(named.second, named.first);
        TORCH_CHECK(
            named.second.device() == agent_ids_by_alive_order.device(),
            named.first,
            " must share the owner CUDA device"
        );
    }
    TORCH_CHECK(
        agent_ids_by_alive_order.scalar_type() == torch::kInt64
            && agent_ids_by_alive_order.dim() == 1,
        "agent_ids_by_alive_order must be fixed-capacity int64 [capacity]"
    );
    TORCH_CHECK(
        active_slots_by_alive_order.scalar_type() == torch::kInt32
            && active_slots_by_alive_order.sizes()
                == agent_ids_by_alive_order.sizes(),
        "active_slots_by_alive_order must be aligned int32 [capacity]"
    );
    TORCH_CHECK(
        active_count.scalar_type() == torch::kInt64
            && active_count.dim() == 1
            && active_count.numel() == 1,
        "active_count must be CUDA int64[1]"
    );
    TORCH_CHECK(
        pos_y.scalar_type() == torch::kInt32
            && pos_x.scalar_type() == torch::kInt32,
        "positions must be int32"
    );
    TORCH_CHECK(
        energy.scalar_type() == torch::kFloat32
            && health.scalar_type() == torch::kFloat32,
        "energy and health must be float32"
    );
    TORCH_CHECK(
        pos_y.sizes() == energy.sizes()
            && pos_x.sizes() == energy.sizes()
            && energy.sizes() == health.sizes(),
        "position, energy, and health slot banks must align"
    );
    TORCH_CHECK(
        agent_grid.scalar_type() == torch::kInt32
            && food_grid.scalar_type() == torch::kInt32
            && agent_grid.dim() == 2
            && food_grid.sizes() == agent_grid.sizes(),
        "grids must be aligned int32 2D tensors"
    );
    for (const auto& tensor : {
             patch_food_amount,
             patch_cover,
             patch_shelter,
             patch_scent,
             patch_waste,
             patch_movement_cost,
         }) {
        TORCH_CHECK(
            tensor.scalar_type() == torch::kFloat32
                && tensor.sizes() == agent_grid.sizes(),
            "Patch observation grids must be aligned float32 tensors"
        );
    }

    const int64_t capacity = agent_ids_by_alive_order.size(0);
    auto row_indices = torch::arange(capacity, agent_ids_by_alive_order.options());
    auto active_mask = row_indices < torch::clamp(active_count, 0, capacity);
    auto safe_agent_ids = torch::where(
        active_mask,
        agent_ids_by_alive_order,
        torch::zeros_like(agent_ids_by_alive_order)
    ).contiguous();
    auto slots = torch::where(
        active_mask,
        active_slots_by_alive_order,
        torch::zeros_like(active_slots_by_alive_order)
    ).contiguous();
    auto agent_ids = safe_agent_ids.to(torch::kInt32).contiguous();
    const int64_t radius = max_i64(0, view_radius);
    const int64_t side = 2 * radius + 1;
    const int64_t local_size = side * side;
    const int64_t height = agent_grid.size(0);
    const int64_t width = agent_grid.size(1);
    const int64_t channels = patch_enabled ? 8 : 1;
    auto out = torch::zeros({capacity, channels * local_size + 2}, energy.options());
    if (capacity == 0) {
        return out;
    }

    constexpr int threads = 128;
    const int obs_blocks = static_cast<int>((capacity + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (!patch_enabled) {
        flat_observation_kernel<<<obs_blocks, threads, 0, stream>>>(
            slots.data_ptr<int32_t>(),
            slots.data_ptr<int32_t>(),
            pos_y.data_ptr<int32_t>(),
            pos_x.data_ptr<int32_t>(),
            agent_grid.data_ptr<int32_t>(),
            food_grid.data_ptr<int32_t>(),
            energy.data_ptr<float>(),
            health.data_ptr<float>(),
            out.data_ptr<float>(),
            capacity,
            active_count.data_ptr<int64_t>(),
            height,
            width,
            radius,
            static_cast<float>(max_energy == 0.0 ? 1.0 : max_energy),
            static_cast<float>(start_health == 0.0 ? 1.0 : start_health)
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }

    auto density = torch::zeros({height, width}, energy.options());
    const int64_t cells = height * width;
    const int density_blocks = static_cast<int>((cells + threads - 1) / threads);
    density_counts_kernel<<<density_blocks, threads, 0, stream>>>(
        agent_grid.data_ptr<int32_t>(),
        density.data_ptr<float>(),
        height,
        width,
        max_i64(0, patch_density_radius),
        patch_density_include_self
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int64_t density_radius = max_i64(0, patch_density_radius);
    float density_denom = static_cast<float>(
        (2 * density_radius + 1) * (2 * density_radius + 1)
    );
    if (!patch_density_include_self) {
        density_denom -= 1.0f;
    }
    if (density_denom <= 0.0f) {
        density_denom = 1.0f;
    }
    patch_observation_kernel<<<obs_blocks, threads, 0, stream>>>(
        agent_ids.data_ptr<int32_t>(),
        slots.data_ptr<int32_t>(),
        slots.data_ptr<int32_t>(),
        pos_y.data_ptr<int32_t>(),
        pos_x.data_ptr<int32_t>(),
        agent_grid.data_ptr<int32_t>(),
        food_grid.data_ptr<int32_t>(),
        energy.data_ptr<float>(),
        health.data_ptr<float>(),
        patch_food_amount.data_ptr<float>(),
        patch_cover.data_ptr<float>(),
        patch_shelter.data_ptr<float>(),
        patch_scent.data_ptr<float>(),
        patch_waste.data_ptr<float>(),
        patch_movement_cost.data_ptr<float>(),
        density.data_ptr<float>(),
        out.data_ptr<float>(),
        capacity,
        active_count.data_ptr<int64_t>(),
        height,
        width,
        radius,
        static_cast<float>(max_energy == 0.0 ? 1.0 : max_energy),
        static_cast<float>(start_health == 0.0 ? 1.0 : start_health),
        static_cast<float>(patch_food_amount_max),
        static_cast<float>(patch_scent_max),
        static_cast<float>(patch_waste_max),
        density_denom,
        patch_density_normalize,
        static_cast<float>(patch_movement_cost_min),
        static_cast<float>(patch_movement_cost_max)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor eiel_cuda_owner_observation_frame_batched_counted_v1(
    torch::Tensor agent_ids_by_alive_order,
    torch::Tensor active_count,
    torch::Tensor agent_id_to_slot,
    torch::Tensor pos_y,
    torch::Tensor pos_x,
    torch::Tensor agent_grid,
    torch::Tensor food_grid,
    torch::Tensor energy,
    torch::Tensor health,
    torch::Tensor patch_food_amount,
    torch::Tensor patch_cover,
    torch::Tensor patch_shelter,
    torch::Tensor patch_scent,
    torch::Tensor patch_waste,
    torch::Tensor patch_movement_cost,
    const bool patch_enabled,
    const int64_t view_radius,
    const double max_energy,
    const double start_health,
    const double patch_food_amount_max,
    const double patch_scent_max,
    const double patch_waste_max,
    const int64_t patch_density_radius,
    const bool patch_density_include_self,
    const bool patch_density_normalize,
    const double patch_movement_cost_min,
    const double patch_movement_cost_max
) {
    TORCH_CHECK(
        agent_ids_by_alive_order.is_cuda(),
        "batched observations require CUDA tensors"
    );
    const auto device = agent_ids_by_alive_order.device();
    c10::cuda::CUDAGuard device_guard(device);
    for (const auto& named : std::vector<std::pair<const char*, torch::Tensor>>{
             {"agent_ids_by_alive_order", agent_ids_by_alive_order},
             {"active_count", active_count},
             {"agent_id_to_slot", agent_id_to_slot},
             {"pos_y", pos_y},
             {"pos_x", pos_x},
             {"agent_grid", agent_grid},
             {"food_grid", food_grid},
             {"energy", energy},
             {"health", health},
             {"patch_food_amount", patch_food_amount},
             {"patch_cover", patch_cover},
             {"patch_shelter", patch_shelter},
             {"patch_scent", patch_scent},
             {"patch_waste", patch_waste},
             {"patch_movement_cost", patch_movement_cost},
         }) {
        check_cuda_tensor(named.second, named.first);
        TORCH_CHECK(
            named.second.device() == device,
            named.first,
            " must share one CUDA device"
        );
    }
    TORCH_CHECK(
        agent_ids_by_alive_order.scalar_type() == torch::kInt64
            && agent_ids_by_alive_order.dim() == 2,
        "batched agent_ids_by_alive_order must be int64[cells,capacity]"
    );
    const int64_t cell_count = agent_ids_by_alive_order.size(0);
    const int64_t capacity = agent_ids_by_alive_order.size(1);
    TORCH_CHECK(cell_count > 0, "batched observations require at least one cell");
    TORCH_CHECK(
        active_count.scalar_type() == torch::kInt64
            && active_count.dim() == 2
            && active_count.size(0) == cell_count
            && active_count.size(1) == 1,
        "batched active_count must be int64[cells,1]"
    );
    TORCH_CHECK(
        agent_id_to_slot.scalar_type() == torch::kInt32
            && agent_id_to_slot.dim() == 2
            && agent_id_to_slot.size(0) == cell_count,
        "batched agent_id_to_slot must be int32[cells,agent_ids]"
    );
    const auto check_slot_bank = [&](const torch::Tensor& tensor,
                                     const char* name,
                                     c10::ScalarType dtype) {
        TORCH_CHECK(
            tensor.scalar_type() == dtype
                && tensor.dim() == 2
                && tensor.size(0) == cell_count
                && tensor.size(1) == capacity,
            name,
            " must be [cells,capacity]"
        );
    };
    check_slot_bank(pos_y, "pos_y", torch::kInt32);
    check_slot_bank(pos_x, "pos_x", torch::kInt32);
    check_slot_bank(energy, "energy", torch::kFloat32);
    check_slot_bank(health, "health", torch::kFloat32);
    TORCH_CHECK(
        agent_grid.scalar_type() == torch::kInt32
            && agent_grid.dim() == 3
            && agent_grid.size(0) == cell_count,
        "batched agent_grid must be int32[cells,height,width]"
    );
    TORCH_CHECK(
        food_grid.scalar_type() == torch::kInt32
            && food_grid.sizes() == agent_grid.sizes(),
        "batched food_grid must align with agent_grid"
    );
    const int64_t height = agent_grid.size(1);
    const int64_t width = agent_grid.size(2);
    for (const auto& tensor : {
             patch_food_amount,
             patch_cover,
             patch_shelter,
             patch_scent,
             patch_waste,
             patch_movement_cost,
         }) {
        TORCH_CHECK(
            tensor.scalar_type() == torch::kFloat32
                && tensor.sizes() == agent_grid.sizes(),
            "batched Patch observation grids must be float32 and world-aligned"
        );
    }

    auto row_indices = torch::arange(
        capacity,
        agent_ids_by_alive_order.options()
    ).reshape({1, capacity});
    auto active_mask = row_indices < torch::clamp(active_count, 0, capacity);
    auto safe_agent_ids = torch::where(
        active_mask,
        agent_ids_by_alive_order,
        torch::zeros_like(agent_ids_by_alive_order)
    ).contiguous();
    const int64_t agent_id_capacity = agent_id_to_slot.size(1);
    auto id_offsets = torch::arange(
        cell_count,
        agent_ids_by_alive_order.options()
    ).reshape({cell_count, 1}) * agent_id_capacity;
    auto global_agent_ids = (safe_agent_ids + id_offsets).reshape({-1});
    auto slots = agent_id_to_slot.reshape({-1})
        .index_select(0, global_agent_ids)
        .reshape({cell_count, capacity});
    slots = torch::where(active_mask, slots, torch::zeros_like(slots)).contiguous();
    auto agent_ids = safe_agent_ids.to(torch::kInt32).contiguous();
    const int64_t radius = max_i64(0, view_radius);
    const int64_t side = 2 * radius + 1;
    const int64_t local_size = side * side;
    const int64_t channels = patch_enabled ? 8 : 1;
    auto out = torch::zeros(
        {cell_count, capacity, channels * local_size + 2},
        energy.options()
    );
    if (capacity == 0) {
        return out;
    }

    constexpr int threads = 128;
    const int64_t total_rows = cell_count * capacity;
    const int obs_blocks = static_cast<int>((total_rows + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (!patch_enabled) {
        flat_observation_batched_kernel<<<obs_blocks, threads, 0, stream>>>(
            slots.data_ptr<int32_t>(),
            pos_y.data_ptr<int32_t>(),
            pos_x.data_ptr<int32_t>(),
            agent_grid.data_ptr<int32_t>(),
            food_grid.data_ptr<int32_t>(),
            energy.data_ptr<float>(),
            health.data_ptr<float>(),
            active_count.data_ptr<int64_t>(),
            out.data_ptr<float>(),
            cell_count,
            capacity,
            height,
            width,
            radius,
            static_cast<float>(max_energy == 0.0 ? 1.0 : max_energy),
            static_cast<float>(start_health == 0.0 ? 1.0 : start_health)
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }

    auto density = torch::zeros(
        {cell_count, height, width},
        energy.options()
    );
    const int64_t total_world_cells = cell_count * height * width;
    const int density_blocks = static_cast<int>(
        (total_world_cells + threads - 1) / threads
    );
    density_counts_batched_kernel<<<density_blocks, threads, 0, stream>>>(
        agent_grid.data_ptr<int32_t>(),
        density.data_ptr<float>(),
        cell_count,
        height,
        width,
        max_i64(0, patch_density_radius),
        patch_density_include_self
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int64_t density_radius = max_i64(0, patch_density_radius);
    float density_denom = static_cast<float>(
        (2 * density_radius + 1) * (2 * density_radius + 1)
    );
    if (!patch_density_include_self) {
        density_denom -= 1.0f;
    }
    if (density_denom <= 0.0f) {
        density_denom = 1.0f;
    }
    patch_observation_batched_kernel<<<obs_blocks, threads, 0, stream>>>(
        agent_ids.data_ptr<int32_t>(),
        slots.data_ptr<int32_t>(),
        pos_y.data_ptr<int32_t>(),
        pos_x.data_ptr<int32_t>(),
        agent_grid.data_ptr<int32_t>(),
        food_grid.data_ptr<int32_t>(),
        energy.data_ptr<float>(),
        health.data_ptr<float>(),
        patch_food_amount.data_ptr<float>(),
        patch_cover.data_ptr<float>(),
        patch_shelter.data_ptr<float>(),
        patch_scent.data_ptr<float>(),
        patch_waste.data_ptr<float>(),
        patch_movement_cost.data_ptr<float>(),
        density.data_ptr<float>(),
        active_count.data_ptr<int64_t>(),
        out.data_ptr<float>(),
        cell_count,
        capacity,
        height,
        width,
        radius,
        static_cast<float>(max_energy == 0.0 ? 1.0 : max_energy),
        static_cast<float>(start_health == 0.0 ? 1.0 : start_health),
        static_cast<float>(patch_food_amount_max),
        static_cast<float>(patch_scent_max),
        static_cast<float>(patch_waste_max),
        density_denom,
        patch_density_normalize,
        static_cast<float>(patch_movement_cost_min),
        static_cast<float>(patch_movement_cost_max)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

namespace {

constexpr int32_t CUDA_EMPTY = 0;
constexpr int32_t CUDA_ACT_GATHER = 4;
constexpr int32_t CUDA_ACT_REST = 5;
constexpr int32_t CUDA_ACT_ATTACK = 6;

__device__ __forceinline__ float action_cost_cuda(
    const int32_t action,
    const double move_cost,
    const double gather_cost,
    const double rest_cost,
    const double attack_cost,
    const double action_cost_mult,
    const double rest_cost_mult
) {
    float cost = static_cast<float>(move_cost);
    if (action == CUDA_ACT_GATHER) {
        cost = static_cast<float>(gather_cost);
    } else if (action == CUDA_ACT_REST) {
        cost = static_cast<float>(rest_cost);
    } else if (action == CUDA_ACT_ATTACK) {
        cost = static_cast<float>(attack_cost);
    }
    if (action_cost_mult != 1.0) {
        cost = static_cast<float>(static_cast<double>(cost) * action_cost_mult);
    }
    if (rest_cost_mult != 1.0 && action == CUDA_ACT_REST) {
        cost = static_cast<float>(static_cast<double>(cost) * rest_cost_mult);
    }
    return cost;
}

__global__ void scripted_action_generation_kernel(
    const int32_t* __restrict__ agent_ids,
    const int64_t* __restrict__ order,
    const int32_t* __restrict__ state_indices,
    const int32_t* __restrict__ position_indices,
    const int32_t* __restrict__ actions,
    int32_t* __restrict__ pos_y,
    int32_t* __restrict__ pos_x,
    int32_t* __restrict__ agent_grid,
    int32_t* __restrict__ food_grid,
    float* __restrict__ patch_food_amount,
    float* __restrict__ patch_scent_emit_pending,
    float* __restrict__ patch_waste_emit_pending,
    const float* __restrict__ patch_movement_cost,
    float* __restrict__ energies,
    const float* __restrict__ healths,
    bool* __restrict__ died_mask,
    float* __restrict__ food_gained_total,
    int32_t* __restrict__ moved_count,
    int32_t* __restrict__ tick_moved,
    float* __restrict__ tick_food_gained,
    float* __restrict__ metrics_float,
    int32_t* __restrict__ metrics_int,
    const int64_t ticks,
    const int64_t capacity,
    const int64_t* __restrict__ active_count,
    const int64_t action_width,
    const int64_t height,
    const int64_t width,
    const bool patch_enabled,
    const double move_cost,
    const double gather_cost,
    const double rest_cost,
    const double attack_cost,
    const double action_cost_mult,
    const double rest_cost_mult,
    const double max_energy,
    const double food_energy,
    const double patch_food_gather_amount,
    const double patch_food_presence_threshold,
    const double patch_scent_move_emit,
    const double patch_waste_move_emit,
    const double patch_scent_gather_emit,
    const double patch_waste_consume_emit,
    const bool patch_movement_cost_apply_enable,
    const double patch_movement_cost_energy_scale
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const int64_t n = bounded_active_count_cuda(active_count, capacity);

    for (int64_t row = 0; row < n; ++row) {
        died_mask[row] = false;
        food_gained_total[row] = 0.0f;
        moved_count[row] = 0;
    }
    for (int32_t i = 0; i < 6; ++i) {
        metrics_float[i] = 0.0f;
    }
    metrics_int[0] = 0;
    metrics_int[1] = static_cast<int32_t>(ticks);

    double gather_amount = patch_food_gather_amount;
    if (gather_amount <= 0.0) {
        gather_amount = 1.0;
    }
    const double threshold = patch_food_presence_threshold;

    for (int64_t tick = 0; tick < ticks; ++tick) {
        for (int64_t row = 0; row < n; ++row) {
            tick_moved[row] = 0;
            tick_food_gained[row] = 0.0f;
        }

        // Order-sensitive occupancy is intentionally serialized by agent id.
        for (int64_t sorted_i = 0; sorted_i < n; ++sorted_i) {
            const int64_t row = order[sorted_i];
            const int32_t state_index = state_indices[row];
            const int32_t position_index = position_indices[row];
            const int32_t action = actions[tick * action_width + state_index];
            if (action < 0 || action > 3) {
                continue;
            }
            const int32_t aid = agent_ids[row];
            int32_t y = pos_y[position_index];
            int32_t x = pos_x[position_index];
            int32_t ny = y;
            int32_t nx = x;
            if (action == 0) {
                --ny;
            } else if (action == 1) {
                ++ny;
            } else if (action == 2) {
                ++nx;
            } else {
                --nx;
            }
            if (ny < 0 || ny >= height || nx < 0 || nx >= width) {
                continue;
            }
            if (agent_grid[ny * width + nx] != -1) {
                continue;
            }
            if (agent_grid[y * width + x] == aid) {
                agent_grid[y * width + x] = -1;
            }
            agent_grid[ny * width + nx] = aid;
            pos_y[position_index] = ny;
            pos_x[position_index] = nx;
            tick_moved[row] = 1;
            moved_count[row] += 1;
        }

        if (patch_enabled) {
            for (int64_t sorted_i = 0; sorted_i < n; ++sorted_i) {
                const int64_t row = order[sorted_i];
                if (tick_moved[row] == 0) {
                    continue;
                }
                const int32_t position_index = position_indices[row];
                const int64_t cell = static_cast<int64_t>(pos_y[position_index]) * width + pos_x[position_index];
                if (patch_scent_move_emit > 0.0) {
                    const float delta = static_cast<float>(patch_scent_move_emit);
                    patch_scent_emit_pending[cell] += delta;
                    metrics_float[1] += delta;
                }
                if (patch_waste_move_emit > 0.0) {
                    const float delta = static_cast<float>(patch_waste_move_emit);
                    patch_waste_emit_pending[cell] += delta;
                    metrics_float[2] += delta;
                }
            }
        }

        for (int64_t sorted_i = 0; sorted_i < n; ++sorted_i) {
            const int64_t row = order[sorted_i];
            const int32_t state_index = state_indices[row];
            if (actions[tick * action_width + state_index] != CUDA_ACT_GATHER) {
                continue;
            }
            const int32_t position_index = position_indices[row];
            const int64_t cell = static_cast<int64_t>(pos_y[position_index]) * width + pos_x[position_index];
            if (!patch_enabled) {
                if (food_grid[cell] == FOOD) {
                    food_grid[cell] = CUDA_EMPTY;
                    tick_food_gained[row] = static_cast<float>(food_energy);
                }
                continue;
            }

            // The CPU oracle promotes the authoritative float32 cell value to
            // float64 and rounds only when state is written back.
            const double amount = static_cast<double>(patch_food_amount[cell]);
            if (amount <= threshold) {
                food_grid[cell] = CUDA_EMPTY;
                continue;
            }
            const double harvested = amount > gather_amount ? gather_amount : amount;
            double post = amount - harvested;
            if (post < 0.0) {
                post = 0.0;
            }
            patch_food_amount[cell] = static_cast<float>(post);
            food_grid[cell] = post > threshold ? FOOD : CUDA_EMPTY;
            const double harvest_scale = harvested / gather_amount;
            tick_food_gained[row] = static_cast<float>(food_energy * harvest_scale);
            if (patch_scent_gather_emit > 0.0) {
                const double delta = patch_scent_gather_emit * harvest_scale;
                patch_scent_emit_pending[cell] = static_cast<float>(
                    static_cast<double>(patch_scent_emit_pending[cell]) + delta
                );
                metrics_float[3] += static_cast<float>(delta);
            }
            if (patch_waste_consume_emit > 0.0) {
                const double delta = patch_waste_consume_emit * harvest_scale;
                patch_waste_emit_pending[cell] = static_cast<float>(
                    static_cast<double>(patch_waste_emit_pending[cell]) + delta
                );
                metrics_float[4] += static_cast<float>(delta);
            }
            metrics_float[0] += static_cast<float>(harvested);
        }

        for (int64_t sorted_i = 0; sorted_i < n; ++sorted_i) {
            const int64_t row = order[sorted_i];
            const int32_t state_index = state_indices[row];
            const int32_t action = actions[tick * action_width + state_index];
            float cost = action_cost_cuda(
                action,
                move_cost,
                gather_cost,
                rest_cost,
                attack_cost,
                action_cost_mult,
                rest_cost_mult
            );
            if (patch_enabled && tick_moved[row] != 0 && patch_movement_cost_apply_enable) {
                const int32_t position_index = position_indices[row];
                const int64_t cell = static_cast<int64_t>(pos_y[position_index]) * width + pos_x[position_index];
                const float terrain_cost = patch_movement_cost[cell];
                const float mult = fmaxf(
                    0.0f,
                    1.0f + static_cast<float>(patch_movement_cost_energy_scale) * (terrain_cost - 1.0f)
                );
                const float old_cost = cost;
                cost = fmaxf(0.0f, old_cost * mult);
                metrics_float[5] += cost - old_cost;
                metrics_int[0] += 1;
            }
            float energy = energies[state_index] - cost + tick_food_gained[row];
            if (energy > static_cast<float>(max_energy)) {
                energy = static_cast<float>(max_energy);
            }
            if (energy < 0.0f) {
                energy = 0.0f;
            }
            energies[state_index] = energy;
            food_gained_total[row] += tick_food_gained[row];
            died_mask[row] = energy <= 0.0f || healths[state_index] <= 0.0f;
        }
    }
}

}  // namespace

std::vector<torch::Tensor> eiel_cuda_scripted_action_generation_v1(
    torch::Tensor agent_ids,
    torch::Tensor actions_by_tick_agent,
    torch::Tensor agent_pos_y,
    torch::Tensor agent_pos_x,
    torch::Tensor agent_grid,
    torch::Tensor food_grid,
    torch::Tensor patch_food_amount,
    torch::Tensor patch_scent_emit_pending,
    torch::Tensor patch_waste_emit_pending,
    torch::Tensor patch_movement_cost,
    torch::Tensor energies,
    torch::Tensor healths,
    const bool patch_enabled,
    const double move_cost,
    const double gather_cost,
    const double rest_cost,
    const double attack_cost,
    const double action_cost_mult,
    const double rest_cost_mult,
    const double max_energy,
    const double food_energy,
    const double patch_food_gather_amount,
    const double patch_food_presence_threshold,
    const double patch_scent_move_emit,
    const double patch_waste_move_emit,
    const double patch_scent_gather_emit,
    const double patch_waste_consume_emit,
    const bool patch_movement_cost_apply_enable,
    const double patch_movement_cost_energy_scale
) {
    for (const auto& named : std::vector<std::pair<const char*, torch::Tensor>>{
             {"agent_ids", agent_ids},
             {"actions_by_tick_agent", actions_by_tick_agent},
             {"agent_pos_y", agent_pos_y},
             {"agent_pos_x", agent_pos_x},
             {"agent_grid", agent_grid},
             {"food_grid", food_grid},
             {"patch_food_amount", patch_food_amount},
             {"patch_scent_emit_pending", patch_scent_emit_pending},
             {"patch_waste_emit_pending", patch_waste_emit_pending},
             {"patch_movement_cost", patch_movement_cost},
             {"energies", energies},
             {"healths", healths},
         }) {
        check_cuda_tensor(named.second, named.first);
        TORCH_CHECK(named.second.device() == agent_ids.device(), named.first, " must share the owner CUDA device");
    }
    TORCH_CHECK(agent_ids.scalar_type() == torch::kInt32 && agent_ids.dim() == 1, "agent_ids must be int32 [agents]");
    TORCH_CHECK(
        actions_by_tick_agent.scalar_type() == torch::kInt32 && actions_by_tick_agent.dim() == 2,
        "actions_by_tick_agent must be int32 [ticks, agents]"
    );
    TORCH_CHECK(actions_by_tick_agent.size(1) == agent_ids.size(0), "scripted action agent count mismatch");
    TORCH_CHECK(agent_pos_y.scalar_type() == torch::kInt32 && agent_pos_x.scalar_type() == torch::kInt32, "positions must be int32");
    TORCH_CHECK(agent_grid.scalar_type() == torch::kInt32 && food_grid.scalar_type() == torch::kInt32, "grids must be int32");
    TORCH_CHECK(agent_grid.dim() == 2 && food_grid.sizes() == agent_grid.sizes(), "grids must be aligned 2D tensors");
    TORCH_CHECK(energies.scalar_type() == torch::kFloat32 && healths.scalar_type() == torch::kFloat32, "energy and health must be float32");
    TORCH_CHECK(energies.sizes() == agent_ids.sizes() && healths.sizes() == agent_ids.sizes(), "energy and health must align to agents");
    for (const auto& tensor : {patch_food_amount, patch_scent_emit_pending, patch_waste_emit_pending, patch_movement_cost}) {
        TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, "Patch grids must be float32");
        TORCH_CHECK(tensor.sizes() == agent_grid.sizes(), "Patch grids must align to agent_grid");
    }

    auto order = torch::argsort(agent_ids.to(torch::kLong)).contiguous();
    auto state_indices = torch::arange(agent_ids.size(0), agent_ids.options().dtype(torch::kInt32));
    auto position_indices = agent_ids.clone();
    auto died_mask = torch::zeros(agent_ids.sizes(), agent_ids.options().dtype(torch::kBool));
    auto food_gained_total = torch::zeros(agent_ids.sizes(), agent_ids.options().dtype(torch::kFloat32));
    auto moved_count = torch::zeros(agent_ids.sizes(), agent_ids.options().dtype(torch::kInt32));
    auto tick_moved = torch::zeros(agent_ids.sizes(), agent_ids.options().dtype(torch::kInt32));
    auto tick_food_gained = torch::zeros(agent_ids.sizes(), agent_ids.options().dtype(torch::kFloat32));
    auto metrics_float = torch::zeros({6}, agent_ids.options().dtype(torch::kFloat32));
    auto metrics_int = torch::zeros({2}, agent_ids.options().dtype(torch::kInt32));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    scripted_action_generation_kernel<<<1, 1, 0, stream>>>(
        agent_ids.data_ptr<int32_t>(),
        order.data_ptr<int64_t>(),
        state_indices.data_ptr<int32_t>(),
        position_indices.data_ptr<int32_t>(),
        actions_by_tick_agent.data_ptr<int32_t>(),
        agent_pos_y.data_ptr<int32_t>(),
        agent_pos_x.data_ptr<int32_t>(),
        agent_grid.data_ptr<int32_t>(),
        food_grid.data_ptr<int32_t>(),
        patch_food_amount.data_ptr<float>(),
        patch_scent_emit_pending.data_ptr<float>(),
        patch_waste_emit_pending.data_ptr<float>(),
        patch_movement_cost.data_ptr<float>(),
        energies.data_ptr<float>(),
        healths.data_ptr<float>(),
        died_mask.data_ptr<bool>(),
        food_gained_total.data_ptr<float>(),
        moved_count.data_ptr<int32_t>(),
        tick_moved.data_ptr<int32_t>(),
        tick_food_gained.data_ptr<float>(),
        metrics_float.data_ptr<float>(),
        metrics_int.data_ptr<int32_t>(),
        actions_by_tick_agent.size(0),
        agent_ids.size(0),
        nullptr,
        actions_by_tick_agent.size(1),
        agent_grid.size(0),
        agent_grid.size(1),
        patch_enabled,
        move_cost,
        gather_cost,
        rest_cost,
        attack_cost,
        action_cost_mult,
        rest_cost_mult,
        max_energy,
        food_energy,
        patch_food_gather_amount,
        patch_food_presence_threshold,
        patch_scent_move_emit,
        patch_waste_move_emit,
        patch_scent_gather_emit,
        patch_waste_consume_emit,
        patch_movement_cost_apply_enable,
        patch_movement_cost_energy_scale
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {energies, died_mask, food_gained_total, moved_count, metrics_float, metrics_int};
}

std::vector<torch::Tensor> eiel_cuda_scripted_owner_action_generation_v1(
    torch::Tensor agent_ids_by_alive_order,
    torch::Tensor agent_id_to_slot,
    torch::Tensor actions_by_tick_slot,
    torch::Tensor pos_y,
    torch::Tensor pos_x,
    torch::Tensor agent_grid,
    torch::Tensor food_grid,
    torch::Tensor patch_food_amount,
    torch::Tensor patch_scent_emit_pending,
    torch::Tensor patch_waste_emit_pending,
    torch::Tensor patch_movement_cost,
    torch::Tensor energy,
    torch::Tensor health,
    const bool patch_enabled,
    const double move_cost,
    const double gather_cost,
    const double rest_cost,
    const double attack_cost,
    const double action_cost_mult,
    const double rest_cost_mult,
    const double max_energy,
    const double food_energy,
    const double patch_food_gather_amount,
    const double patch_food_presence_threshold,
    const double patch_scent_move_emit,
    const double patch_waste_move_emit,
    const double patch_scent_gather_emit,
    const double patch_waste_consume_emit,
    const bool patch_movement_cost_apply_enable,
    const double patch_movement_cost_energy_scale
) {
    for (const auto& named : std::vector<std::pair<const char*, torch::Tensor>>{
             {"agent_ids_by_alive_order", agent_ids_by_alive_order},
             {"agent_id_to_slot", agent_id_to_slot},
             {"actions_by_tick_slot", actions_by_tick_slot},
             {"pos_y", pos_y},
             {"pos_x", pos_x},
             {"agent_grid", agent_grid},
             {"food_grid", food_grid},
             {"patch_food_amount", patch_food_amount},
             {"patch_scent_emit_pending", patch_scent_emit_pending},
             {"patch_waste_emit_pending", patch_waste_emit_pending},
             {"patch_movement_cost", patch_movement_cost},
             {"energy", energy},
             {"health", health},
         }) {
        check_cuda_tensor(named.second, named.first);
        TORCH_CHECK(
            named.second.device() == agent_ids_by_alive_order.device(),
            named.first,
            " must share the owner CUDA device"
        );
    }
    TORCH_CHECK(
        agent_ids_by_alive_order.scalar_type() == torch::kInt64 && agent_ids_by_alive_order.dim() == 1,
        "agent_ids_by_alive_order must be int64 [alive]"
    );
    TORCH_CHECK(agent_id_to_slot.scalar_type() == torch::kInt32 && agent_id_to_slot.dim() == 1, "agent_id_to_slot must be int32 [agent_ids]");
    TORCH_CHECK(
        actions_by_tick_slot.scalar_type() == torch::kInt32 && actions_by_tick_slot.dim() == 2,
        "actions_by_tick_slot must be int32 [ticks, slots]"
    );
    TORCH_CHECK(pos_y.scalar_type() == torch::kInt32 && pos_x.scalar_type() == torch::kInt32, "positions must be int32");
    TORCH_CHECK(pos_y.sizes() == energy.sizes() && pos_x.sizes() == energy.sizes(), "position and energy slot banks must align");
    TORCH_CHECK(actions_by_tick_slot.size(1) == energy.size(0), "scripted action slot count mismatch");
    TORCH_CHECK(energy.scalar_type() == torch::kFloat32 && health.scalar_type() == torch::kFloat32, "energy and health must be float32");
    TORCH_CHECK(energy.sizes() == health.sizes(), "energy and health slot banks must align");
    TORCH_CHECK(agent_grid.scalar_type() == torch::kInt32 && food_grid.scalar_type() == torch::kInt32, "grids must be int32");
    TORCH_CHECK(agent_grid.dim() == 2 && food_grid.sizes() == agent_grid.sizes(), "grids must be aligned 2D tensors");
    for (const auto& tensor : {patch_food_amount, patch_scent_emit_pending, patch_waste_emit_pending, patch_movement_cost}) {
        TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, "Patch grids must be float32");
        TORCH_CHECK(tensor.sizes() == agent_grid.sizes(), "Patch grids must align to agent_grid");
    }

    auto agent_ids = agent_ids_by_alive_order.to(torch::kInt32).contiguous();
    auto state_indices = agent_id_to_slot.index_select(0, agent_ids_by_alive_order).contiguous();
    auto position_indices = state_indices;
    auto order = torch::argsort(agent_ids_by_alive_order).contiguous();
    auto died_mask = torch::zeros(agent_ids.sizes(), agent_ids.options().dtype(torch::kBool));
    auto food_gained_total = torch::zeros(agent_ids.sizes(), agent_ids.options().dtype(torch::kFloat32));
    auto moved_count = torch::zeros(agent_ids.sizes(), agent_ids.options().dtype(torch::kInt32));
    auto tick_moved = torch::zeros(agent_ids.sizes(), agent_ids.options().dtype(torch::kInt32));
    auto tick_food_gained = torch::zeros(agent_ids.sizes(), agent_ids.options().dtype(torch::kFloat32));
    auto metrics_float = torch::zeros({6}, agent_ids.options().dtype(torch::kFloat32));
    auto metrics_int = torch::zeros({2}, agent_ids.options().dtype(torch::kInt32));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    scripted_action_generation_kernel<<<1, 1, 0, stream>>>(
        agent_ids.data_ptr<int32_t>(),
        order.data_ptr<int64_t>(),
        state_indices.data_ptr<int32_t>(),
        position_indices.data_ptr<int32_t>(),
        actions_by_tick_slot.data_ptr<int32_t>(),
        pos_y.data_ptr<int32_t>(),
        pos_x.data_ptr<int32_t>(),
        agent_grid.data_ptr<int32_t>(),
        food_grid.data_ptr<int32_t>(),
        patch_food_amount.data_ptr<float>(),
        patch_scent_emit_pending.data_ptr<float>(),
        patch_waste_emit_pending.data_ptr<float>(),
        patch_movement_cost.data_ptr<float>(),
        energy.data_ptr<float>(),
        health.data_ptr<float>(),
        died_mask.data_ptr<bool>(),
        food_gained_total.data_ptr<float>(),
        moved_count.data_ptr<int32_t>(),
        tick_moved.data_ptr<int32_t>(),
        tick_food_gained.data_ptr<float>(),
        metrics_float.data_ptr<float>(),
        metrics_int.data_ptr<int32_t>(),
        actions_by_tick_slot.size(0),
        agent_ids.size(0),
        nullptr,
        actions_by_tick_slot.size(1),
        agent_grid.size(0),
        agent_grid.size(1),
        patch_enabled,
        move_cost,
        gather_cost,
        rest_cost,
        attack_cost,
        action_cost_mult,
        rest_cost_mult,
        max_energy,
        food_energy,
        patch_food_gather_amount,
        patch_food_presence_threshold,
        patch_scent_move_emit,
        patch_waste_move_emit,
        patch_scent_gather_emit,
        patch_waste_consume_emit,
        patch_movement_cost_apply_enable,
        patch_movement_cost_energy_scale
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {energy, died_mask, food_gained_total, moved_count, metrics_float, metrics_int, state_indices};
}

std::vector<torch::Tensor> eiel_cuda_scripted_owner_action_generation_counted_v1(
    torch::Tensor agent_ids_by_alive_order,
    torch::Tensor active_count,
    torch::Tensor agent_id_to_slot,
    torch::Tensor actions_by_tick_slot,
    torch::Tensor pos_y,
    torch::Tensor pos_x,
    torch::Tensor agent_grid,
    torch::Tensor food_grid,
    torch::Tensor patch_food_amount,
    torch::Tensor patch_scent_emit_pending,
    torch::Tensor patch_waste_emit_pending,
    torch::Tensor patch_movement_cost,
    torch::Tensor energy,
    torch::Tensor health,
    const bool patch_enabled,
    const double move_cost,
    const double gather_cost,
    const double rest_cost,
    const double attack_cost,
    const double action_cost_mult,
    const double rest_cost_mult,
    const double max_energy,
    const double food_energy,
    const double patch_food_gather_amount,
    const double patch_food_presence_threshold,
    const double patch_scent_move_emit,
    const double patch_waste_move_emit,
    const double patch_scent_gather_emit,
    const double patch_waste_consume_emit,
    const bool patch_movement_cost_apply_enable,
    const double patch_movement_cost_energy_scale
) {
    for (const auto& named : std::vector<std::pair<const char*, torch::Tensor>>{
             {"agent_ids_by_alive_order", agent_ids_by_alive_order},
             {"active_count", active_count},
             {"agent_id_to_slot", agent_id_to_slot},
             {"actions_by_tick_slot", actions_by_tick_slot},
             {"pos_y", pos_y},
             {"pos_x", pos_x},
             {"agent_grid", agent_grid},
             {"food_grid", food_grid},
             {"patch_food_amount", patch_food_amount},
             {"patch_scent_emit_pending", patch_scent_emit_pending},
             {"patch_waste_emit_pending", patch_waste_emit_pending},
             {"patch_movement_cost", patch_movement_cost},
             {"energy", energy},
             {"health", health},
         }) {
        check_cuda_tensor(named.second, named.first);
        TORCH_CHECK(
            named.second.device() == agent_ids_by_alive_order.device(),
            named.first,
            " must share the owner CUDA device"
        );
    }
    TORCH_CHECK(
        agent_ids_by_alive_order.scalar_type() == torch::kInt64 && agent_ids_by_alive_order.dim() == 1,
        "agent_ids_by_alive_order must be fixed-capacity int64 [capacity]"
    );
    TORCH_CHECK(
        active_count.scalar_type() == torch::kInt64 && active_count.dim() == 1 && active_count.numel() == 1,
        "active_count must be CUDA int64[1]"
    );
    TORCH_CHECK(agent_id_to_slot.scalar_type() == torch::kInt32 && agent_id_to_slot.dim() == 1, "agent_id_to_slot must be int32 [agent_ids]");
    TORCH_CHECK(
        actions_by_tick_slot.scalar_type() == torch::kInt32 && actions_by_tick_slot.dim() == 2,
        "actions_by_tick_slot must be int32 [ticks, slots]"
    );
    TORCH_CHECK(pos_y.scalar_type() == torch::kInt32 && pos_x.scalar_type() == torch::kInt32, "positions must be int32");
    TORCH_CHECK(pos_y.sizes() == energy.sizes() && pos_x.sizes() == energy.sizes(), "position and energy slot banks must align");
    TORCH_CHECK(actions_by_tick_slot.size(1) == energy.size(0), "scripted action slot count mismatch");
    TORCH_CHECK(energy.scalar_type() == torch::kFloat32 && health.scalar_type() == torch::kFloat32, "energy and health must be float32");
    TORCH_CHECK(energy.sizes() == health.sizes(), "energy and health slot banks must align");
    TORCH_CHECK(agent_grid.scalar_type() == torch::kInt32 && food_grid.scalar_type() == torch::kInt32, "grids must be int32");
    TORCH_CHECK(agent_grid.dim() == 2 && food_grid.sizes() == agent_grid.sizes(), "grids must be aligned 2D tensors");
    for (const auto& tensor : {patch_food_amount, patch_scent_emit_pending, patch_waste_emit_pending, patch_movement_cost}) {
        TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, "Patch grids must be float32");
        TORCH_CHECK(tensor.sizes() == agent_grid.sizes(), "Patch grids must align to agent_grid");
    }

    const int64_t capacity = agent_ids_by_alive_order.size(0);
    auto row_indices = torch::arange(capacity, agent_ids_by_alive_order.options());
    auto active_mask = row_indices < torch::clamp(active_count, 0, capacity);
    auto safe_agent_ids_i64 = torch::where(
        active_mask,
        agent_ids_by_alive_order,
        torch::zeros_like(agent_ids_by_alive_order)
    ).contiguous();
    auto gathered_state_indices = agent_id_to_slot.index_select(0, safe_agent_ids_i64).contiguous();
    auto safe_state_indices = torch::where(
        active_mask,
        gathered_state_indices,
        torch::zeros_like(gathered_state_indices)
    ).contiguous();
    auto state_indices = torch::where(
        active_mask,
        gathered_state_indices,
        torch::full_like(gathered_state_indices, -1)
    ).contiguous();
    auto agent_ids = safe_agent_ids_i64.to(torch::kInt32).contiguous();
    auto inactive_sort_key = torch::full_like(agent_ids_by_alive_order, agent_id_to_slot.size(0));
    auto sort_key = torch::where(active_mask, agent_ids_by_alive_order, inactive_sort_key);
    auto order = torch::argsort(sort_key).contiguous();
    auto died_mask = torch::zeros({capacity}, agent_ids.options().dtype(torch::kBool));
    auto food_gained_total = torch::zeros({capacity}, agent_ids.options().dtype(torch::kFloat32));
    auto moved_count = torch::zeros({capacity}, agent_ids.options().dtype(torch::kInt32));
    auto tick_moved = torch::zeros({capacity}, agent_ids.options().dtype(torch::kInt32));
    auto tick_food_gained = torch::zeros({capacity}, agent_ids.options().dtype(torch::kFloat32));
    auto metrics_float = torch::zeros({6}, agent_ids.options().dtype(torch::kFloat32));
    auto metrics_int = torch::zeros({2}, agent_ids.options().dtype(torch::kInt32));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    scripted_action_generation_kernel<<<1, 1, 0, stream>>>(
        agent_ids.data_ptr<int32_t>(),
        order.data_ptr<int64_t>(),
        safe_state_indices.data_ptr<int32_t>(),
        safe_state_indices.data_ptr<int32_t>(),
        actions_by_tick_slot.data_ptr<int32_t>(),
        pos_y.data_ptr<int32_t>(),
        pos_x.data_ptr<int32_t>(),
        agent_grid.data_ptr<int32_t>(),
        food_grid.data_ptr<int32_t>(),
        patch_food_amount.data_ptr<float>(),
        patch_scent_emit_pending.data_ptr<float>(),
        patch_waste_emit_pending.data_ptr<float>(),
        patch_movement_cost.data_ptr<float>(),
        energy.data_ptr<float>(),
        health.data_ptr<float>(),
        died_mask.data_ptr<bool>(),
        food_gained_total.data_ptr<float>(),
        moved_count.data_ptr<int32_t>(),
        tick_moved.data_ptr<int32_t>(),
        tick_food_gained.data_ptr<float>(),
        metrics_float.data_ptr<float>(),
        metrics_int.data_ptr<int32_t>(),
        actions_by_tick_slot.size(0),
        capacity,
        active_count.data_ptr<int64_t>(),
        actions_by_tick_slot.size(1),
        agent_grid.size(0),
        agent_grid.size(1),
        patch_enabled,
        move_cost,
        gather_cost,
        rest_cost,
        attack_cost,
        action_cost_mult,
        rest_cost_mult,
        max_energy,
        food_energy,
        patch_food_gather_amount,
        patch_food_presence_threshold,
        patch_scent_move_emit,
        patch_waste_move_emit,
        patch_scent_gather_emit,
        patch_waste_consume_emit,
        patch_movement_cost_apply_enable,
        patch_movement_cost_energy_scale
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {energy, died_mask, food_gained_total, moved_count, metrics_float, metrics_int, state_indices};
}

std::vector<torch::Tensor> eiel_cuda_scripted_owner_action_generation_slotted_v1(
    torch::Tensor agent_ids_by_alive_order,
    torch::Tensor active_slots_by_alive_order,
    torch::Tensor active_count,
    torch::Tensor actions_by_tick_slot,
    torch::Tensor pos_y,
    torch::Tensor pos_x,
    torch::Tensor agent_grid,
    torch::Tensor food_grid,
    torch::Tensor patch_food_amount,
    torch::Tensor patch_scent_emit_pending,
    torch::Tensor patch_waste_emit_pending,
    torch::Tensor patch_movement_cost,
    torch::Tensor energy,
    torch::Tensor health,
    const bool patch_enabled,
    const double move_cost,
    const double gather_cost,
    const double rest_cost,
    const double attack_cost,
    const double action_cost_mult,
    const double rest_cost_mult,
    const double max_energy,
    const double food_energy,
    const double patch_food_gather_amount,
    const double patch_food_presence_threshold,
    const double patch_scent_move_emit,
    const double patch_waste_move_emit,
    const double patch_scent_gather_emit,
    const double patch_waste_consume_emit,
    const bool patch_movement_cost_apply_enable,
    const double patch_movement_cost_energy_scale
) {
    for (const auto& named : std::vector<std::pair<const char*, torch::Tensor>>{
             {"agent_ids_by_alive_order", agent_ids_by_alive_order},
             {"active_slots_by_alive_order", active_slots_by_alive_order},
             {"active_count", active_count},
             {"actions_by_tick_slot", actions_by_tick_slot},
             {"pos_y", pos_y},
             {"pos_x", pos_x},
             {"agent_grid", agent_grid},
             {"food_grid", food_grid},
             {"patch_food_amount", patch_food_amount},
             {"patch_scent_emit_pending", patch_scent_emit_pending},
             {"patch_waste_emit_pending", patch_waste_emit_pending},
             {"patch_movement_cost", patch_movement_cost},
             {"energy", energy},
             {"health", health},
         }) {
        check_cuda_tensor(named.second, named.first);
        TORCH_CHECK(
            named.second.device() == agent_ids_by_alive_order.device(),
            named.first,
            " must share the owner CUDA device"
        );
    }
    TORCH_CHECK(
        agent_ids_by_alive_order.scalar_type() == torch::kInt64
            && agent_ids_by_alive_order.dim() == 1,
        "agent_ids_by_alive_order must be fixed-capacity int64 [capacity]"
    );
    TORCH_CHECK(
        active_slots_by_alive_order.scalar_type() == torch::kInt32
            && active_slots_by_alive_order.sizes()
                == agent_ids_by_alive_order.sizes(),
        "active_slots_by_alive_order must be aligned int32 [capacity]"
    );
    TORCH_CHECK(
        active_count.scalar_type() == torch::kInt64
            && active_count.dim() == 1
            && active_count.numel() == 1,
        "active_count must be CUDA int64[1]"
    );
    TORCH_CHECK(
        actions_by_tick_slot.scalar_type() == torch::kInt32
            && actions_by_tick_slot.dim() == 2,
        "actions_by_tick_slot must be int32 [ticks, slots]"
    );
    TORCH_CHECK(
        pos_y.scalar_type() == torch::kInt32
            && pos_x.scalar_type() == torch::kInt32,
        "positions must be int32"
    );
    TORCH_CHECK(
        pos_y.sizes() == energy.sizes()
            && pos_x.sizes() == energy.sizes()
            && actions_by_tick_slot.size(1) == energy.size(0),
        "action and position slot banks must align"
    );
    TORCH_CHECK(
        energy.scalar_type() == torch::kFloat32
            && health.scalar_type() == torch::kFloat32
            && energy.sizes() == health.sizes(),
        "energy and health must be aligned float32 tensors"
    );
    TORCH_CHECK(
        agent_grid.scalar_type() == torch::kInt32
            && food_grid.scalar_type() == torch::kInt32
            && agent_grid.dim() == 2
            && food_grid.sizes() == agent_grid.sizes(),
        "grids must be aligned int32 2D tensors"
    );
    for (const auto& tensor : {
             patch_food_amount,
             patch_scent_emit_pending,
             patch_waste_emit_pending,
             patch_movement_cost,
         }) {
        TORCH_CHECK(
            tensor.scalar_type() == torch::kFloat32
                && tensor.sizes() == agent_grid.sizes(),
            "Patch grids must be aligned float32 tensors"
        );
    }

    const int64_t capacity = agent_ids_by_alive_order.size(0);
    auto row_indices = torch::arange(capacity, agent_ids_by_alive_order.options());
    auto active_mask = row_indices < torch::clamp(active_count, 0, capacity);
    auto safe_agent_ids_i64 = torch::where(
        active_mask,
        agent_ids_by_alive_order,
        torch::zeros_like(agent_ids_by_alive_order)
    ).contiguous();
    auto safe_state_indices = torch::where(
        active_mask,
        active_slots_by_alive_order,
        torch::zeros_like(active_slots_by_alive_order)
    ).contiguous();
    auto state_indices = torch::where(
        active_mask,
        active_slots_by_alive_order,
        torch::full_like(active_slots_by_alive_order, -1)
    ).contiguous();
    auto agent_ids = safe_agent_ids_i64.to(torch::kInt32).contiguous();
    auto inactive_sort_key = torch::full_like(
        agent_ids_by_alive_order,
        std::numeric_limits<int64_t>::max()
    );
    auto sort_key = torch::where(
        active_mask, agent_ids_by_alive_order, inactive_sort_key
    );
    auto order = torch::argsort(sort_key).contiguous();
    auto died_mask = torch::zeros(
        {capacity}, agent_ids.options().dtype(torch::kBool)
    );
    auto food_gained_total = torch::zeros(
        {capacity}, agent_ids.options().dtype(torch::kFloat32)
    );
    auto moved_count = torch::zeros(
        {capacity}, agent_ids.options().dtype(torch::kInt32)
    );
    auto tick_moved = torch::zeros(
        {capacity}, agent_ids.options().dtype(torch::kInt32)
    );
    auto tick_food_gained = torch::zeros(
        {capacity}, agent_ids.options().dtype(torch::kFloat32)
    );
    auto metrics_float = torch::zeros(
        {6}, agent_ids.options().dtype(torch::kFloat32)
    );
    auto metrics_int = torch::zeros(
        {2}, agent_ids.options().dtype(torch::kInt32)
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    scripted_action_generation_kernel<<<1, 1, 0, stream>>>(
        agent_ids.data_ptr<int32_t>(),
        order.data_ptr<int64_t>(),
        safe_state_indices.data_ptr<int32_t>(),
        safe_state_indices.data_ptr<int32_t>(),
        actions_by_tick_slot.data_ptr<int32_t>(),
        pos_y.data_ptr<int32_t>(),
        pos_x.data_ptr<int32_t>(),
        agent_grid.data_ptr<int32_t>(),
        food_grid.data_ptr<int32_t>(),
        patch_food_amount.data_ptr<float>(),
        patch_scent_emit_pending.data_ptr<float>(),
        patch_waste_emit_pending.data_ptr<float>(),
        patch_movement_cost.data_ptr<float>(),
        energy.data_ptr<float>(),
        health.data_ptr<float>(),
        died_mask.data_ptr<bool>(),
        food_gained_total.data_ptr<float>(),
        moved_count.data_ptr<int32_t>(),
        tick_moved.data_ptr<int32_t>(),
        tick_food_gained.data_ptr<float>(),
        metrics_float.data_ptr<float>(),
        metrics_int.data_ptr<int32_t>(),
        actions_by_tick_slot.size(0),
        capacity,
        active_count.data_ptr<int64_t>(),
        actions_by_tick_slot.size(1),
        agent_grid.size(0),
        agent_grid.size(1),
        patch_enabled,
        move_cost,
        gather_cost,
        rest_cost,
        attack_cost,
        action_cost_mult,
        rest_cost_mult,
        max_energy,
        food_energy,
        patch_food_gather_amount,
        patch_food_presence_threshold,
        patch_scent_move_emit,
        patch_waste_move_emit,
        patch_scent_gather_emit,
        patch_waste_consume_emit,
        patch_movement_cost_apply_enable,
        patch_movement_cost_energy_scale
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {
        energy,
        died_mask,
        food_gained_total,
        moved_count,
        metrics_float,
        metrics_int,
        state_indices,
    };
}
