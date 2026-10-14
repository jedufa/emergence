#include <cstdint>
#include <cuda_runtime.h>

__global__ void cxxgen0_agent_summary_kernel(
    const float* pre_energy,
    const float* post_energy,
    const uint8_t* died_mask,
    const uint8_t* moved_mask,
    const float* food_gained,
    int64_t agents,
    double* delta_by_agent,
    double* died_by_agent,
    double* moved_by_agent,
    double* food_by_agent
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= agents) {
        return;
    }
    delta_by_agent[idx] = static_cast<double>(post_energy[idx] - pre_energy[idx]);
    died_by_agent[idx] = died_mask[idx] ? 1.0 : 0.0;
    moved_by_agent[idx] = moved_mask[idx] ? 1.0 : 0.0;
    food_by_agent[idx] = static_cast<double>(food_gained[idx]);
}

__global__ void cxxgen0_action_count_kernel(
    const int32_t* actions_by_tick,
    int64_t total_actions,
    double* action_counts
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_actions) {
        return;
    }
    const int32_t action = actions_by_tick[idx];
    if (action >= 0 && action < 7) {
        atomicAdd(&action_counts[action], 1.0);
    }
}

static int cxxgen0_cuda_fail(cudaError_t err) {
    return err == cudaSuccess ? 0 : 1000 + static_cast<int>(err);
}

extern "C" int cxxgen0_cuda_event_summary_v1(
    const int32_t* actions_by_tick,
    const float* pre_energy,
    const float* post_energy,
    const uint8_t* died_mask,
    const uint8_t* moved_mask,
    const float* food_gained,
    int64_t ticks,
    int64_t agents,
    double* out_summary
) {
    if (!actions_by_tick || !pre_energy || !post_energy || !died_mask || !moved_mask || !food_gained || !out_summary) {
        return 1;
    }
    if (ticks < 0 || agents < 0) {
        return 2;
    }

    const int64_t total_actions = ticks * agents;
    int32_t* d_actions = nullptr;
    float* d_pre = nullptr;
    float* d_post = nullptr;
    uint8_t* d_died = nullptr;
    uint8_t* d_moved = nullptr;
    float* d_food = nullptr;
    double* d_delta = nullptr;
    double* d_died_sum = nullptr;
    double* d_moved_sum = nullptr;
    double* d_food_sum = nullptr;
    double* d_action_counts = nullptr;

    cudaError_t err = cudaSuccess;
    if (total_actions > 0) {
        err = cudaMalloc(&d_actions, sizeof(int32_t) * static_cast<size_t>(total_actions));
        if (err != cudaSuccess) return cxxgen0_cuda_fail(err);
        err = cudaMemcpy(d_actions, actions_by_tick, sizeof(int32_t) * static_cast<size_t>(total_actions), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) return cxxgen0_cuda_fail(err);
    }
    if (agents > 0) {
        err = cudaMalloc(&d_pre, sizeof(float) * static_cast<size_t>(agents));
        if (err != cudaSuccess) return cxxgen0_cuda_fail(err);
        err = cudaMalloc(&d_post, sizeof(float) * static_cast<size_t>(agents));
        if (err != cudaSuccess) return cxxgen0_cuda_fail(err);
        err = cudaMalloc(&d_died, sizeof(uint8_t) * static_cast<size_t>(agents));
        if (err != cudaSuccess) return cxxgen0_cuda_fail(err);
        err = cudaMalloc(&d_moved, sizeof(uint8_t) * static_cast<size_t>(agents));
        if (err != cudaSuccess) return cxxgen0_cuda_fail(err);
        err = cudaMalloc(&d_food, sizeof(float) * static_cast<size_t>(agents));
        if (err != cudaSuccess) return cxxgen0_cuda_fail(err);
        err = cudaMalloc(&d_delta, sizeof(double) * static_cast<size_t>(agents));
        if (err != cudaSuccess) return cxxgen0_cuda_fail(err);
        err = cudaMalloc(&d_died_sum, sizeof(double) * static_cast<size_t>(agents));
        if (err != cudaSuccess) return cxxgen0_cuda_fail(err);
        err = cudaMalloc(&d_moved_sum, sizeof(double) * static_cast<size_t>(agents));
        if (err != cudaSuccess) return cxxgen0_cuda_fail(err);
        err = cudaMalloc(&d_food_sum, sizeof(double) * static_cast<size_t>(agents));
        if (err != cudaSuccess) return cxxgen0_cuda_fail(err);
        err = cudaMemcpy(d_pre, pre_energy, sizeof(float) * static_cast<size_t>(agents), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) return cxxgen0_cuda_fail(err);
        err = cudaMemcpy(d_post, post_energy, sizeof(float) * static_cast<size_t>(agents), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) return cxxgen0_cuda_fail(err);
        err = cudaMemcpy(d_died, died_mask, sizeof(uint8_t) * static_cast<size_t>(agents), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) return cxxgen0_cuda_fail(err);
        err = cudaMemcpy(d_moved, moved_mask, sizeof(uint8_t) * static_cast<size_t>(agents), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) return cxxgen0_cuda_fail(err);
        err = cudaMemcpy(d_food, food_gained, sizeof(float) * static_cast<size_t>(agents), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) return cxxgen0_cuda_fail(err);
    }
    err = cudaMalloc(&d_action_counts, sizeof(double) * 7);
    if (err != cudaSuccess) return cxxgen0_cuda_fail(err);
    err = cudaMemset(d_action_counts, 0, sizeof(double) * 7);
    if (err != cudaSuccess) return cxxgen0_cuda_fail(err);

    const int threads = 256;
    if (agents > 0) {
        const int blocks = static_cast<int>((agents + threads - 1) / threads);
        cxxgen0_agent_summary_kernel<<<blocks, threads>>>(
            d_pre, d_post, d_died, d_moved, d_food, agents, d_delta, d_died_sum, d_moved_sum, d_food_sum
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) return cxxgen0_cuda_fail(err);
    }
    if (total_actions > 0) {
        const int blocks = static_cast<int>((total_actions + threads - 1) / threads);
        cxxgen0_action_count_kernel<<<blocks, threads>>>(d_actions, total_actions, d_action_counts);
        err = cudaGetLastError();
        if (err != cudaSuccess) return cxxgen0_cuda_fail(err);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return cxxgen0_cuda_fail(err);

    double* h_delta = nullptr;
    double* h_died = nullptr;
    double* h_moved = nullptr;
    double* h_food = nullptr;
    if (agents > 0) {
        h_delta = new double[static_cast<size_t>(agents)];
        h_died = new double[static_cast<size_t>(agents)];
        h_moved = new double[static_cast<size_t>(agents)];
        h_food = new double[static_cast<size_t>(agents)];
        err = cudaMemcpy(h_delta, d_delta, sizeof(double) * static_cast<size_t>(agents), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return cxxgen0_cuda_fail(err);
        err = cudaMemcpy(h_died, d_died_sum, sizeof(double) * static_cast<size_t>(agents), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return cxxgen0_cuda_fail(err);
        err = cudaMemcpy(h_moved, d_moved_sum, sizeof(double) * static_cast<size_t>(agents), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return cxxgen0_cuda_fail(err);
        err = cudaMemcpy(h_food, d_food_sum, sizeof(double) * static_cast<size_t>(agents), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return cxxgen0_cuda_fail(err);
    }
    double h_counts[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    err = cudaMemcpy(h_counts, d_action_counts, sizeof(double) * 7, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return cxxgen0_cuda_fail(err);

    for (int i = 0; i < 11; ++i) {
        out_summary[i] = 0.0;
    }
    for (int64_t i = 0; i < agents; ++i) {
        out_summary[0] += h_delta[i];
        out_summary[1] += h_died[i];
        out_summary[2] += h_moved[i];
        out_summary[3] += h_food[i];
    }
    for (int i = 0; i < 7; ++i) {
        out_summary[4 + i] = h_counts[i];
    }

    delete[] h_delta;
    delete[] h_died;
    delete[] h_moved;
    delete[] h_food;
    cudaFree(d_actions);
    cudaFree(d_pre);
    cudaFree(d_post);
    cudaFree(d_died);
    cudaFree(d_moved);
    cudaFree(d_food);
    cudaFree(d_delta);
    cudaFree(d_died_sum);
    cudaFree(d_moved_sum);
    cudaFree(d_food_sum);
    cudaFree(d_action_counts);
    return 0;
}
