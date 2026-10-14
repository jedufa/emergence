#include "cuda_architecture_b_persistent_wdl_v1.cuh"
#include "wdl_native_contract.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cuda_wdl = eiel::cuda_architecture_b::persistent_wdl_v1;
namespace cpu_wdl = eiel::wdl::native_v1;

namespace {

void require(const bool condition, const char* const message) {
    if (!condition) {
        std::fprintf(stderr, "persistent_wdl_v1_probe failure: %s\n", message);
        std::exit(1);
    }
}

void require_cuda(const cudaError_t status, const char* const message) {
    if (status != cudaSuccess) {
        std::fprintf(
            stderr,
            "persistent_wdl_v1_probe CUDA failure: %s: %s\n",
            message,
            cudaGetErrorString(status)
        );
        std::exit(1);
    }
}

template <typename T>
T* managed(const std::int64_t count) {
    T* pointer = nullptr;
    require_cuda(
        cudaMallocManaged(&pointer, static_cast<std::size_t>(count) * sizeof(T)),
        "cudaMallocManaged"
    );
    return pointer;
}

__global__ void advance_kernel(
    const cuda_wdl::Config config,
    const cuda_wdl::StateView state,
    cuda_wdl::ScheduleState* const schedule,
    const std::int64_t absolute_step,
    cuda_wdl::TickMetrics* const metrics,
    cuda_wdl::ScheduledEventRow* const event,
    cuda_wdl::Status* const status
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *status = cuda_wdl::advance_tick_exact(
            config, state, schedule, absolute_step, false, metrics, event
        );
    }
}

__global__ void primitive_kernel(
    const cuda_wdl::Config config,
    const cuda_wdl::ConstStateView state,
    const std::int64_t seed,
    const std::int64_t event_index,
    const std::int32_t y,
    const std::int32_t x,
    float* const health,
    float* const health_delta,
    cuda_wdl::EventCoordinate* const coordinate,
    float* const observation,
    cuda_wdl::DamageResult* const damage
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *coordinate = cuda_wdl::stable_event_coordinate_exact(
            seed, event_index, config.height, config.width
        );
        *observation = cuda_wdl::cue_observation_value_exact(
            1.0F, config.observation_dropout_rate, seed, y, x
        );
        *damage = cuda_wdl::apply_damage_to_slot_exact(
            config, state, true, y, x, health, health_delta
        );
    }
}

bool same_float(const float left, const float right) {
    return std::memcmp(&left, &right, sizeof(float)) == 0;
}

}  // namespace

int main() {
    constexpr std::int32_t height = 5;
    constexpr std::int32_t width = 7;
    constexpr std::int64_t cells = static_cast<std::int64_t>(height) * width;

    cpu_wdl::Config cpu_config;
    cpu_config.height = height;
    cpu_config.width = width;
    cpu_config.delay_steps = 4;
    cpu_config.cue_duration_steps = 2;
    cpu_config.active_duration_steps = 3;
    cpu_config.event_period_steps = 12;
    cpu_config.radius = 1;
    cpu_config.intensity = 0.25F;
    cpu_config.observation_dropout_rate = 0.375F;
    cpu_config.enabled = 1;
    cpu_config.cue_enabled = 1;
    cpu_config.observation_enabled = 1;
    cpu_config.damage_enabled = 1;
    cpu_config.deterministic_schedule_enabled = 1;

    const cuda_wdl::Config cuda_config{
        cpu_config.schema_version,
        cpu_config.height,
        cpu_config.width,
        cpu_config.delay_steps,
        cpu_config.cue_duration_steps,
        cpu_config.active_duration_steps,
        cpu_config.event_period_steps,
        cpu_config.radius,
        cpu_config.intensity,
        cpu_config.observation_dropout_rate,
        cpu_config.enabled,
        cpu_config.cue_enabled,
        cpu_config.observation_enabled,
        cpu_config.damage_enabled,
        cpu_config.deterministic_schedule_enabled,
    };

    std::vector<float> cpu_cue(cells);
    std::vector<std::int32_t> cpu_timer(cells);
    std::vector<float> cpu_active(cells);
    std::vector<float> cpu_intensity(cells);
    std::vector<std::int32_t> cpu_age(cells);
    std::vector<std::int64_t> cpu_event_id(cells);
    std::vector<std::int8_t> cpu_prior_cue(cells);
    cpu_wdl::StateView cpu_state{
        cpu_cue.data(),
        cpu_timer.data(),
        cpu_active.data(),
        cpu_intensity.data(),
        cpu_age.data(),
        cpu_event_id.data(),
        cpu_prior_cue.data(),
        static_cast<std::size_t>(cells),
    };
    require(
        cpu_wdl::initialize_state(cpu_config, cpu_state) == cpu_wdl::Status::kOk,
        "CPU state initialization"
    );

    auto* cue = managed<float>(cells);
    auto* timer = managed<std::int32_t>(cells);
    auto* active = managed<float>(cells);
    auto* intensity = managed<float>(cells);
    auto* age = managed<std::int32_t>(cells);
    auto* event_id = managed<std::int64_t>(cells);
    auto* prior_cue = managed<std::int8_t>(cells);
    for (std::int64_t index = 0; index < cells; ++index) {
        cue[index] = 0.0F;
        timer[index] = -1;
        active[index] = 0.0F;
        intensity[index] = 0.0F;
        age[index] = 0;
        event_id[index] = 0;
        prior_cue[index] = 0;
    }
    const cuda_wdl::StateView cuda_state{
        cue, timer, active, intensity, age, event_id, prior_cue, cells,
    };
    auto* cuda_schedule = managed<cuda_wdl::ScheduleState>(1);
    auto* cuda_metrics = managed<cuda_wdl::TickMetrics>(1);
    auto* cuda_event = managed<cuda_wdl::ScheduledEventRow>(1);
    auto* cuda_status = managed<cuda_wdl::Status>(1);
    cpu_wdl::ScheduleState cpu_schedule{101, 0};
    *cuda_schedule = {101, 0};

    for (std::int64_t step = 0; step < 25; ++step) {
        cpu_wdl::TickMetrics cpu_metrics{};
        cpu_wdl::ScheduledEventRow cpu_event{};
        require(
            cpu_wdl::advance_tick(
                cpu_config,
                cpu_state,
                &cpu_schedule,
                step,
                false,
                &cpu_metrics,
                &cpu_event
            ) == cpu_wdl::Status::kOk,
            "CPU phase advance"
        );
        advance_kernel<<<1, 1>>>(
            cuda_config,
            cuda_state,
            cuda_schedule,
            step,
            cuda_metrics,
            cuda_event,
            cuda_status
        );
        require_cuda(cudaGetLastError(), "advance kernel launch");
        require_cuda(cudaDeviceSynchronize(), "advance kernel synchronization");
        require(*cuda_status == cuda_wdl::Status::kOk, "CUDA phase advance");
        require(cuda_schedule->next_abs_step == cpu_schedule.next_abs_step, "schedule");
        require(cuda_metrics->cue_cell_count == cpu_metrics.cue_cell_count, "cue cells");
        require(cuda_metrics->active_cell_count == cpu_metrics.active_cell_count, "active cells");
        require(cuda_metrics->event_count == cpu_metrics.event_count, "event count");
        require(
            cuda_metrics->cue_without_active_count
                == cpu_metrics.cue_without_active_count,
            "cue without active"
        );
        require(
            cuda_metrics->active_without_prior_cue_count
                == cpu_metrics.active_without_prior_cue_count,
            "active without prior cue"
        );
        require(
            cuda_metrics->cue_to_active_delay_sum
                == cpu_metrics.cue_to_active_delay_sum,
            "cue delay sum"
        );
        require(
            cuda_metrics->cue_to_active_delay_count
                == cpu_metrics.cue_to_active_delay_count,
            "cue delay count"
        );
        require(cuda_event->event_id == cpu_event.event_id, "scheduled event id");
        require(cuda_event->abs_step == cpu_event.abs_step, "scheduled event step");
        require(cuda_event->center_y == cpu_event.center_y, "scheduled event y");
        require(cuda_event->center_x == cpu_event.center_x, "scheduled event x");
        require(
            cuda_event->scheduled_cell_count == cpu_event.scheduled_cell_count,
            "scheduled event cell count"
        );
        require(same_float(cuda_event->intensity, cpu_event.intensity), "event intensity");
        for (std::int64_t index = 0; index < cells; ++index) {
            require(same_float(cue[index], cpu_cue[index]), "cue state");
            require(timer[index] == cpu_timer[index], "timer state");
            require(same_float(active[index], cpu_active[index]), "active state");
            require(same_float(intensity[index], cpu_intensity[index]), "intensity state");
            require(age[index] == cpu_age[index], "age state");
            require(event_id[index] == cpu_event_id[index], "event id state");
            require(prior_cue[index] == cpu_prior_cue[index], "prior cue state");
        }
    }

    const auto cpu_coordinate = cpu_wdl::stable_event_coordinate(202, 7, height, width);
    auto* cuda_coordinate = managed<cuda_wdl::EventCoordinate>(1);
    auto* cuda_observation = managed<float>(1);
    auto* cuda_damage = managed<cuda_wdl::DamageResult>(1);
    auto* cuda_health = managed<float>(1);
    auto* cuda_health_delta = managed<float>(1);
    *cuda_health = 0.20F;
    *cuda_health_delta = -0.01F;

    const std::int32_t damage_y = cpu_coordinate.y;
    const std::int32_t damage_x = cpu_coordinate.x;
    const std::int64_t damage_cell =
        static_cast<std::int64_t>(damage_y) * width + damage_x;
    active[damage_cell] = 1.0F;
    intensity[damage_cell] = 0.25F;
    const cuda_wdl::ConstStateView cuda_const_state{
        cue, timer, active, intensity, age, event_id, prior_cue, cells,
    };
    primitive_kernel<<<1, 1>>>(
        cuda_config,
        cuda_const_state,
        202,
        7,
        damage_y,
        damage_x,
        cuda_health,
        cuda_health_delta,
        cuda_coordinate,
        cuda_observation,
        cuda_damage
    );
    require_cuda(cudaGetLastError(), "primitive kernel launch");
    require_cuda(cudaDeviceSynchronize(), "primitive kernel synchronization");
    require(cuda_coordinate->y == cpu_coordinate.y, "coordinate y");
    require(cuda_coordinate->x == cpu_coordinate.x, "coordinate x");
    require(cuda_coordinate->event_id == cpu_coordinate.event_id, "coordinate event id");
    require(
        same_float(
            *cuda_observation,
            cpu_wdl::cue_observation_value(
                1.0F,
                cpu_config.observation_dropout_rate,
                202,
                damage_y,
                damage_x
            )
        ),
        "cue observation dropout"
    );
    require(cuda_damage->exposed, "damage exposure");
    require(cuda_damage->damaged, "damage applied");
    require(cuda_damage->lethal, "lethal damage");
    require(same_float(cuda_damage->damage, 0.25F), "damage amount");
    require(same_float(*cuda_health, 0.0F), "health settlement");
    require(same_float(*cuda_health_delta, -0.26F), "health delta settlement");

    require_cuda(cudaFree(cue), "free cue");
    require_cuda(cudaFree(timer), "free timer");
    require_cuda(cudaFree(active), "free active");
    require_cuda(cudaFree(intensity), "free intensity");
    require_cuda(cudaFree(age), "free age");
    require_cuda(cudaFree(event_id), "free event id");
    require_cuda(cudaFree(prior_cue), "free prior cue");
    require_cuda(cudaFree(cuda_schedule), "free schedule");
    require_cuda(cudaFree(cuda_metrics), "free metrics");
    require_cuda(cudaFree(cuda_event), "free event");
    require_cuda(cudaFree(cuda_status), "free status");
    require_cuda(cudaFree(cuda_coordinate), "free coordinate");
    require_cuda(cudaFree(cuda_observation), "free observation");
    require_cuda(cudaFree(cuda_damage), "free damage");
    require_cuda(cudaFree(cuda_health), "free health");
    require_cuda(cudaFree(cuda_health_delta), "free health delta");

    std::puts("persistent_wdl_v1_probe=ok ticks=25 coordinate=exact damage=exact");
    return 0;
}
