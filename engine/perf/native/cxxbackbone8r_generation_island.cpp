#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>

namespace {

constexpr int32_t FOOD = 1;
constexpr int32_t EMPTY = 0;
constexpr int32_t ACT_GATHER = 4;
constexpr int32_t ACT_REST = 5;
constexpr int32_t ACT_ATTACK = 6;

float clamp01(const float value) {
    if (value < 0.0f) {
        return 0.0f;
    }
    if (value > 1.0f) {
        return 1.0f;
    }
    return value;
}

float safe_norm(const float value, const float denom) {
    if (denom <= 0.0f) {
        return 0.0f;
    }
    return clamp01(value / denom);
}

float movement_cost_norm(const float value, const float lo, const float hi) {
    const float span = hi - lo;
    if (span <= 0.0f) {
        return 0.0f;
    }
    return clamp01((value - lo) / span);
}

void check_cpu(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(!tensor.is_cuda(), name, " must be a CPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

std::vector<torch::Tensor> cxxbackbone8r_flat_apply_actions_cpu_v1(
    torch::Tensor agent_ids,
    torch::Tensor actions,
    torch::Tensor agent_pos_y,
    torch::Tensor agent_pos_x,
    torch::Tensor agent_grid,
    torch::Tensor food_grid,
    torch::Tensor energies,
    torch::Tensor healths,
    torch::Tensor injuries,
    torch::Tensor scars,
    const double move_cost,
    const double gather_cost,
    const double rest_cost,
    const double attack_cost,
    const double action_cost_mult,
    const double rest_cost_mult,
    const double max_energy,
    const double food_energy
) {
    check_cpu(agent_ids, "agent_ids");
    check_cpu(actions, "actions");
    check_cpu(agent_pos_y, "agent_pos_y");
    check_cpu(agent_pos_x, "agent_pos_x");
    check_cpu(agent_grid, "agent_grid");
    check_cpu(food_grid, "food_grid");
    check_cpu(energies, "energies");
    check_cpu(healths, "healths");
    check_cpu(injuries, "injuries");
    check_cpu(scars, "scars");
    TORCH_CHECK(agent_ids.scalar_type() == torch::kInt32, "agent_ids must be int32");
    TORCH_CHECK(actions.scalar_type() == torch::kInt32, "actions must be int32");
    TORCH_CHECK(agent_pos_y.scalar_type() == torch::kInt32, "agent_pos_y must be int32");
    TORCH_CHECK(agent_pos_x.scalar_type() == torch::kInt32, "agent_pos_x must be int32");
    TORCH_CHECK(agent_grid.scalar_type() == torch::kInt32, "agent_grid must be int32");
    TORCH_CHECK(food_grid.scalar_type() == torch::kInt32, "food_grid must be int32");
    TORCH_CHECK(energies.scalar_type() == torch::kFloat32, "energies must be float32");
    TORCH_CHECK(healths.scalar_type() == torch::kFloat32, "healths must be float32");
    TORCH_CHECK(injuries.scalar_type() == torch::kFloat32, "injuries must be float32");
    TORCH_CHECK(scars.scalar_type() == torch::kFloat32, "scars must be float32");
    TORCH_CHECK(agent_ids.dim() == 1, "agent_ids must be 1D");
    TORCH_CHECK(actions.dim() == 1 && actions.size(0) == agent_ids.size(0), "actions must align to agent_ids");
    TORCH_CHECK(agent_grid.dim() == 2 && food_grid.sizes() == agent_grid.sizes(), "grids must be aligned 2D tensors");

    const int64_t n = agent_ids.size(0);
    auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(agent_ids.device());
    auto f_opts = torch::TensorOptions().dtype(torch::kFloat32).device(agent_ids.device());
    auto i_opts = torch::TensorOptions().dtype(torch::kInt32).device(agent_ids.device());
    auto died_mask = torch::zeros({n}, bool_opts);
    auto new_energy = torch::empty({n}, f_opts);
    auto new_health = healths.clone();
    auto food_gained = torch::zeros({n}, f_opts);
    auto new_injury = injuries.clone();
    auto new_scar = scars.clone();
    auto moved_mask = torch::zeros({n}, i_opts);

    auto ids_a = agent_ids.accessor<int32_t, 1>();
    auto actions_a = actions.accessor<int32_t, 1>();
    auto pos_y_a = agent_pos_y.accessor<int32_t, 1>();
    auto pos_x_a = agent_pos_x.accessor<int32_t, 1>();
    auto grid_a = agent_grid.accessor<int32_t, 2>();
    auto food_a = food_grid.accessor<int32_t, 2>();
    auto energies_a = energies.accessor<float, 1>();
    auto healths_a = healths.accessor<float, 1>();
    auto costs = std::vector<float>(static_cast<size_t>(n), 0.0f);
    auto ys = std::vector<int32_t>(static_cast<size_t>(n), 0);
    auto xs = std::vector<int32_t>(static_cast<size_t>(n), 0);
    auto order = std::vector<int64_t>(static_cast<size_t>(n), 0);
    auto inv = std::vector<int64_t>(static_cast<size_t>(n), 0);
    for (int64_t i = 0; i < n; ++i) {
        order[static_cast<size_t>(i)] = i;
    }
    std::stable_sort(order.begin(), order.end(), [&](const int64_t left, const int64_t right) {
        return ids_a[left] < ids_a[right];
    });
    for (int64_t sorted_i = 0; sorted_i < n; ++sorted_i) {
        inv[static_cast<size_t>(order[static_cast<size_t>(sorted_i)])] = sorted_i;
    }

    const int64_t height = agent_grid.size(0);
    const int64_t width = agent_grid.size(1);
    auto moved_a = moved_mask.accessor<int32_t, 1>();
    auto gained_a = food_gained.accessor<float, 1>();

    for (int64_t sorted_i = 0; sorted_i < n; ++sorted_i) {
        const int64_t src_i = order[static_cast<size_t>(sorted_i)];
        const int32_t action = actions_a[src_i];
        float cost = static_cast<float>(move_cost);
        if (action == ACT_GATHER) {
            cost = static_cast<float>(gather_cost);
        } else if (action == ACT_REST) {
            cost = static_cast<float>(rest_cost);
        } else if (action == ACT_ATTACK) {
            cost = static_cast<float>(attack_cost);
        }
        if (action_cost_mult != 1.0) {
            cost = static_cast<float>(static_cast<double>(cost) * action_cost_mult);
        }
        if (rest_cost_mult != 1.0 && action == ACT_REST) {
            cost = static_cast<float>(static_cast<double>(cost) * rest_cost_mult);
        }
        costs[static_cast<size_t>(sorted_i)] = cost;
        const int32_t aid = ids_a[src_i];
        ys[static_cast<size_t>(sorted_i)] = pos_y_a[aid];
        xs[static_cast<size_t>(sorted_i)] = pos_x_a[aid];
    }

    for (int64_t sorted_i = 0; sorted_i < n; ++sorted_i) {
        const int64_t src_i = order[static_cast<size_t>(sorted_i)];
        const int32_t action = actions_a[src_i];
        if (action < 0 || action > 3) {
            continue;
        }
        const int32_t aid = ids_a[src_i];
        int32_t y = ys[static_cast<size_t>(sorted_i)];
        int32_t x = xs[static_cast<size_t>(sorted_i)];
        int32_t ny = y;
        int32_t nx = x;
        if (action == 0) {
            ny -= 1;
        } else if (action == 1) {
            ny += 1;
        } else if (action == 2) {
            nx += 1;
        } else {
            nx -= 1;
        }
        if (ny < 0 || ny >= height || nx < 0 || nx >= width) {
            continue;
        }
        if (grid_a[ny][nx] != -1) {
            continue;
        }
        if (grid_a[y][x] == aid) {
            grid_a[y][x] = -1;
        }
        grid_a[ny][nx] = aid;
        ys[static_cast<size_t>(sorted_i)] = ny;
        xs[static_cast<size_t>(sorted_i)] = nx;
        moved_a[src_i] = 1;
    }

    for (int64_t sorted_i = 0; sorted_i < n; ++sorted_i) {
        const int64_t src_i = order[static_cast<size_t>(sorted_i)];
        if (actions_a[src_i] != ACT_GATHER) {
            continue;
        }
        const int32_t y = ys[static_cast<size_t>(sorted_i)];
        const int32_t x = xs[static_cast<size_t>(sorted_i)];
        if (food_a[y][x] == FOOD) {
            food_a[y][x] = EMPTY;
            gained_a[src_i] = static_cast<float>(food_energy);
        }
    }

    auto new_energy_a = new_energy.accessor<float, 1>();
    auto died_a = died_mask.accessor<bool, 1>();
    auto new_health_a = new_health.accessor<float, 1>();
    for (int64_t sorted_i = 0; sorted_i < n; ++sorted_i) {
        const int64_t src_i = order[static_cast<size_t>(sorted_i)];
        float e = energies_a[src_i] - costs[static_cast<size_t>(sorted_i)] + gained_a[src_i];
        if (e > static_cast<float>(max_energy)) {
            e = static_cast<float>(max_energy);
        }
        if (e < 0.0f) {
            e = 0.0f;
        }
        new_energy_a[src_i] = e;
        died_a[src_i] = (e <= 0.0f || healths_a[src_i] <= 0.0f);
    }

    for (int64_t sorted_i = 0; sorted_i < n; ++sorted_i) {
        const int64_t src_i = order[static_cast<size_t>(sorted_i)];
        if (moved_a[src_i] != 0) {
            const int32_t aid = ids_a[src_i];
            pos_y_a[aid] = ys[static_cast<size_t>(sorted_i)];
            pos_x_a[aid] = xs[static_cast<size_t>(sorted_i)];
        }
    }

    (void)new_health_a;
    return {died_mask, new_energy, new_health, food_gained, new_injury, new_scar, moved_mask};
}

std::vector<torch::Tensor> cxxbackbone8r_patch_apply_actions_cpu_v1(
    torch::Tensor agent_ids,
    torch::Tensor actions,
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
    torch::Tensor injuries,
    torch::Tensor scars,
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
    check_cpu(agent_ids, "agent_ids");
    check_cpu(actions, "actions");
    check_cpu(agent_pos_y, "agent_pos_y");
    check_cpu(agent_pos_x, "agent_pos_x");
    check_cpu(agent_grid, "agent_grid");
    check_cpu(food_grid, "food_grid");
    check_cpu(patch_food_amount, "patch_food_amount");
    check_cpu(patch_scent_emit_pending, "patch_scent_emit_pending");
    check_cpu(patch_waste_emit_pending, "patch_waste_emit_pending");
    check_cpu(patch_movement_cost, "patch_movement_cost");
    check_cpu(energies, "energies");
    check_cpu(healths, "healths");
    check_cpu(injuries, "injuries");
    check_cpu(scars, "scars");
    TORCH_CHECK(agent_ids.scalar_type() == torch::kInt32, "agent_ids must be int32");
    TORCH_CHECK(actions.scalar_type() == torch::kInt32, "actions must be int32");
    TORCH_CHECK(agent_pos_y.scalar_type() == torch::kInt32, "agent_pos_y must be int32");
    TORCH_CHECK(agent_pos_x.scalar_type() == torch::kInt32, "agent_pos_x must be int32");
    TORCH_CHECK(agent_grid.scalar_type() == torch::kInt32, "agent_grid must be int32");
    TORCH_CHECK(food_grid.scalar_type() == torch::kInt32, "food_grid must be int32");
    TORCH_CHECK(patch_food_amount.scalar_type() == torch::kFloat32, "patch_food_amount must be float32");
    TORCH_CHECK(patch_scent_emit_pending.scalar_type() == torch::kFloat32, "patch_scent_emit_pending must be float32");
    TORCH_CHECK(patch_waste_emit_pending.scalar_type() == torch::kFloat32, "patch_waste_emit_pending must be float32");
    TORCH_CHECK(patch_movement_cost.scalar_type() == torch::kFloat32, "patch_movement_cost must be float32");
    TORCH_CHECK(energies.scalar_type() == torch::kFloat32, "energies must be float32");
    TORCH_CHECK(healths.scalar_type() == torch::kFloat32, "healths must be float32");
    TORCH_CHECK(injuries.scalar_type() == torch::kFloat32, "injuries must be float32");
    TORCH_CHECK(scars.scalar_type() == torch::kFloat32, "scars must be float32");
    TORCH_CHECK(agent_ids.dim() == 1, "agent_ids must be 1D");
    TORCH_CHECK(actions.dim() == 1 && actions.size(0) == agent_ids.size(0), "actions must align to agent_ids");
    TORCH_CHECK(agent_grid.dim() == 2 && food_grid.sizes() == agent_grid.sizes(), "grids must be aligned 2D tensors");
    TORCH_CHECK(patch_food_amount.sizes() == agent_grid.sizes(), "patch_food_amount must align to grid");
    TORCH_CHECK(patch_scent_emit_pending.sizes() == agent_grid.sizes(), "patch_scent_emit_pending must align to grid");
    TORCH_CHECK(patch_waste_emit_pending.sizes() == agent_grid.sizes(), "patch_waste_emit_pending must align to grid");
    TORCH_CHECK(patch_movement_cost.sizes() == agent_grid.sizes(), "patch_movement_cost must align to grid");

    const int64_t n = agent_ids.size(0);
    auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(agent_ids.device());
    auto f_opts = torch::TensorOptions().dtype(torch::kFloat32).device(agent_ids.device());
    auto i_opts = torch::TensorOptions().dtype(torch::kInt32).device(agent_ids.device());
    auto died_mask = torch::zeros({n}, bool_opts);
    auto new_energy = torch::empty({n}, f_opts);
    auto new_health = healths.clone();
    auto food_gained = torch::zeros({n}, f_opts);
    auto new_injury = injuries.clone();
    auto new_scar = scars.clone();
    auto moved_mask = torch::zeros({n}, i_opts);
    auto patch_food_harvest_total_t = torch::zeros({1}, f_opts);
    auto patch_scent_move_total_t = torch::zeros({1}, f_opts);
    auto patch_waste_move_total_t = torch::zeros({1}, f_opts);
    auto patch_scent_gather_total_t = torch::zeros({1}, f_opts);
    auto patch_waste_gather_total_t = torch::zeros({1}, f_opts);
    auto patch_movement_extra_t = torch::zeros({1}, f_opts);
    auto patch_movement_event_count_t = torch::zeros({1}, i_opts);

    auto ids_a = agent_ids.accessor<int32_t, 1>();
    auto actions_a = actions.accessor<int32_t, 1>();
    auto pos_y_a = agent_pos_y.accessor<int32_t, 1>();
    auto pos_x_a = agent_pos_x.accessor<int32_t, 1>();
    auto grid_a = agent_grid.accessor<int32_t, 2>();
    auto food_a = food_grid.accessor<int32_t, 2>();
    auto patch_food_a = patch_food_amount.accessor<float, 2>();
    auto scent_pending_a = patch_scent_emit_pending.accessor<float, 2>();
    auto waste_pending_a = patch_waste_emit_pending.accessor<float, 2>();
    auto movement_cost_a = patch_movement_cost.accessor<float, 2>();
    auto energies_a = energies.accessor<float, 1>();
    auto healths_a = healths.accessor<float, 1>();
    auto costs = std::vector<float>(static_cast<size_t>(n), 0.0f);
    auto ys = std::vector<int32_t>(static_cast<size_t>(n), 0);
    auto xs = std::vector<int32_t>(static_cast<size_t>(n), 0);
    auto order = std::vector<int64_t>(static_cast<size_t>(n), 0);
    for (int64_t i = 0; i < n; ++i) {
        order[static_cast<size_t>(i)] = i;
    }
    std::stable_sort(order.begin(), order.end(), [&](const int64_t left, const int64_t right) {
        return ids_a[left] < ids_a[right];
    });

    const int64_t height = agent_grid.size(0);
    const int64_t width = agent_grid.size(1);
    auto moved_a = moved_mask.accessor<int32_t, 1>();
    auto gained_a = food_gained.accessor<float, 1>();

    for (int64_t sorted_i = 0; sorted_i < n; ++sorted_i) {
        const int64_t src_i = order[static_cast<size_t>(sorted_i)];
        const int32_t action = actions_a[src_i];
        float cost = static_cast<float>(move_cost);
        if (action == ACT_GATHER) {
            cost = static_cast<float>(gather_cost);
        } else if (action == ACT_REST) {
            cost = static_cast<float>(rest_cost);
        } else if (action == ACT_ATTACK) {
            cost = static_cast<float>(attack_cost);
        }
        if (action_cost_mult != 1.0) {
            cost = static_cast<float>(static_cast<double>(cost) * action_cost_mult);
        }
        if (rest_cost_mult != 1.0 && action == ACT_REST) {
            cost = static_cast<float>(static_cast<double>(cost) * rest_cost_mult);
        }
        costs[static_cast<size_t>(sorted_i)] = cost;
        const int32_t aid = ids_a[src_i];
        ys[static_cast<size_t>(sorted_i)] = pos_y_a[aid];
        xs[static_cast<size_t>(sorted_i)] = pos_x_a[aid];
    }

    for (int64_t sorted_i = 0; sorted_i < n; ++sorted_i) {
        const int64_t src_i = order[static_cast<size_t>(sorted_i)];
        const int32_t action = actions_a[src_i];
        if (action < 0 || action > 3) {
            continue;
        }
        const int32_t aid = ids_a[src_i];
        int32_t y = ys[static_cast<size_t>(sorted_i)];
        int32_t x = xs[static_cast<size_t>(sorted_i)];
        int32_t ny = y;
        int32_t nx = x;
        if (action == 0) {
            ny -= 1;
        } else if (action == 1) {
            ny += 1;
        } else if (action == 2) {
            nx += 1;
        } else {
            nx -= 1;
        }
        if (ny < 0 || ny >= height || nx < 0 || nx >= width) {
            continue;
        }
        if (grid_a[ny][nx] != -1) {
            continue;
        }
        if (grid_a[y][x] == aid) {
            grid_a[y][x] = -1;
        }
        grid_a[ny][nx] = aid;
        ys[static_cast<size_t>(sorted_i)] = ny;
        xs[static_cast<size_t>(sorted_i)] = nx;
        moved_a[src_i] = 1;
    }

    float movement_extra = 0.0f;
    int32_t movement_events = 0;
    float scent_move_total = 0.0f;
    float waste_move_total = 0.0f;
    for (int64_t sorted_i = 0; sorted_i < n; ++sorted_i) {
        const int64_t src_i = order[static_cast<size_t>(sorted_i)];
        if (moved_a[src_i] == 0) {
            continue;
        }
        const int32_t y = ys[static_cast<size_t>(sorted_i)];
        const int32_t x = xs[static_cast<size_t>(sorted_i)];
        if (patch_movement_cost_apply_enable) {
            const float terrain_cost = movement_cost_a[y][x];
            const float mult = std::max(0.0f, 1.0f + static_cast<float>(patch_movement_cost_energy_scale) * (terrain_cost - 1.0f));
            const float old_cost = costs[static_cast<size_t>(sorted_i)];
            const float new_cost = std::max(0.0f, old_cost * mult);
            costs[static_cast<size_t>(sorted_i)] = new_cost;
            movement_extra += new_cost - old_cost;
            movement_events += 1;
        }
        if (patch_scent_move_emit > 0.0) {
            const float delta = static_cast<float>(patch_scent_move_emit);
            scent_pending_a[y][x] += delta;
            scent_move_total += delta;
        }
        if (patch_waste_move_emit > 0.0) {
            const float delta = static_cast<float>(patch_waste_move_emit);
            waste_pending_a[y][x] += delta;
            waste_move_total += delta;
        }
    }

    float gather_amount = static_cast<float>(patch_food_gather_amount);
    if (gather_amount <= 0.0f) {
        gather_amount = 1.0f;
    }
    const float threshold = static_cast<float>(patch_food_presence_threshold);
    float harvest_total = 0.0f;
    float scent_gather_total = 0.0f;
    float waste_gather_total = 0.0f;
    for (int64_t sorted_i = 0; sorted_i < n; ++sorted_i) {
        const int64_t src_i = order[static_cast<size_t>(sorted_i)];
        if (actions_a[src_i] != ACT_GATHER) {
            continue;
        }
        const int32_t y = ys[static_cast<size_t>(sorted_i)];
        const int32_t x = xs[static_cast<size_t>(sorted_i)];
        const float amount = patch_food_a[y][x];
        if (amount <= threshold) {
            food_a[y][x] = EMPTY;
            continue;
        }
        float harvested = amount;
        if (harvested > gather_amount) {
            harvested = gather_amount;
        }
        float post = amount - harvested;
        if (post < 0.0f) {
            post = 0.0f;
        }
        patch_food_a[y][x] = post;
        food_a[y][x] = (post > threshold) ? FOOD : EMPTY;
        const float harvest_scale = harvested / gather_amount;
        gained_a[src_i] = static_cast<float>(food_energy) * harvest_scale;
        if (patch_scent_gather_emit > 0.0) {
            const float delta = static_cast<float>(patch_scent_gather_emit) * harvest_scale;
            scent_pending_a[y][x] += delta;
            scent_gather_total += delta;
        }
        if (patch_waste_consume_emit > 0.0) {
            const float delta = static_cast<float>(patch_waste_consume_emit) * harvest_scale;
            waste_pending_a[y][x] += delta;
            waste_gather_total += delta;
        }
        harvest_total += harvested;
    }

    auto new_energy_a = new_energy.accessor<float, 1>();
    auto died_a = died_mask.accessor<bool, 1>();
    for (int64_t sorted_i = 0; sorted_i < n; ++sorted_i) {
        const int64_t src_i = order[static_cast<size_t>(sorted_i)];
        float e = energies_a[src_i] - costs[static_cast<size_t>(sorted_i)] + gained_a[src_i];
        if (e > static_cast<float>(max_energy)) {
            e = static_cast<float>(max_energy);
        }
        if (e < 0.0f) {
            e = 0.0f;
        }
        new_energy_a[src_i] = e;
        died_a[src_i] = (e <= 0.0f || healths_a[src_i] <= 0.0f);
    }

    for (int64_t sorted_i = 0; sorted_i < n; ++sorted_i) {
        const int64_t src_i = order[static_cast<size_t>(sorted_i)];
        if (moved_a[src_i] != 0) {
            const int32_t aid = ids_a[src_i];
            pos_y_a[aid] = ys[static_cast<size_t>(sorted_i)];
            pos_x_a[aid] = xs[static_cast<size_t>(sorted_i)];
        }
    }

    patch_food_harvest_total_t.accessor<float, 1>()[0] = harvest_total;
    patch_scent_move_total_t.accessor<float, 1>()[0] = scent_move_total;
    patch_waste_move_total_t.accessor<float, 1>()[0] = waste_move_total;
    patch_scent_gather_total_t.accessor<float, 1>()[0] = scent_gather_total;
    patch_waste_gather_total_t.accessor<float, 1>()[0] = waste_gather_total;
    patch_movement_extra_t.accessor<float, 1>()[0] = movement_extra;
    patch_movement_event_count_t.accessor<int32_t, 1>()[0] = movement_events;

    return {
        died_mask,
        new_energy,
        new_health,
        food_gained,
        new_injury,
        new_scar,
        moved_mask,
        patch_food_harvest_total_t,
        patch_scent_move_total_t,
        patch_waste_move_total_t,
        patch_scent_gather_total_t,
        patch_waste_gather_total_t,
        patch_movement_extra_t,
        patch_movement_event_count_t,
    };
}

torch::Tensor cxxbackbone8r_flat_observation_frame_cpu_v1(
    torch::Tensor agent_ids,
    torch::Tensor slots,
    torch::Tensor agent_pos_y,
    torch::Tensor agent_pos_x,
    torch::Tensor agent_grid,
    torch::Tensor food_grid,
    torch::Tensor energy_by_slot,
    torch::Tensor health_by_slot,
    const int64_t view_radius,
    const double max_energy,
    const double start_health
) {
    check_cpu(agent_ids, "agent_ids");
    check_cpu(slots, "slots");
    check_cpu(agent_pos_y, "agent_pos_y");
    check_cpu(agent_pos_x, "agent_pos_x");
    check_cpu(agent_grid, "agent_grid");
    check_cpu(food_grid, "food_grid");
    check_cpu(energy_by_slot, "energy_by_slot");
    check_cpu(health_by_slot, "health_by_slot");
    TORCH_CHECK(agent_ids.scalar_type() == torch::kInt32, "agent_ids must be int32");
    TORCH_CHECK(slots.scalar_type() == torch::kInt32, "slots must be int32");
    TORCH_CHECK(agent_pos_y.scalar_type() == torch::kInt32, "agent_pos_y must be int32");
    TORCH_CHECK(agent_pos_x.scalar_type() == torch::kInt32, "agent_pos_x must be int32");
    TORCH_CHECK(agent_grid.scalar_type() == torch::kInt32, "agent_grid must be int32");
    TORCH_CHECK(food_grid.scalar_type() == torch::kInt32, "food_grid must be int32");
    TORCH_CHECK(energy_by_slot.scalar_type() == torch::kFloat32, "energy_by_slot must be float32");
    TORCH_CHECK(health_by_slot.scalar_type() == torch::kFloat32, "health_by_slot must be float32");
    const int64_t n = agent_ids.size(0);
    TORCH_CHECK(slots.size(0) == n, "slots must align to agent_ids");
    const int64_t radius = std::max<int64_t>(0, view_radius);
    const int64_t k = 2 * radius + 1;
    const int64_t local_size = k * k;
    auto out = torch::empty({n, local_size + 2}, torch::TensorOptions().dtype(torch::kFloat32).device(agent_ids.device()));
    auto ids_a = agent_ids.accessor<int32_t, 1>();
    auto slots_a = slots.accessor<int32_t, 1>();
    auto pos_y_a = agent_pos_y.accessor<int32_t, 1>();
    auto pos_x_a = agent_pos_x.accessor<int32_t, 1>();
    auto grid_a = agent_grid.accessor<int32_t, 2>();
    auto food_a = food_grid.accessor<int32_t, 2>();
    auto energy_a = energy_by_slot.accessor<float, 1>();
    auto health_a = health_by_slot.accessor<float, 1>();
    auto out_a = out.accessor<float, 2>();
    const int64_t height = agent_grid.size(0);
    const int64_t width = agent_grid.size(1);
    const float max_e = static_cast<float>(max_energy == 0.0 ? 1.0 : max_energy);
    const float start_h = static_cast<float>(start_health == 0.0 ? 1.0 : start_health);
    for (int64_t i = 0; i < n; ++i) {
        const int32_t aid = ids_a[i];
        const int32_t slot = slots_a[i];
        const int32_t cy = pos_y_a[aid];
        const int32_t cx = pos_x_a[aid];
        int64_t t = 0;
        for (int64_t dy = -radius; dy <= radius; ++dy) {
            for (int64_t dx = -radius; dx <= radius; ++dx) {
                const int64_t y = static_cast<int64_t>(cy) + dy;
                const int64_t x = static_cast<int64_t>(cx) + dx;
                float value = 0.0f;
                if (y >= 0 && y < height && x >= 0 && x < width) {
                    if (grid_a[y][x] != -1) {
                        value = -1.0f;
                    } else if (food_a[y][x] == FOOD) {
                        value = 1.0f;
                    }
                }
                out_a[i][t] = value;
                ++t;
            }
        }
        out_a[i][local_size] = energy_a[slot] / max_e;
        out_a[i][local_size + 1] = health_a[slot] / start_h;
    }
    return out;
}

torch::Tensor cxxbackbone8r_patch_observation_frame_cpu_v1(
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
    check_cpu(agent_ids, "agent_ids");
    check_cpu(slots, "slots");
    check_cpu(agent_pos_y, "agent_pos_y");
    check_cpu(agent_pos_x, "agent_pos_x");
    check_cpu(agent_grid, "agent_grid");
    check_cpu(food_grid, "food_grid");
    check_cpu(energy_by_slot, "energy_by_slot");
    check_cpu(health_by_slot, "health_by_slot");
    check_cpu(patch_food_amount, "patch_food_amount");
    check_cpu(patch_cover, "patch_cover");
    check_cpu(patch_shelter, "patch_shelter");
    check_cpu(patch_scent, "patch_scent");
    check_cpu(patch_waste, "patch_waste");
    check_cpu(patch_movement_cost, "patch_movement_cost");
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
    TORCH_CHECK(slots.size(0) == n, "slots must align to agent_ids");
    const int64_t radius = std::max<int64_t>(0, view_radius);
    const int64_t k = 2 * radius + 1;
    const int64_t local_size = k * k;
    const int64_t channel_count = 8;
    auto out = torch::empty(
        {n, local_size * channel_count + 2},
        torch::TensorOptions().dtype(torch::kFloat32).device(agent_ids.device())
    );

    auto ids_a = agent_ids.accessor<int32_t, 1>();
    auto slots_a = slots.accessor<int32_t, 1>();
    auto pos_y_a = agent_pos_y.accessor<int32_t, 1>();
    auto pos_x_a = agent_pos_x.accessor<int32_t, 1>();
    auto grid_a = agent_grid.accessor<int32_t, 2>();
    auto food_a = food_grid.accessor<int32_t, 2>();
    auto energy_a = energy_by_slot.accessor<float, 1>();
    auto health_a = health_by_slot.accessor<float, 1>();
    auto food_amount_a = patch_food_amount.accessor<float, 2>();
    auto cover_a = patch_cover.accessor<float, 2>();
    auto shelter_a = patch_shelter.accessor<float, 2>();
    auto scent_a = patch_scent.accessor<float, 2>();
    auto waste_a = patch_waste.accessor<float, 2>();
    auto movement_a = patch_movement_cost.accessor<float, 2>();
    auto out_a = out.accessor<float, 2>();
    const int64_t height = agent_grid.size(0);
    const int64_t width = agent_grid.size(1);

    const int64_t density_radius = std::max<int64_t>(0, patch_density_radius);
    auto density = std::vector<float>(static_cast<size_t>(height * width), 0.0f);
    for (int64_t y = 0; y < height; ++y) {
        for (int64_t x = 0; x < width; ++x) {
            if (grid_a[y][x] == -1) {
                continue;
            }
            const int64_t y0 = std::max<int64_t>(0, y - density_radius);
            const int64_t y1 = std::min<int64_t>(height, y + density_radius + 1);
            const int64_t x0 = std::max<int64_t>(0, x - density_radius);
            const int64_t x1 = std::min<int64_t>(width, x + density_radius + 1);
            for (int64_t yy = y0; yy < y1; ++yy) {
                for (int64_t xx = x0; xx < x1; ++xx) {
                    density[static_cast<size_t>(yy * width + xx)] += 1.0f;
                }
            }
            if (!patch_density_include_self) {
                density[static_cast<size_t>(y * width + x)] -= 1.0f;
            }
        }
    }

    float density_denom = static_cast<float>((2 * density_radius + 1) * (2 * density_radius + 1));
    if (!patch_density_include_self) {
        density_denom -= 1.0f;
    }
    if (density_denom <= 0.0f) {
        density_denom = 1.0f;
    }
    const float max_e = static_cast<float>(max_energy == 0.0 ? 1.0 : max_energy);
    const float start_h = static_cast<float>(start_health == 0.0 ? 1.0 : start_health);
    const float food_max = static_cast<float>(patch_food_amount_max);
    const float scent_max = static_cast<float>(patch_scent_max);
    const float waste_max = static_cast<float>(patch_waste_max);
    const float move_min = static_cast<float>(patch_movement_cost_min);
    const float move_max = static_cast<float>(patch_movement_cost_max);

    for (int64_t i = 0; i < n; ++i) {
        const int32_t aid = ids_a[i];
        const int32_t slot = slots_a[i];
        const int32_t cy = pos_y_a[aid];
        const int32_t cx = pos_x_a[aid];
        int64_t t = 0;
        for (int64_t dy = -radius; dy <= radius; ++dy) {
            for (int64_t dx = -radius; dx <= radius; ++dx) {
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
                    if (grid_a[y][x] != -1) {
                        legacy = -1.0f;
                    } else if (food_a[y][x] == FOOD) {
                        legacy = 1.0f;
                    }
                    food_norm = safe_norm(food_amount_a[y][x], food_max);
                    cover = clamp01(cover_a[y][x]);
                    shelter = clamp01(shelter_a[y][x]);
                    scent_norm = safe_norm(scent_a[y][x], scent_max);
                    waste_norm = safe_norm(waste_a[y][x], waste_max);
                    const float density_count = density[static_cast<size_t>(y * width + x)];
                    density_norm = patch_density_normalize ? clamp01(density_count) : safe_norm(density_count, density_denom);
                    movement_norm = movement_cost_norm(movement_a[y][x], move_min, move_max);
                }
                out_a[i][t] = legacy;
                out_a[i][local_size + t] = food_norm;
                out_a[i][2 * local_size + t] = cover;
                out_a[i][3 * local_size + t] = shelter;
                out_a[i][4 * local_size + t] = scent_norm;
                out_a[i][5 * local_size + t] = waste_norm;
                out_a[i][6 * local_size + t] = density_norm;
                out_a[i][7 * local_size + t] = movement_norm;
                ++t;
            }
        }
        out_a[i][channel_count * local_size] = energy_a[slot] / max_e;
        out_a[i][channel_count * local_size + 1] = health_a[slot] / start_h;
    }
    return out;
}

std::vector<torch::Tensor> cxxbackbone8r_group_rows_cpu_v1(
    torch::Tensor agent_ids,
    torch::Tensor agent_id_to_slot,
    torch::Tensor use_gru
) {
    check_cpu(agent_ids, "agent_ids");
    check_cpu(agent_id_to_slot, "agent_id_to_slot");
    check_cpu(use_gru, "use_gru");
    TORCH_CHECK(agent_ids.scalar_type() == torch::kInt32, "agent_ids must be int32");
    TORCH_CHECK(agent_id_to_slot.scalar_type() == torch::kInt32, "agent_id_to_slot must be int32");
    TORCH_CHECK(use_gru.scalar_type() == torch::kBool, "use_gru must be bool");
    const int64_t n = agent_ids.size(0);
    auto slots = torch::empty({n}, torch::TensorOptions().dtype(torch::kInt32).device(agent_ids.device()));
    auto ids_a = agent_ids.accessor<int32_t, 1>();
    auto map_a = agent_id_to_slot.accessor<int32_t, 1>();
    auto gru_a = use_gru.accessor<bool, 1>();
    auto slots_a = slots.accessor<int32_t, 1>();
    std::vector<int64_t> mlp;
    std::vector<int64_t> gru;
    mlp.reserve(static_cast<size_t>(n));
    gru.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const int32_t aid = ids_a[i];
        TORCH_CHECK(aid >= 0 && aid < agent_id_to_slot.size(0), "agent id out of id-to-slot map range");
        const int32_t slot = map_a[aid];
        TORCH_CHECK(slot >= 0 && slot < use_gru.size(0), "agent id maps to invalid slot");
        slots_a[i] = slot;
        if (gru_a[slot]) {
            gru.push_back(i);
        } else {
            mlp.push_back(i);
        }
    }
    auto idx_opts = torch::TensorOptions().dtype(torch::kInt64).device(agent_ids.device());
    auto mlp_idx = torch::empty({static_cast<int64_t>(mlp.size())}, idx_opts);
    auto gru_idx = torch::empty({static_cast<int64_t>(gru.size())}, idx_opts);
    auto mlp_a = mlp_idx.accessor<int64_t, 1>();
    auto gru_idx_a = gru_idx.accessor<int64_t, 1>();
    for (int64_t i = 0; i < static_cast<int64_t>(mlp.size()); ++i) {
        mlp_a[i] = mlp[static_cast<size_t>(i)];
    }
    for (int64_t i = 0; i < static_cast<int64_t>(gru.size()); ++i) {
        gru_idx_a[i] = gru[static_cast<size_t>(i)];
    }
    return {slots, mlp_idx, gru_idx};
}
