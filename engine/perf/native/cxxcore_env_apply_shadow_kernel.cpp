#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

extern "C" void cxxcore1_energy_delta(
    const float* pre_energy,
    const float* post_energy,
    const std::int64_t n,
    float* out_delta
) {
    for (std::int64_t i = 0; i < n; ++i) {
        out_delta[i] = post_energy[i] - pre_energy[i];
    }
}

extern "C" int cxxcore1_env_apply_no_rng(
    const std::int32_t* agent_ids,
    const std::int32_t* actions,
    const float* energies,
    const float* healths,
    const float* injuries,
    const float* scars,
    const std::int32_t* pos_y_by_id,
    const std::int32_t* pos_x_by_id,
    const std::int64_t pos_len,
    const std::int32_t* agent_grid,
    const std::int8_t* grid,
    const std::int64_t height,
    const std::int64_t width,
    const float* move_cost_mults,
    const float* juvenile_harvest_mults,
    const float move_cost,
    const float gather_cost,
    const float rest_cost,
    const float attack_cost,
    const float action_cost_mult,
    const float rest_cost_mult,
    const float food_energy,
    const float max_energy,
    const std::int64_t n,
    float* out_energy,
    float* out_health,
    float* out_injury,
    float* out_scar,
    float* out_food_gained,
    std::uint8_t* out_died,
    std::int32_t* out_pos_y,
    std::int32_t* out_pos_x,
    std::uint8_t* out_moved,
    float* out_energy_delta,
    float* out_health_delta,
    float* out_injury_delta,
    float* out_scar_delta
) {
    if (n < 0 || height <= 0 || width <= 0 || pos_len <= 0) {
        return 1;
    }

    std::vector<std::int64_t> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::int64_t a, std::int64_t b) {
        return agent_ids[a] < agent_ids[b];
    });

    const std::int64_t cells = height * width;
    std::vector<std::int32_t> occ(agent_grid, agent_grid + cells);
    std::vector<std::int8_t> food_grid(grid, grid + cells);
    std::vector<float> energy_o(static_cast<std::size_t>(n));
    std::vector<float> health_o(static_cast<std::size_t>(n));
    std::vector<float> injury_o(static_cast<std::size_t>(n));
    std::vector<float> scar_o(static_cast<std::size_t>(n));
    std::vector<float> food_o(static_cast<std::size_t>(n), 0.0f);
    std::vector<float> cost_o(static_cast<std::size_t>(n), 0.0f);
    std::vector<std::int32_t> ys(static_cast<std::size_t>(n));
    std::vector<std::int32_t> xs(static_cast<std::size_t>(n));
    std::vector<std::uint8_t> moved(static_cast<std::size_t>(n), 0);

    for (std::int64_t oi = 0; oi < n; ++oi) {
        const std::int64_t src = order[static_cast<std::size_t>(oi)];
        const std::int32_t aid = agent_ids[src];
        if (aid < 0 || static_cast<std::int64_t>(aid) >= pos_len) {
            return 2;
        }
        energy_o[static_cast<std::size_t>(oi)] = energies[src];
        health_o[static_cast<std::size_t>(oi)] = healths[src];
        injury_o[static_cast<std::size_t>(oi)] = injuries[src];
        scar_o[static_cast<std::size_t>(oi)] = scars[src];
        ys[static_cast<std::size_t>(oi)] = pos_y_by_id[aid];
        xs[static_cast<std::size_t>(oi)] = pos_x_by_id[aid];

        const std::int32_t act = actions[src];
        float cost = move_cost * move_cost_mults[src];
        if (act == 4) {
            cost = gather_cost;
        } else if (act == 5) {
            cost = rest_cost;
        } else if (act == 6) {
            cost = attack_cost;
        }
        if (action_cost_mult != 1.0f) {
            cost *= action_cost_mult;
        }
        if (rest_cost_mult != 1.0f && act == 5) {
            cost *= rest_cost_mult;
        }
        cost_o[static_cast<std::size_t>(oi)] = cost;
    }

    for (std::int64_t oi = 0; oi < n; ++oi) {
        const std::int64_t src = order[static_cast<std::size_t>(oi)];
        const std::int32_t act = actions[src];
        if (act < 0 || act > 3) {
            continue;
        }
        const std::int32_t aid = agent_ids[src];
        const std::int32_t y = ys[static_cast<std::size_t>(oi)];
        const std::int32_t x = xs[static_cast<std::size_t>(oi)];
        std::int32_t ny = y;
        std::int32_t nx = x;
        if (act == 0) {
            ny -= 1;
        } else if (act == 1) {
            ny += 1;
        } else if (act == 2) {
            nx += 1;
        } else {
            nx -= 1;
        }
        if (ny < 0 || ny >= height || nx < 0 || nx >= width) {
            continue;
        }
        const std::int64_t old_idx = static_cast<std::int64_t>(y) * width + x;
        const std::int64_t new_idx = static_cast<std::int64_t>(ny) * width + nx;
        if (occ[static_cast<std::size_t>(new_idx)] != -1) {
            continue;
        }
        if (occ[static_cast<std::size_t>(old_idx)] == aid) {
            occ[static_cast<std::size_t>(old_idx)] = -1;
        }
        occ[static_cast<std::size_t>(new_idx)] = aid;
        ys[static_cast<std::size_t>(oi)] = ny;
        xs[static_cast<std::size_t>(oi)] = nx;
        moved[static_cast<std::size_t>(oi)] = 1;
    }

    for (std::int64_t oi = 0; oi < n; ++oi) {
        const std::int64_t src = order[static_cast<std::size_t>(oi)];
        if (actions[src] != 4) {
            continue;
        }
        const std::int32_t y = ys[static_cast<std::size_t>(oi)];
        const std::int32_t x = xs[static_cast<std::size_t>(oi)];
        const std::int64_t idx = static_cast<std::int64_t>(y) * width + x;
        if (food_grid[static_cast<std::size_t>(idx)] == 1) {
            food_grid[static_cast<std::size_t>(idx)] = 0;
            food_o[static_cast<std::size_t>(oi)] = food_energy * juvenile_harvest_mults[src];
        }
    }

    for (std::int64_t oi = 0; oi < n; ++oi) {
        const std::int64_t src = order[static_cast<std::size_t>(oi)];
        float post_energy = energy_o[static_cast<std::size_t>(oi)] - cost_o[static_cast<std::size_t>(oi)] + food_o[static_cast<std::size_t>(oi)];
        if (post_energy > max_energy) {
            post_energy = max_energy;
        }
        if (post_energy < 0.0f) {
            post_energy = 0.0f;
        }
        out_energy[src] = post_energy;
        out_health[src] = health_o[static_cast<std::size_t>(oi)];
        out_injury[src] = injury_o[static_cast<std::size_t>(oi)];
        out_scar[src] = scar_o[static_cast<std::size_t>(oi)];
        out_food_gained[src] = food_o[static_cast<std::size_t>(oi)];
        out_died[src] = (post_energy <= 0.0f || out_health[src] <= 0.0f) ? 1 : 0;
        out_pos_y[src] = ys[static_cast<std::size_t>(oi)];
        out_pos_x[src] = xs[static_cast<std::size_t>(oi)];
        out_moved[src] = moved[static_cast<std::size_t>(oi)];
        out_energy_delta[src] = out_energy[src] - energies[src];
        out_health_delta[src] = out_health[src] - healths[src];
        out_injury_delta[src] = out_injury[src] - injuries[src];
        out_scar_delta[src] = out_scar[src] - scars[src];
    }
    return 0;
}
