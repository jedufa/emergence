#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

extern "C" int cxxcore4_cost_layout_scalars(
    const double* rows,
    const std::int64_t n,
    const std::int64_t cols,
    const std::int32_t charge_channels,
    const std::int32_t charge_radius,
    const std::int32_t ledger_enabled,
    double* out
) {
    if (rows == nullptr || out == nullptr || n < 0 || cols != 8) {
        return 1;
    }
    if (ledger_enabled == 0) {
        for (std::int64_t i = 0; i < n; ++i) {
            out[i] = 0.0;
        }
        return 0;
    }
    for (std::int64_t i = 0; i < n; ++i) {
        const double* row = rows + (i * cols);
        double perception = 0.0;
        if (charge_channels != 0) {
            perception += row[0];
        }
        if (charge_radius != 0) {
            perception += row[1];
        }
        out[i] = perception + row[2] + row[3] + row[4] + row[5] + row[6] + row[7];
    }
    return 0;
}

static inline float layout_scalar_f32(
    const double* rows,
    const std::int64_t row,
    const std::int64_t cols,
    const std::int32_t charge_channels,
    const std::int32_t charge_radius,
    const std::int32_t ledger_enabled
) {
    if (ledger_enabled == 0) {
        return 0.0f;
    }
    const double* r = rows + (row * cols);
    double perception = 0.0;
    if (charge_channels != 0) {
        perception += r[0];
    }
    if (charge_radius != 0) {
        perception += r[1];
    }
    return static_cast<float>(perception + r[2] + r[3] + r[4] + r[5] + r[6] + r[7]);
}

extern "C" int cxxcore4_backend_run_with_cost_layout(
    const std::int32_t* agent_ids,
    const float* initial_energy,
    const float* initial_health,
    const float* initial_injury,
    const float* initial_scar,
    const std::int32_t* initial_pos_y_by_id,
    const std::int32_t* initial_pos_x_by_id,
    const std::int64_t pos_len,
    const std::int32_t* initial_agent_grid,
    const std::int8_t* initial_grid,
    const float* move_cost_mults,
    const float* juvenile_harvest_mults,
    const std::int32_t* actions_by_tick,
    const double* cost_layout_rows,
    const std::int64_t cost_layout_cols,
    const std::int32_t charge_channels,
    const std::int32_t charge_radius,
    const std::int32_t ledger_enabled,
    const std::int64_t ticks,
    const std::int64_t n,
    const std::int64_t height,
    const std::int64_t width,
    const float move_cost,
    const float gather_cost,
    const float rest_cost,
    const float attack_cost,
    const float action_cost_mult,
    const float rest_cost_mult,
    const float food_energy,
    const float max_energy,
    const std::int32_t export_final,
    float* out_energy,
    float* out_health,
    float* out_injury,
    float* out_scar,
    std::uint8_t* out_died,
    std::int32_t* out_pos_y,
    std::int32_t* out_pos_x,
    std::int8_t* out_grid,
    float* out_cost_total_by_agent,
    double* out_layout_cost_scalar_by_agent,
    double* out_summary
) {
    if (
        n < 0 || ticks < 0 || height <= 0 || width <= 0 || pos_len <= 0
        || cost_layout_cols != 8 || cost_layout_rows == nullptr
    ) {
        return 1;
    }
    const std::int64_t cells = height * width;
    std::vector<float> energy(initial_energy, initial_energy + n);
    std::vector<float> health(initial_health, initial_health + n);
    std::vector<float> injury(initial_injury, initial_injury + n);
    std::vector<float> scar(initial_scar, initial_scar + n);
    std::vector<float> cost_total(static_cast<std::size_t>(n), 0.0f);
    std::vector<float> layout_scalar(static_cast<std::size_t>(n), 0.0f);
    std::vector<std::uint8_t> died(static_cast<std::size_t>(n), 0);
    std::vector<std::int32_t> pos_y(initial_pos_y_by_id, initial_pos_y_by_id + pos_len);
    std::vector<std::int32_t> pos_x(initial_pos_x_by_id, initial_pos_x_by_id + pos_len);
    std::vector<std::int32_t> occ(initial_agent_grid, initial_agent_grid + cells);
    std::vector<std::int8_t> grid(initial_grid, initial_grid + cells);
    std::vector<std::int64_t> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::int64_t a, std::int64_t b) {
        return agent_ids[a] < agent_ids[b];
    });
    for (std::int64_t i = 0; i < n; ++i) {
        const std::int32_t aid = agent_ids[i];
        if (aid < 0 || static_cast<std::int64_t>(aid) >= pos_len) {
            return 2;
        }
        layout_scalar[static_cast<std::size_t>(i)] = layout_scalar_f32(
            cost_layout_rows,
            i,
            cost_layout_cols,
            charge_channels,
            charge_radius,
            ledger_enabled
        );
    }

    for (std::int64_t tick = 0; tick < ticks; ++tick) {
        std::vector<float> costs(static_cast<std::size_t>(n), 0.0f);
        std::vector<float> food_gained(static_cast<std::size_t>(n), 0.0f);
        const std::int64_t tick_offset = tick * n;
        for (std::int64_t oi = 0; oi < n; ++oi) {
            const std::int64_t src = order[static_cast<std::size_t>(oi)];
            const std::int32_t action = actions_by_tick[tick_offset + src];
            float cost = move_cost * move_cost_mults[src];
            if (action == 4) {
                cost = gather_cost;
            } else if (action == 5) {
                cost = rest_cost;
            } else if (action == 6) {
                cost = attack_cost;
            }
            if (action_cost_mult != 1.0f) {
                cost *= action_cost_mult;
            }
            if (rest_cost_mult != 1.0f && action == 5) {
                cost *= rest_cost_mult;
            }
            costs[static_cast<std::size_t>(src)] = cost;
        }
        for (std::int64_t oi = 0; oi < n; ++oi) {
            const std::int64_t src = order[static_cast<std::size_t>(oi)];
            const std::int32_t action = actions_by_tick[tick_offset + src];
            const std::int32_t aid = agent_ids[src];
            const std::int32_t y = pos_y[static_cast<std::size_t>(aid)];
            const std::int32_t x = pos_x[static_cast<std::size_t>(aid)];
            if (action >= 0 && action <= 3) {
                std::int32_t ny = y;
                std::int32_t nx = x;
                if (action == 0) {
                    ny -= 1;
                } else if (action == 1) {
                    ny += 1;
                } else if (action == 2) {
                    nx += 1;
                } else {
                    nx -= 1;
                }
                if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                    const std::int64_t old_idx = static_cast<std::int64_t>(y) * width + x;
                    const std::int64_t new_idx = static_cast<std::int64_t>(ny) * width + nx;
                    if (occ[static_cast<std::size_t>(new_idx)] == -1) {
                        if (occ[static_cast<std::size_t>(old_idx)] == aid) {
                            occ[static_cast<std::size_t>(old_idx)] = -1;
                        }
                        occ[static_cast<std::size_t>(new_idx)] = aid;
                        pos_y[static_cast<std::size_t>(aid)] = ny;
                        pos_x[static_cast<std::size_t>(aid)] = nx;
                    }
                }
            } else if (action == 4) {
                const std::int64_t idx = static_cast<std::int64_t>(y) * width + x;
                if (grid[static_cast<std::size_t>(idx)] == 1) {
                    grid[static_cast<std::size_t>(idx)] = 0;
                    food_gained[static_cast<std::size_t>(src)] = food_energy * juvenile_harvest_mults[src];
                }
            }
        }
        for (std::int64_t i = 0; i < n; ++i) {
            const float extra_cost = layout_scalar[static_cast<std::size_t>(i)];
            cost_total[static_cast<std::size_t>(i)] += extra_cost;
            float post = energy[static_cast<std::size_t>(i)] - costs[static_cast<std::size_t>(i)] + food_gained[static_cast<std::size_t>(i)] - extra_cost;
            if (post > max_energy) {
                post = max_energy;
            }
            if (post < 0.0f) {
                post = 0.0f;
            }
            energy[static_cast<std::size_t>(i)] = post;
            died[static_cast<std::size_t>(i)] = (post <= 0.0f || health[static_cast<std::size_t>(i)] <= 0.0f) ? 1 : 0;
        }
    }

    double energy_sum = 0.0;
    double energy_min = n > 0 ? std::numeric_limits<double>::infinity() : 0.0;
    double energy_max = n > 0 ? -std::numeric_limits<double>::infinity() : 0.0;
    double died_count = 0.0;
    double pos_checksum = 0.0;
    double grid_sum = 0.0;
    double cost_sum = 0.0;
    for (std::int64_t i = 0; i < n; ++i) {
        const std::int32_t aid = agent_ids[i];
        const double e = static_cast<double>(energy[static_cast<std::size_t>(i)]);
        energy_sum += e;
        if (e < energy_min) {
            energy_min = e;
        }
        if (e > energy_max) {
            energy_max = e;
        }
        died_count += died[static_cast<std::size_t>(i)] ? 1.0 : 0.0;
        pos_checksum += static_cast<double>(pos_y[static_cast<std::size_t>(aid)]) * 1009.0
            + static_cast<double>(pos_x[static_cast<std::size_t>(aid)]) * 917.0;
        cost_sum += static_cast<double>(cost_total[static_cast<std::size_t>(i)]);
    }
    for (std::int64_t i = 0; i < cells; ++i) {
        grid_sum += static_cast<double>(grid[static_cast<std::size_t>(i)]);
    }
    if (out_summary != nullptr) {
        out_summary[0] = energy_sum;
        out_summary[1] = energy_min;
        out_summary[2] = energy_max;
        out_summary[3] = died_count;
        out_summary[4] = pos_checksum;
        out_summary[5] = grid_sum;
        out_summary[6] = cost_sum;
    }
    if (out_layout_cost_scalar_by_agent != nullptr) {
        for (std::int64_t i = 0; i < n; ++i) {
            out_layout_cost_scalar_by_agent[i] = static_cast<double>(layout_scalar[static_cast<std::size_t>(i)]);
        }
    }
    if (export_final != 0) {
        for (std::int64_t i = 0; i < n; ++i) {
            const std::int32_t aid = agent_ids[i];
            out_energy[i] = energy[static_cast<std::size_t>(i)];
            out_health[i] = health[static_cast<std::size_t>(i)];
            out_injury[i] = injury[static_cast<std::size_t>(i)];
            out_scar[i] = scar[static_cast<std::size_t>(i)];
            out_died[i] = died[static_cast<std::size_t>(i)];
            out_pos_y[i] = pos_y[static_cast<std::size_t>(aid)];
            out_pos_x[i] = pos_x[static_cast<std::size_t>(aid)];
            out_cost_total_by_agent[i] = cost_total[static_cast<std::size_t>(i)];
        }
        for (std::int64_t i = 0; i < cells; ++i) {
            out_grid[i] = grid[static_cast<std::size_t>(i)];
        }
    }
    return 0;
}
