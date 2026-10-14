#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

static inline float cxxwave1h_tanh(float value) {
    return std::tanh(value);
}

static inline void cxxwave1h_write_summary(
    const std::int32_t* agent_ids,
    const std::int64_t n,
    const std::int64_t height,
    const std::int64_t width,
    const std::vector<float>& energy,
    const std::vector<std::uint8_t>& died,
    const std::vector<std::int32_t>& pos_y,
    const std::vector<std::int32_t>& pos_x,
    const std::vector<std::int8_t>& grid,
    const std::vector<float>& cost_total,
    double* out
) {
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
    const std::int64_t cells = height * width;
    for (std::int64_t i = 0; i < cells; ++i) {
        grid_sum += static_cast<double>(grid[static_cast<std::size_t>(i)]);
    }
    out[0] = energy_sum;
    out[1] = energy_min;
    out[2] = energy_max;
    out[3] = died_count;
    out[4] = pos_checksum;
    out[5] = grid_sum;
    out[6] = cost_sum;
}

extern "C" int cxxwave1h_flat_mlp_generation_v1(
    const std::int32_t* agent_ids,
    const float* initial_energy,
    const float* initial_health,
    const std::int32_t* initial_pos_y_by_id,
    const std::int32_t* initial_pos_x_by_id,
    const std::int64_t pos_len,
    const std::int32_t* initial_agent_grid,
    const std::int8_t* initial_grid,
    const float* mlp_pack,
    const std::int64_t pack_stride,
    const float* drive_rows,
    const float* gain,
    const float* aggression,
    const float* temperature,
    const float* move_cost_mults,
    const float* juvenile_harvest_mults,
    const float* scalar_costs,
    const std::int32_t* action_cache_cursor,
    const std::int64_t cache_stride,
    const std::int64_t ticks,
    const std::int64_t n,
    const std::int64_t height,
    const std::int64_t width,
    const std::int64_t view_radius,
    const std::int64_t hidden,
    const std::int64_t obs_cols,
    const std::int64_t actions,
    const float move_cost,
    const float gather_cost,
    const float rest_cost,
    const float attack_cost,
    const float action_cost_mult,
    const float rest_cost_mult,
    const float food_energy,
    const float max_energy,
    const float w_hunger_gather,
    const float w_pain_rest,
    const float w_fatigue_rest,
    const float w_pain_attack,
    const float w_fear_attack,
    float* out_energy,
    float* out_health,
    std::int32_t* out_pos_y,
    std::int32_t* out_pos_x,
    std::int8_t* out_grid,
    std::int32_t* out_agent_grid,
    float* out_cost_total_by_agent,
    float* out_food_gained,
    std::uint8_t* out_moved,
    std::uint8_t* out_died,
    std::uint8_t* out_views_dirty,
    std::int32_t* out_actions_by_tick,
    float* out_logits_by_tick,
    std::int32_t* out_action_cache_cursor,
    double* out_final_summary
) {
    if (
        !agent_ids || !initial_energy || !initial_health || !initial_pos_y_by_id || !initial_pos_x_by_id
        || !initial_agent_grid || !initial_grid || !mlp_pack || !drive_rows || !gain || !aggression || !temperature
        || !move_cost_mults || !juvenile_harvest_mults || !scalar_costs || !action_cache_cursor
        || !out_energy || !out_health || !out_pos_y || !out_pos_x || !out_grid || !out_agent_grid
        || !out_cost_total_by_agent || !out_food_gained || !out_moved || !out_died || !out_views_dirty
        || !out_actions_by_tick || !out_logits_by_tick || !out_action_cache_cursor || !out_final_summary
    ) {
        return 1;
    }
    if (
        n < 0 || ticks < 0 || height <= 0 || width <= 0 || pos_len <= 0
        || view_radius < 0 || hidden <= 0 || obs_cols <= 0 || actions != 7
    ) {
        return 2;
    }
    const std::int64_t expected_obs = (2 * view_radius + 1) * (2 * view_radius + 1);
    if (obs_cols != expected_obs) {
        return 3;
    }
    const std::int64_t expected_pack = hidden * obs_cols + hidden + hidden * hidden + hidden + actions * hidden + actions;
    if (pack_stride != expected_pack) {
        return 4;
    }

    const std::int64_t cells = height * width;
    std::vector<float> energy(initial_energy, initial_energy + n);
    std::vector<float> health(initial_health, initial_health + n);
    std::vector<float> cost_total(static_cast<std::size_t>(n), 0.0f);
    std::vector<float> food_total(static_cast<std::size_t>(n), 0.0f);
    std::vector<std::uint8_t> moved(static_cast<std::size_t>(n), 0);
    std::vector<std::uint8_t> died(static_cast<std::size_t>(n), 0);
    std::vector<std::int32_t> cursor(action_cache_cursor, action_cache_cursor + n);
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
            return 5;
        }
    }

    const std::int64_t w1_off = 0;
    const std::int64_t b1_off = w1_off + hidden * obs_cols;
    const std::int64_t w2_off = b1_off + hidden;
    const std::int64_t b2_off = w2_off + hidden * hidden;
    const std::int64_t w3_off = b2_off + hidden;
    const std::int64_t b3_off = w3_off + actions * hidden;
    std::vector<float> obs(static_cast<std::size_t>(obs_cols), 0.0f);
    std::vector<float> h1(static_cast<std::size_t>(hidden), 0.0f);
    std::vector<float> h2(static_cast<std::size_t>(hidden), 0.0f);
    std::vector<float> logits(static_cast<std::size_t>(actions), 0.0f);

    for (std::int64_t tick = 0; tick < ticks; ++tick) {
        std::vector<std::int32_t> tick_actions(static_cast<std::size_t>(n), 0);
        std::vector<float> costs(static_cast<std::size_t>(n), 0.0f);
        for (std::int64_t oi = 0; oi < n; ++oi) {
            const std::int64_t src = order[static_cast<std::size_t>(oi)];
            const std::int32_t aid = agent_ids[src];
            const std::int32_t cy = pos_y[static_cast<std::size_t>(aid)];
            const std::int32_t cx = pos_x[static_cast<std::size_t>(aid)];
            std::int64_t col = 0;
            for (std::int64_t dy = -view_radius; dy <= view_radius; ++dy) {
                for (std::int64_t dx = -view_radius; dx <= view_radius; ++dx) {
                    const std::int64_t yy = static_cast<std::int64_t>(cy) + dy;
                    const std::int64_t xx = static_cast<std::int64_t>(cx) + dx;
                    float value = 0.0f;
                    if (yy >= 0 && yy < height && xx >= 0 && xx < width) {
                        const std::int64_t cell = yy * width + xx;
                        value = occ[static_cast<std::size_t>(cell)] != -1
                            ? -1.0f
                            : static_cast<float>(grid[static_cast<std::size_t>(cell)]);
                    }
                    obs[static_cast<std::size_t>(col)] = value;
                    ++col;
                }
            }

            const float* pack = mlp_pack + src * pack_stride;
            for (std::int64_t h = 0; h < hidden; ++h) {
                float acc = pack[b1_off + h];
                for (std::int64_t c = 0; c < obs_cols; ++c) {
                    acc = static_cast<float>(acc + static_cast<float>(pack[w1_off + h * obs_cols + c] * obs[static_cast<std::size_t>(c)]));
                }
                h1[static_cast<std::size_t>(h)] = cxxwave1h_tanh(acc);
            }
            for (std::int64_t h = 0; h < hidden; ++h) {
                float acc = pack[b2_off + h];
                for (std::int64_t c = 0; c < hidden; ++c) {
                    acc = static_cast<float>(acc + static_cast<float>(pack[w2_off + h * hidden + c] * h1[static_cast<std::size_t>(c)]));
                }
                h2[static_cast<std::size_t>(h)] = cxxwave1h_tanh(acc);
            }
            for (std::int64_t a = 0; a < actions; ++a) {
                float acc = pack[b3_off + a];
                for (std::int64_t h = 0; h < hidden; ++h) {
                    acc = static_cast<float>(acc + static_cast<float>(pack[w3_off + a * hidden + h] * h2[static_cast<std::size_t>(h)]));
                }
                logits[static_cast<std::size_t>(a)] = acc;
            }
            logits[6] += aggression[src];
            const float temp = std::max(temperature[src], 0.05f);
            for (std::int64_t a = 0; a < actions; ++a) {
                logits[static_cast<std::size_t>(a)] = logits[static_cast<std::size_t>(a)] / temp;
            }
            const float* drive = drive_rows + src * 4;
            logits[4] += gain[src] * (w_hunger_gather * drive[0]);
            logits[5] += gain[src] * (w_pain_rest * drive[1] + w_fatigue_rest * drive[2]);
            logits[6] -= gain[src] * (w_pain_attack * drive[1] + w_fear_attack * drive[3]);

            std::int32_t best = 0;
            float best_value = logits[0];
            const std::int64_t out_offset = (tick * n + src) * actions;
            for (std::int64_t a = 0; a < actions; ++a) {
                const float value = logits[static_cast<std::size_t>(a)];
                out_logits_by_tick[out_offset + a] = value;
                if (value > best_value) {
                    best_value = value;
                    best = static_cast<std::int32_t>(a);
                }
            }
            tick_actions[static_cast<std::size_t>(src)] = best;
            out_actions_by_tick[tick * n + src] = best;
            if (cache_stride > 0) {
                cursor[static_cast<std::size_t>(src)] = static_cast<std::int32_t>(
                    (static_cast<std::int64_t>(cursor[static_cast<std::size_t>(src)]) + 1) % cache_stride
                );
            } else {
                cursor[static_cast<std::size_t>(src)] += 1;
            }

            float cost = move_cost * move_cost_mults[src];
            if (best == 4) {
                cost = gather_cost;
            } else if (best == 5) {
                cost = rest_cost;
            } else if (best == 6) {
                cost = attack_cost;
            }
            if (action_cost_mult != 1.0f) {
                cost *= action_cost_mult;
            }
            if (rest_cost_mult != 1.0f && best == 5) {
                cost *= rest_cost_mult;
            }
            costs[static_cast<std::size_t>(src)] = cost;
        }

        std::vector<float> food_gained(static_cast<std::size_t>(n), 0.0f);
        for (std::int64_t oi = 0; oi < n; ++oi) {
            const std::int64_t src = order[static_cast<std::size_t>(oi)];
            const std::int32_t action = tick_actions[static_cast<std::size_t>(src)];
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
                        moved[static_cast<std::size_t>(src)] = 1;
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
            const float extra_cost = scalar_costs[i];
            cost_total[static_cast<std::size_t>(i)] += extra_cost;
            food_total[static_cast<std::size_t>(i)] += food_gained[static_cast<std::size_t>(i)];
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

    cxxwave1h_write_summary(agent_ids, n, height, width, energy, died, pos_y, pos_x, grid, cost_total, out_final_summary);
    for (std::int64_t i = 0; i < n; ++i) {
        const std::int32_t aid = agent_ids[i];
        out_energy[i] = energy[static_cast<std::size_t>(i)];
        out_health[i] = health[static_cast<std::size_t>(i)];
        out_pos_y[i] = pos_y[static_cast<std::size_t>(aid)];
        out_pos_x[i] = pos_x[static_cast<std::size_t>(aid)];
        out_cost_total_by_agent[i] = cost_total[static_cast<std::size_t>(i)];
        out_food_gained[i] = food_total[static_cast<std::size_t>(i)];
        out_moved[i] = moved[static_cast<std::size_t>(i)];
        out_died[i] = died[static_cast<std::size_t>(i)];
        out_action_cache_cursor[i] = cursor[static_cast<std::size_t>(i)];
    }
    for (std::int64_t i = 0; i < cells; ++i) {
        out_grid[i] = grid[static_cast<std::size_t>(i)];
        out_agent_grid[i] = occ[static_cast<std::size_t>(i)];
    }
    std::uint8_t views_dirty = 0;
    for (std::int64_t i = 0; i < n; ++i) {
        if (moved[static_cast<std::size_t>(i)] || food_total[static_cast<std::size_t>(i)] > 0.0f) {
            views_dirty = 1;
        }
    }
    *out_views_dirty = views_dirty;
    return 0;
}
