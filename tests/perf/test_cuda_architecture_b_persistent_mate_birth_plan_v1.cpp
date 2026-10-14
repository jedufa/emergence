#include "../../engine/perf/cuda_architecture_b_persistent_mate_birth_plan_v1.cuh"

#include <array>
#include <cstdint>
#include <iostream>

namespace mate_plan =
    eiel::cuda_architecture_b::persistent_mate_birth_plan_v1;

namespace {

constexpr int64_t kCapacity = 64;
constexpr int64_t kHeight = 8;
constexpr int64_t kWidth = 8;

#define CHECK(condition)                                                      \
    do {                                                                      \
        if (!(condition)) {                                                   \
            std::cerr << "check failed at line " << __LINE__ << ": "         \
                      << #condition << '\n';                                  \
            return false;                                                     \
        }                                                                     \
    } while (false)

struct Fixture {
    std::array<int64_t, kCapacity> agent_ids{};
    std::array<bool, kCapacity> alive{};
    std::array<int32_t, kCapacity> pos_y{};
    std::array<int32_t, kCapacity> pos_x{};
    std::array<int32_t, kHeight * kWidth> grid{};
    std::array<double, kCapacity> energy{};
    std::array<float, kCapacity> health{};
    std::array<float, kCapacity> injury{};
    std::array<float, kCapacity> scar{};
    std::array<int32_t, kCapacity> survival_steps{};
    std::array<float, kCapacity> viability{};
    std::array<bool, kCapacity> chemistry_blocked{};
    std::array<int32_t, kCapacity> cooldown{};
    std::array<float, kCapacity> threshold{};
    std::array<int64_t, kCapacity> free_queue{};
    int64_t free_count = 0;
    int64_t free_head = 0;
    std::array<uint64_t, 4> env_state{11, 13, 17, 19};
    std::array<int64_t, 2> env_buffer{0, 0};
    int64_t policy_semantic = 0;
    std::array<uint64_t, 4> runner_state{23, 29, 31, 37};
    std::array<int64_t, 2> runner_buffer{0, 0};
    int64_t metrics_semantic = 0;

    std::array<int32_t, kCapacity> planned_cooldown{};
    std::array<double, kCapacity> planned_energy{};
    std::array<int64_t, kCapacity> planned_parents{};
    std::array<int64_t, kCapacity> planned_mates{};
    std::array<int64_t, kCapacity> planned_children{};
    std::array<int32_t, kCapacity * 2> planned_positions{};
    std::array<double, kCapacity> planned_child_energy{};
    std::array<double, kCapacity> planned_parent_energy_after{};
    std::array<double, kCapacity> planned_mate_energy_after{};
    std::array<int64_t, kCapacity> decision_parents{};
    std::array<int64_t, kCapacity> decision_mates{};
    std::array<int32_t, kCapacity> decision_reasons{};
    std::array<int32_t, kCapacity> decision_adjacent_counts{};
    std::array<int32_t, kCapacity> decision_spawn_counts{};
    std::array<int64_t, mate_plan::kReasonCount> reason_counts{};
    int64_t planned_count = 0;
    int64_t decision_count = 0;
    int64_t env_bounded_draws = 0;
    int64_t env_semantic_draws = 0;
    int64_t runner_bounded_draws = 0;
    int64_t runner_semantic_draws = 0;
    int32_t status = -1;

    Fixture() {
        agent_ids.fill(-1);
        pos_y.fill(-1);
        pos_x.fill(-1);
        grid.fill(-1);
        energy.fill(20.0);
        health.fill(1.0f);
        viability.fill(1.0f);
        threshold.fill(5.0f);
        free_queue.fill(-1);
    }

    void add_agent(
        const int64_t slot,
        const int64_t agent_id,
        const int32_t y,
        const int32_t x,
        const float agent_health = 1.0f
    ) {
        agent_ids[slot] = agent_id;
        alive[slot] = true;
        pos_y[slot] = y;
        pos_x[slot] = x;
        health[slot] = agent_health;
        grid[static_cast<int64_t>(y) * kWidth + x] =
            static_cast<int32_t>(agent_id);
    }

    void build_free_queue() {
        free_count = 0;
        free_head = 0;
        for (int64_t slot = 0; slot < kCapacity; ++slot) {
            if (!alive[slot]) {
                free_queue[free_count++] = slot;
            }
        }
    }

    mate_plan::FlatMateBirthPlanStateV1 state() {
        return {
            agent_ids.data(),
            alive.data(),
            pos_y.data(),
            pos_x.data(),
            grid.data(),
            energy.data(),
            health.data(),
            injury.data(),
            scar.data(),
            survival_steps.data(),
            viability.data(),
            chemistry_blocked.data(),
            cooldown.data(),
            threshold.data(),
            free_queue.data(),
            &free_count,
            &free_head,
            env_state.data(),
            env_buffer.data(),
            &policy_semantic,
            runner_state.data(),
            runner_buffer.data(),
            &metrics_semantic,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
        };
    }

    mate_plan::FlatMateBirthPlanBuffersV1 buffers() {
        return {
            planned_cooldown.data(),
            planned_energy.data(),
            planned_parents.data(),
            planned_mates.data(),
            planned_children.data(),
            planned_positions.data(),
            planned_child_energy.data(),
            planned_parent_energy_after.data(),
            planned_mate_energy_after.data(),
            decision_parents.data(),
            decision_mates.data(),
            decision_reasons.data(),
            decision_adjacent_counts.data(),
            decision_spawn_counts.data(),
            reason_counts.data(),
            &planned_count,
            &decision_count,
            &env_bounded_draws,
            &env_semantic_draws,
            &runner_bounded_draws,
            &runner_semantic_draws,
            &status,
        };
    }
};

mate_plan::MateBirthPlanConfigV1 config(
    const int64_t max_births,
    const bool requires_mate = true,
    const bool env_policy_alias = true
) {
    return {
        kCapacity,
        kHeight,
        kWidth,
        kCapacity,
        max_births,
        0.25,
        2.0,
        3,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        env_policy_alias,
        requires_mate,
        false,
        false,
        false,
        0,
        0,
        0,
        0,
        0,
        1.0f,
    };
}

bool test_no_mate_with_b3() {
    Fixture fixture;
    fixture.add_agent(7, 17, 1, 1);
    fixture.build_free_queue();
    auto state = fixture.state();
    auto buffers = fixture.buffers();
    mate_plan::plan_births_serial_v1(state, buffers, config(3));

    CHECK(fixture.status == mate_plan::kStatusOk);
    CHECK(fixture.planned_count == 0);
    CHECK(fixture.decision_count == 1);
    CHECK(fixture.decision_reasons[0] == mate_plan::kReasonNoMate);
    CHECK(fixture.reason_counts[mate_plan::kReasonNoMate] == 1);
    CHECK(fixture.env_semantic_draws == 0);
    CHECK(fixture.runner_semantic_draws == 0);
    CHECK(fixture.free_head == 0);
    return true;
}

bool test_multiple_pairs_stable_order_and_rng_ownership() {
    Fixture fixture;
    fixture.add_agent(5, 10, 5, 4);
    fixture.add_agent(3, 2, 1, 2);
    fixture.add_agent(1, 11, 5, 5);
    fixture.add_agent(7, 1, 1, 1);
    fixture.energy[7] = 20.0;
    fixture.energy[3] = 12.0;
    fixture.energy[5] = 40.0;
    fixture.energy[1] = 8.0;
    fixture.build_free_queue();
    auto state = fixture.state();
    auto buffers = fixture.buffers();
    mate_plan::plan_births_serial_v1(state, buffers, config(3));

    CHECK(fixture.status == mate_plan::kStatusOk);
    CHECK(fixture.planned_count == 2);
    CHECK(fixture.agent_ids[fixture.planned_parents[0]] == 1);
    CHECK(fixture.agent_ids[fixture.planned_mates[0]] == 2);
    CHECK(fixture.agent_ids[fixture.planned_parents[1]] == 10);
    CHECK(fixture.agent_ids[fixture.planned_mates[1]] == 11);
    CHECK(fixture.decision_count == 4);
    CHECK(fixture.agent_ids[fixture.decision_parents[0]] == 1);
    CHECK(fixture.agent_ids[fixture.decision_parents[1]] == 2);
    CHECK(fixture.agent_ids[fixture.decision_parents[2]] == 10);
    CHECK(fixture.agent_ids[fixture.decision_parents[3]] == 11);
    CHECK(fixture.planned_positions[0] != fixture.planned_positions[2]
          || fixture.planned_positions[1] != fixture.planned_positions[3]);
    CHECK(fixture.env_bounded_draws == 14);
    CHECK(fixture.env_semantic_draws == 16);
    CHECK(fixture.runner_bounded_draws == 0);
    CHECK(fixture.runner_semantic_draws == 2);
    CHECK(fixture.policy_semantic == 16);
    CHECK(fixture.metrics_semantic == 2);
    CHECK(fixture.free_head == 2);
    CHECK(fixture.cooldown[7] == 3);
    CHECK(fixture.cooldown[3] == 2);
    CHECK(fixture.planned_child_energy[0] == 8.0);
    CHECK(fixture.planned_child_energy[1] == 12.0);
    CHECK(fixture.planned_energy[7] == 15.0);
    CHECK(fixture.planned_energy[3] == 9.0);
    CHECK(fixture.planned_energy[5] == 30.0);
    CHECK(fixture.planned_energy[1] == 6.0);
    CHECK(fixture.energy[7] == 20.0);
    CHECK(fixture.energy[3] == 12.0);
    CHECK(fixture.energy[5] == 40.0);
    CHECK(fixture.energy[1] == 8.0);
    return true;
}

bool test_cooldown_ages_before_parent_gates() {
    Fixture fixture;
    fixture.add_agent(0, 1, 0, 0);
    fixture.add_agent(1, 2, 0, 4);
    fixture.add_agent(2, 3, 4, 0);
    fixture.add_agent(3, 4, 4, 4);
    fixture.cooldown[0] = 1;
    fixture.cooldown[1] = 1;
    fixture.cooldown[2] = 1;
    fixture.cooldown[3] = 1;
    fixture.injury[0] = 0.75f;
    fixture.scar[1] = 0.75f;
    fixture.chemistry_blocked[2] = true;
    fixture.energy[3] = 4.0;
    fixture.build_free_queue();
    auto state = fixture.state();
    auto buffers = fixture.buffers();
    auto gate_config = config(3);
    gate_config.injury_gate = 0.5;
    gate_config.scar_gate = 0.5;
    gate_config.injury_gate_active = true;
    gate_config.scar_gate_active = true;
    gate_config.chemistry_viability_active = true;
    mate_plan::plan_births_serial_v1(state, buffers, gate_config);

    CHECK(fixture.status == mate_plan::kStatusOk);
    CHECK(fixture.planned_count == 0);
    CHECK(fixture.decision_count == 4);
    CHECK(fixture.decision_reasons[0] == mate_plan::kReasonHealth);
    CHECK(fixture.decision_reasons[1] == mate_plan::kReasonHealth);
    CHECK(fixture.decision_reasons[2] == mate_plan::kReasonHealth);
    CHECK(fixture.decision_reasons[3] == mate_plan::kReasonNoEnergy);
    CHECK(fixture.cooldown[0] == 0);
    CHECK(fixture.cooldown[1] == 0);
    CHECK(fixture.cooldown[2] == 0);
    CHECK(fixture.cooldown[3] == 0);
    return true;
}

bool test_cooldown_ages_before_already_mated_exclusion() {
    Fixture fixture;
    fixture.add_agent(0, 1, 1, 1);
    fixture.add_agent(1, 2, 1, 2);
    fixture.build_free_queue();
    auto state = fixture.state();
    auto buffers = fixture.buffers();
    auto one_step_cooldown = config(3);
    one_step_cooldown.reproduce_cooldown_steps = 1;
    mate_plan::plan_births_serial_v1(
        state,
        buffers,
        one_step_cooldown
    );

    CHECK(fixture.status == mate_plan::kStatusOk);
    CHECK(fixture.planned_count == 1);
    CHECK(fixture.decision_count == 2);
    CHECK(fixture.decision_reasons[0] == mate_plan::kReasonAccepted);
    CHECK(fixture.decision_reasons[1] == mate_plan::kReasonAlreadyMated);
    CHECK(fixture.cooldown[0] == 1);
    CHECK(fixture.cooldown[1] == 0);
    CHECK(fixture.planned_energy[0] == 15.0);
    CHECK(fixture.planned_energy[1] == 15.0);
    CHECK(fixture.energy[0] == 20.0);
    CHECK(fixture.energy[1] == 20.0);
    return true;
}

bool test_reserved_spawn_collision() {
    Fixture fixture;
    fixture.add_agent(0, 1, 2, 2);
    fixture.add_agent(1, 2, 2, 3);
    fixture.add_agent(2, 3, 4, 2);
    fixture.add_agent(3, 4, 4, 3);
    const std::array<std::array<int32_t, 2>, 9> blockers{{
        {{1, 2}}, {{2, 1}}, {{1, 3}}, {{3, 3}}, {{2, 4}},
        {{5, 2}}, {{4, 1}}, {{5, 3}}, {{4, 4}},
    }};
    int64_t blocker_slot = 4;
    int64_t blocker_id = 100;
    for (const auto& position : blockers) {
        fixture.add_agent(
            blocker_slot++,
            blocker_id++,
            position[0],
            position[1],
            0.0f
        );
    }
    fixture.build_free_queue();
    auto state = fixture.state();
    auto buffers = fixture.buffers();
    mate_plan::plan_births_serial_v1(state, buffers, config(3));

    CHECK(fixture.status == mate_plan::kStatusOk);
    CHECK(fixture.planned_count == 1);
    CHECK(fixture.planned_positions[0] == 3);
    CHECK(fixture.planned_positions[1] == 2);
    CHECK(fixture.reason_counts[mate_plan::kReasonNoSpace] >= 1);
    CHECK(fixture.free_head == 1);
    return true;
}

bool test_b64_and_failure_are_transactional() {
    Fixture fixture;
    fixture.add_agent(9, 4, 1, 1);
    fixture.add_agent(2, 5, 1, 2);
    fixture.build_free_queue();
    auto state = fixture.state();
    auto buffers = fixture.buffers();
    mate_plan::plan_births_serial_v1(
        state,
        buffers,
        config(64, true, false)
    );

    CHECK(fixture.status == mate_plan::kStatusOk);
    CHECK(fixture.planned_count == 1);
    CHECK(fixture.planned_mates[0] >= 0);
    CHECK(fixture.env_semantic_draws == 8);
    CHECK(fixture.policy_semantic == 0);
    CHECK(fixture.metrics_semantic == 1);

    Fixture invalid;
    invalid.add_agent(0, 1, 1, 1);
    invalid.add_agent(1, 2, 1, 2);
    invalid.cooldown[0] = 2;
    invalid.build_free_queue();
    const auto env_before = invalid.env_state;
    const auto runner_before = invalid.runner_state;
    const int64_t free_head_before = invalid.free_head;
    const int32_t cooldown_before = invalid.cooldown[0];
    auto invalid_state = invalid.state();
    auto invalid_buffers = invalid.buffers();
    mate_plan::plan_births_serial_v1(
        invalid_state,
        invalid_buffers,
        config(65)
    );
    CHECK(invalid.status == mate_plan::kStatusInvalidConfig);
    CHECK(invalid.env_state == env_before);
    CHECK(invalid.runner_state == runner_before);
    CHECK(invalid.free_head == free_head_before);
    CHECK(invalid.cooldown[0] == cooldown_before);
    CHECK(invalid.policy_semantic == 0);
    CHECK(invalid.metrics_semantic == 0);
    return true;
}

bool test_asexual_rows_are_the_only_negative_mates() {
    Fixture fixture;
    fixture.add_agent(0, 1, 1, 1);
    fixture.build_free_queue();
    fixture.planned_parents.fill(911);
    fixture.planned_mates.fill(912);
    fixture.planned_children.fill(913);
    fixture.planned_positions.fill(914);
    fixture.planned_child_energy.fill(915.0);
    fixture.planned_parent_energy_after.fill(916.0);
    fixture.planned_mate_energy_after.fill(917.0);
    auto state = fixture.state();
    auto buffers = fixture.buffers();
    mate_plan::plan_births_serial_v1(
        state,
        buffers,
        config(1, false, true)
    );

    CHECK(fixture.status == mate_plan::kStatusOk);
    CHECK(fixture.planned_count == 1);
    CHECK(fixture.planned_mates[0] == -1);
    CHECK(fixture.planned_parents[1] == 911);
    CHECK(fixture.planned_mates[1] == 912);
    CHECK(fixture.planned_children[1] == 913);
    CHECK(fixture.planned_positions[2] == 914);
    CHECK(fixture.planned_child_energy[1] == 915.0);
    CHECK(fixture.planned_parent_energy_after[1] == 916.0);
    CHECK(fixture.planned_mate_energy_after[1] == 917.0);
    return true;
}

}  // namespace

int main() {
    if (!test_no_mate_with_b3()) {
        return 1;
    }
    if (!test_multiple_pairs_stable_order_and_rng_ownership()) {
        return 2;
    }
    if (!test_cooldown_ages_before_parent_gates()) {
        return 3;
    }
    if (!test_cooldown_ages_before_already_mated_exclusion()) {
        return 4;
    }
    if (!test_reserved_spawn_collision()) {
        return 5;
    }
    if (!test_b64_and_failure_are_transactional()) {
        return 6;
    }
    if (!test_asexual_rows_are_the_only_negative_mates()) {
        return 7;
    }
    return 0;
}
