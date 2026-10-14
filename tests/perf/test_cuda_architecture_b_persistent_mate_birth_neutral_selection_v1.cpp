#include "../../engine/perf/cuda_architecture_b_persistent_mate_birth_plan_v1.cuh"

#include <array>
#include <cstdint>
#include <iostream>

namespace mate =
    eiel::cuda_architecture_b::persistent_mate_birth_plan_v1;

namespace {

constexpr std::int64_t kCapacity = 16;
constexpr std::int64_t kHeight = 8;
constexpr std::int64_t kWidth = 8;

#define CHECK(condition)                                                       \
    do {                                                                       \
        if (!(condition)) {                                                    \
            std::cerr << "check failed at line " << __LINE__ << ": "          \
                      << #condition << '\n';                                   \
            return false;                                                      \
        }                                                                      \
    } while (false)

struct Fixture {
    std::array<std::int64_t, kCapacity> ids{};
    std::array<bool, kCapacity> alive{};
    std::array<std::int32_t, kCapacity> y{};
    std::array<std::int32_t, kCapacity> x{};
    std::array<std::int32_t, kHeight * kWidth> grid{};
    std::array<double, kCapacity> energy{};
    std::array<float, kCapacity> health{};
    std::array<float, kCapacity> injury{};
    std::array<float, kCapacity> scar{};
    std::array<std::int32_t, kCapacity> survival_steps{};
    std::array<float, kCapacity> viability{};
    std::array<bool, kCapacity> chemistry_blocked{};
    std::array<std::int32_t, kCapacity> cooldown{};
    std::array<float, kCapacity> threshold{};
    std::array<std::int64_t, kCapacity> free_queue{};
    std::int64_t free_count = 0;
    std::int64_t free_head = 0;
    std::array<std::uint64_t, 4> env_state{11, 13, 17, 19};
    std::array<std::int64_t, 2> env_buffer{0, 0};
    std::int64_t policy_draws = 0;
    std::array<std::uint64_t, 4> runner_state{23, 29, 31, 37};
    std::array<std::int64_t, 2> runner_buffer{0, 0};
    std::int64_t metrics_draws = 0;

    std::array<std::int32_t, kCapacity> planned_cooldown{};
    std::array<double, kCapacity> planned_energy{};
    std::array<std::int64_t, kCapacity> planned_parents{};
    std::array<std::int64_t, kCapacity> planned_mates{};
    std::array<std::int64_t, kCapacity> planned_children{};
    std::array<std::int32_t, 2 * kCapacity> planned_positions{};
    std::array<double, kCapacity> planned_child_energy{};
    std::array<double, kCapacity> planned_parent_energy_after{};
    std::array<double, kCapacity> planned_mate_energy_after{};
    std::array<std::int64_t, kCapacity> decision_parents{};
    std::array<std::int64_t, kCapacity> decision_mates{};
    std::array<std::int32_t, kCapacity> decision_reasons{};
    std::array<std::int32_t, kCapacity> decision_adjacent{};
    std::array<std::int32_t, kCapacity> decision_spawn{};
    std::array<std::int64_t, mate::kReasonCount> reasons{};
    std::int64_t planned_count = 0;
    std::int64_t decision_count = 0;
    std::int64_t env_bounded = 0;
    std::int64_t env_semantic = 0;
    std::int64_t runner_bounded = 0;
    std::int64_t runner_semantic = 0;
    std::int32_t status = -1;

    Fixture() {
        ids.fill(-1);
        y.fill(-1);
        x.fill(-1);
        grid.fill(-1);
        energy.fill(20.0);
        health.fill(1.0f);
        viability.fill(1.0f);
        threshold.fill(1000.0f);
        free_queue.fill(-1);
        add(5, 1, 1, 1);
        add(1, 2, 1, 4);
        add(8, 3, 3, 1);
        add(3, 4, 3, 4);
        add(10, 5, 5, 1);
        add(0, 6, 5, 4);
        for (std::int64_t slot = 0; slot < kCapacity; ++slot) {
            if (!alive[slot]) {
                free_queue[free_count++] = slot;
            }
        }
    }

    void add(
        const std::int64_t slot,
        const std::int64_t id,
        const std::int32_t py,
        const std::int32_t px
    ) {
        ids[slot] = id;
        alive[slot] = true;
        y[slot] = py;
        x[slot] = px;
        grid[static_cast<std::int64_t>(py) * kWidth + px] =
            static_cast<std::int32_t>(id);
    }

    mate::FlatMateBirthPlanStateV1 state() {
        return {
            ids.data(), alive.data(), y.data(), x.data(), grid.data(),
            energy.data(), health.data(), injury.data(), scar.data(),
            survival_steps.data(),
            viability.data(), chemistry_blocked.data(), cooldown.data(),
            threshold.data(), free_queue.data(), &free_count, &free_head,
            env_state.data(), env_buffer.data(), &policy_draws,
            runner_state.data(), runner_buffer.data(), &metrics_draws,
            nullptr, nullptr, nullptr, nullptr,
        };
    }

    mate::FlatMateBirthPlanBuffersV1 buffers() {
        return {
            planned_cooldown.data(), planned_energy.data(),
            planned_parents.data(), planned_mates.data(),
            planned_children.data(), planned_positions.data(),
            planned_child_energy.data(), planned_parent_energy_after.data(),
            planned_mate_energy_after.data(), decision_parents.data(),
            decision_mates.data(), decision_reasons.data(),
            decision_adjacent.data(), decision_spawn.data(), reasons.data(),
            &planned_count, &decision_count, &env_bounded, &env_semantic,
            &runner_bounded, &runner_semantic, &status,
        };
    }
};

mate::MateBirthPlanConfigV1 config() {
    return {
        kCapacity, kHeight, kWidth, kCapacity, 6, 0.10, 0.0, 0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, false, false, false, false, false,
        0, 0, 0, 0, 0, 1.0f,
    };
}

mate::MateBirthSelectionConfigV1 neutral(
    const std::uint64_t seed,
    const double threshold = 5.0
) {
    return {
        1, mate::kMateBirthSelectionUniformNeutralV1, seed, 5, 9, false,
        threshold,
    };
}

bool same_result(const Fixture& left, const Fixture& right) {
#define SAME_FIELD(field)                                                      \
    do {                                                                       \
        if (left.field != right.field) {                                       \
            std::cerr << "different result field: " << #field << '\n';         \
            return false;                                                      \
        }                                                                      \
    } while (false)
    SAME_FIELD(status);
    SAME_FIELD(planned_count);
    SAME_FIELD(decision_count);
    SAME_FIELD(planned_parents);
    SAME_FIELD(planned_mates);
    SAME_FIELD(planned_children);
    SAME_FIELD(planned_positions);
    SAME_FIELD(decision_parents);
    SAME_FIELD(decision_reasons);
    SAME_FIELD(cooldown);
    SAME_FIELD(free_head);
    SAME_FIELD(env_state);
    SAME_FIELD(env_buffer);
    SAME_FIELD(policy_draws);
    SAME_FIELD(runner_state);
    SAME_FIELD(runner_buffer);
    SAME_FIELD(metrics_draws);
    for (std::int64_t row = 0; row < left.planned_count; ++row) {
        if (left.planned_child_energy[row] != right.planned_child_energy[row]) {
            std::cerr << "different result field: planned_child_energy\n";
            return false;
        }
    }
#undef SAME_FIELD
    return true;
}

std::array<std::int64_t, 6> decision_ids(const Fixture& fixture) {
    std::array<std::int64_t, 6> result{};
    for (std::int64_t row = 0; row < 6; ++row) {
        result[row] = fixture.ids[fixture.decision_parents[row]];
    }
    return result;
}

bool test_explicit_legacy_is_default_exact() {
    Fixture default_run;
    Fixture explicit_run;
    auto default_state = default_run.state();
    auto default_buffers = default_run.buffers();
    mate::plan_births_serial_v1(default_state, default_buffers, config());
    auto explicit_state = explicit_run.state();
    auto explicit_buffers = explicit_run.buffers();
    mate::plan_births_serial_v1(
        explicit_state,
        explicit_buffers,
        config(),
        {1, mate::kMateBirthSelectionLegacyAgentIdV1, 0, 0, 0, true, 0.0}
    );
    CHECK(same_result(default_run, explicit_run));
    return true;
}

bool test_neutral_order_is_seeded_and_deterministic() {
    Fixture first;
    Fixture repeat;
    Fixture other_seed;
    auto first_state = first.state();
    auto first_buffers = first.buffers();
    mate::plan_births_serial_v1(first_state, first_buffers, config(), neutral(101));
    auto repeat_state = repeat.state();
    auto repeat_buffers = repeat.buffers();
    mate::plan_births_serial_v1(
        repeat_state, repeat_buffers, config(), neutral(101)
    );
    auto other_state = other_seed.state();
    auto other_buffers = other_seed.buffers();
    mate::plan_births_serial_v1(
        other_state, other_buffers, config(), neutral(202)
    );
    CHECK(first.status == mate::kStatusOk);
    CHECK(same_result(first, repeat));
    CHECK(decision_ids(first)
          == (std::array<std::int64_t, 6>{6, 5, 2, 3, 1, 4}));
    CHECK(decision_ids(other_seed)
          == (std::array<std::int64_t, 6>{2, 5, 4, 3, 1, 6}));
    CHECK(decision_ids(first) != decision_ids(other_seed));
    return true;
}

bool test_neutral_ignores_trait_metrics_when_disabled() {
    Fixture baseline;
    Fixture changed;
    for (std::int64_t slot = 0; slot < kCapacity; ++slot) {
        changed.threshold[slot] = static_cast<float>(10 + slot * 17);
        changed.injury[slot] = static_cast<float>(slot) * 0.01f;
        changed.scar[slot] = static_cast<float>(slot) * 0.02f;
        changed.viability[slot] = 0.25f + static_cast<float>(slot) * 0.01f;
    }
    auto baseline_state = baseline.state();
    auto baseline_buffers = baseline.buffers();
    mate::plan_births_serial_v1(
        baseline_state, baseline_buffers, config(), neutral(101)
    );
    auto changed_state = changed.state();
    auto changed_buffers = changed.buffers();
    mate::plan_births_serial_v1(
        changed_state, changed_buffers, config(), neutral(101)
    );
    CHECK(same_result(baseline, changed));
    return true;
}

bool test_neutral_threshold_is_authoritative() {
    Fixture eligible;
    Fixture ineligible;
    auto eligible_state = eligible.state();
    auto eligible_buffers = eligible.buffers();
    mate::plan_births_serial_v1(
        eligible_state, eligible_buffers, config(), neutral(101, 5.0)
    );
    auto ineligible_state = ineligible.state();
    auto ineligible_buffers = ineligible.buffers();
    mate::plan_births_serial_v1(
        ineligible_state, ineligible_buffers, config(), neutral(101, 25.0)
    );
    CHECK(eligible.planned_count == 6);
    CHECK(ineligible.planned_count == 0);
    CHECK(ineligible.reasons[mate::kReasonNoEnergy] == 6);
    return true;
}

bool test_invalid_selection_fails_closed() {
    Fixture fixture;
    const auto cooldown_before = fixture.cooldown;
    const auto env_before = fixture.env_state;
    const auto runner_before = fixture.runner_state;
    auto state = fixture.state();
    auto buffers = fixture.buffers();
    auto invalid = neutral(101);
    invalid.mode = 99;
    mate::plan_births_serial_v1(state, buffers, config(), invalid);
    CHECK(fixture.status == mate::kStatusInvalidConfig);
    CHECK(fixture.planned_count == 0);
    CHECK(fixture.free_head == 0);
    CHECK(fixture.cooldown == cooldown_before);
    CHECK(fixture.env_state == env_before);
    CHECK(fixture.runner_state == runner_before);
    return true;
}

}  // namespace

int main() {
    if (!test_explicit_legacy_is_default_exact()) return 1;
    if (!test_neutral_order_is_seeded_and_deterministic()) return 2;
    if (!test_neutral_ignores_trait_metrics_when_disabled()) return 3;
    if (!test_neutral_threshold_is_authoritative()) return 4;
    if (!test_invalid_selection_fails_closed()) return 5;
    return 0;
}
