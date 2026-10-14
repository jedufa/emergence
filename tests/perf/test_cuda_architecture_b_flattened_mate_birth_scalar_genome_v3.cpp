#include <cmath>

using std::isfinite;

#include "../../engine/perf/cuda_architecture_b_flattened_mate_birth_scalar_genome_v3.cuh"

#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>

namespace mate_scalar =
    eiel::cuda_architecture_b::flattened_mate_birth_scalar_genome_v3;

namespace {

constexpr int64_t kBirthCapacity = 64;
constexpr int64_t kSlotCount = 192;
constexpr int64_t kTapeCapacity = 8192;

template <typename T>
using Slots = std::array<T, kSlotCount>;

template <typename T>
using Births = std::array<T, kBirthCapacity>;

struct Fixture {
    Slots<int64_t> agent_ids{};
    Slots<bool> alive{};
    Slots<int32_t> hidden{};
    Slots<int32_t> memory{};
    Slots<int32_t> sensor_mask{};
    Slots<int32_t> sensor_radius{};
    Slots<int32_t> introspection_mask{};
    Slots<int32_t> self_regulation_mask{};
    Slots<int32_t> self_modification_mask{};
    Slots<int32_t> directed_inheritance_mask{};
    Slots<int64_t> policy_memory{};
    Slots<bool> policy_gru{};
    std::array<Slots<float>, 24> floats{};

    Births<int64_t> parents{};
    Births<int64_t> mates{};
    Births<int64_t> children{};
    Births<int64_t> canonical_parent_0{};
    Births<int64_t> canonical_parent_1{};
    std::array<uint8_t, kBirthCapacity * 7> donor_ordinals{};
    int64_t planned_count = 0;

    std::array<Births<int32_t>, 8> staged_int32{};
    Births<int64_t> staged_policy_memory{};
    Births<bool> staged_policy_gru{};
    std::array<Births<float>, 24> staged_float{};

    std::array<uint32_t, 624> input_words{};
    std::array<uint32_t, 624> output_words{};
    std::array<uint32_t, 624> trial_words{};
    int64_t input_index = 0;
    bool input_has_gauss = false;
    double input_gauss_next = 0.0;
    int64_t input_raw_draws = 0;
    int64_t input_semantic_draws = 0;
    std::array<double, kTapeCapacity> primary_tape{};
    std::array<double, kTapeCapacity> secondary_tape{};

    int64_t output_index = -1;
    bool output_has_gauss = false;
    double output_gauss_next = 0.0;
    int64_t output_raw_draws = -1;
    int64_t output_semantic_draws = -1;
    int64_t committed_count = -1;
    int64_t semantic_added = -1;
    int32_t status = -1;

    Fixture() {
        static constexpr int32_t kMemoryLadder[5] = {0, 8, 16, 32, 64};
        for (int64_t slot = 0; slot < kSlotCount; ++slot) {
            agent_ids[slot] = 1000 + slot;
            alive[slot] = true;
            hidden[slot] = 32 + static_cast<int32_t>(16 * (slot % 5));
            memory[slot] = kMemoryLadder[slot % 5];
            sensor_mask[slot] = static_cast<int32_t>(2 + 13 * slot);
            sensor_radius[slot] = static_cast<int32_t>(1 + slot % 3);
            introspection_mask[slot] = static_cast<int32_t>(40000 + slot);
            self_regulation_mask[slot] = static_cast<int32_t>(40 + slot);
            self_modification_mask[slot] = static_cast<int32_t>(70 + slot);
            directed_inheritance_mask[slot] = static_cast<int32_t>(90 + slot);
            policy_memory[slot] = 0;
            policy_gru[slot] = false;

            floats[0][slot] = -0.8f + 0.08f * static_cast<float>(slot % 17);
            floats[1][slot] = -1.5f + 0.12f * static_cast<float>(slot % 19);
            floats[2][slot] = 0.02f + 0.01f * static_cast<float>(slot % 7);
            floats[3][slot] = 10.0f + 2.0f * static_cast<float>(slot % 13);
            floats[4][slot] = 0.4f + 0.15f * static_cast<float>(slot % 9);
            floats[5][slot] = 0.3f + 0.05f * static_cast<float>(slot % 10);
            floats[6][slot] = 0.005f + 0.01f * static_cast<float>(slot % 8);
            floats[7][slot] = 0.2f + 0.2f * static_cast<float>(slot % 11);
            floats[8][slot] = 0.02f + 0.07f * static_cast<float>(slot % 10);
            floats[9][slot] = 0.1f + 0.1f * static_cast<float>(slot % 8);
            floats[10][slot] = 0.02f + 0.08f * static_cast<float>(slot % 9);
            floats[11][slot] = 0.03f + 0.07f * static_cast<float>(slot % 10);
            floats[12][slot] = 0.04f + 0.06f * static_cast<float>(slot % 11);
            floats[13][slot] = 0.1f + 0.09f * static_cast<float>(slot % 9);
            floats[14][slot] = 0.01f + 0.04f * static_cast<float>(slot % 8);
            floats[15][slot] = 0.01f + 0.02f * static_cast<float>(slot % 7);
            for (int32_t field = 16; field < 24; ++field) {
                floats[field][slot] = -99.0f;
            }
        }
        for (int64_t birth = 0; birth < kBirthCapacity; ++birth) {
            parents[birth] = birth;
            mates[birth] = birth % 2 == 0 ? -1 : 64 + birth;
            children[birth] = 128 + birth;
            canonical_parent_0[birth] =
                mates[birth] >= 0 ? mates[birth] : parents[birth];
            canonical_parent_1[birth] = parents[birth];
        }
        donor_ordinals.fill(0xff);
        for (auto& field : staged_int32) field.fill(-777);
        staged_policy_memory.fill(-777);
        staged_policy_gru.fill(true);
        for (auto& field : staged_float) field.fill(-777.0f);
        output_words.fill(0xdeadbeefU);
        trial_words.fill(0xcafebabeU);
    }

    mate_scalar::FlatAuthoritativeFixedMlpScalarGenomeState state() const {
        return {
            agent_ids.data(), alive.data(), hidden.data(), memory.data(),
            sensor_mask.data(), sensor_radius.data(), introspection_mask.data(),
            self_regulation_mask.data(), self_modification_mask.data(),
            directed_inheritance_mask.data(), policy_memory.data(),
            policy_gru.data(), floats[0].data(), floats[1].data(),
            floats[2].data(), floats[3].data(), floats[4].data(),
            floats[5].data(), floats[6].data(), floats[7].data(),
            floats[8].data(), floats[9].data(), floats[10].data(),
            floats[11].data(), floats[12].data(), floats[13].data(),
            floats[14].data(), floats[15].data(), floats[16].data(),
            floats[17].data(), floats[18].data(), floats[19].data(),
            floats[20].data(), floats[21].data(), floats[22].data(),
            floats[23].data()
        };
    }

    mate_scalar::FlatMateOrdinaryScalarGenomePlan plan() const {
        return {parents.data(), mates.data(), children.data(), &planned_count};
    }

    mate_scalar::FlatAuthoritativeCpythonRandomState random_state() const {
        return {
            input_words.data(), &input_index, &input_has_gauss,
            &input_gauss_next, primary_tape.data(), secondary_tape.data(),
            &input_raw_draws, &input_semantic_draws
        };
    }

    mate_scalar::FlatMateOrdinaryInheritancePlanV1 linked_inheritance(
        const bool mutation_enabled = false
    ) {
        return {
            mate_scalar::linkage::InheritanceModeV1::linked_parent_copy_v1,
            mutation_enabled,
            canonical_parent_0.data(),
            canonical_parent_1.data(),
            donor_ordinals.data(),
        };
    }

    mate_scalar::FlatBirthOrderedScalarGenomeScratch scratch() {
        return {
            staged_int32[0].data(), staged_int32[1].data(),
            staged_int32[2].data(), staged_int32[3].data(),
            staged_int32[4].data(), staged_int32[5].data(),
            staged_int32[6].data(), staged_int32[7].data(),
            staged_policy_memory.data(), staged_policy_gru.data(),
            staged_float[0].data(), staged_float[1].data(),
            staged_float[2].data(), staged_float[3].data(),
            staged_float[4].data(), staged_float[5].data(),
            staged_float[6].data(), staged_float[7].data(),
            staged_float[8].data(), staged_float[9].data(),
            staged_float[10].data(), staged_float[11].data(),
            staged_float[12].data(), staged_float[13].data(),
            staged_float[14].data(), staged_float[15].data(),
            staged_float[16].data(), staged_float[17].data(),
            staged_float[18].data(), staged_float[19].data(),
            staged_float[20].data(), staged_float[21].data(),
            staged_float[22].data(), staged_float[23].data(),
            output_words.data(), trial_words.data(), &output_index,
            &output_has_gauss, &output_gauss_next, &output_raw_draws,
            &output_semantic_draws, &committed_count, &semantic_added, &status
        };
    }
};

uint32_t float_bits(const float value) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

bool rejection_contracts_hold(const Fixture& accepted) {
    for (int32_t scenario = 0; scenario < 3; ++scenario) {
        Fixture fixture = accepted;
        const int32_t staged_before = fixture.staged_int32[0][0];
        const uint32_t word_before = fixture.output_words[0];
        fixture.planned_count = 1;
        if (scenario == 0) fixture.mates[0] = fixture.parents[0];
        if (scenario == 1) {
            fixture.mates[0] = 64;
            fixture.policy_gru[64] = true;
        }
        if (scenario == 2) fixture.planned_count = kBirthCapacity + 1;
        auto scratch = fixture.scratch();
        mate_scalar::
            stage_ordinary_fixed_mlp_child_scalar_genomes_with_mates_external_scratch_v3(
                fixture.state(), fixture.plan(), fixture.random_state(), scratch,
                {kBirthCapacity, kSlotCount, kTapeCapacity, 3}
            );
        if (
            fixture.status != mate_scalar::kStatusInvalidScalarBirthPlan
            || fixture.committed_count != 0 || fixture.semantic_added != 0
            || fixture.staged_int32[0][0] != staged_before
            || fixture.output_words[0] != word_before
        ) {
            return false;
        }
    }
    return true;
}

bool linked_inheritance_contracts_hold(const Fixture& accepted) {
    Fixture fixture = accepted;
    fixture.planned_count = 1;
    fixture.parents[0] = 1;
    fixture.mates[0] = 65;
    fixture.children[0] = 128;
    // Deliberately reverse direct parent order. The linked transaction must
    // interpret donor ordinal against this caller-owned canonical order.
    fixture.canonical_parent_0[0] = 65;
    fixture.canonical_parent_1[0] = 1;
    fixture.floats[2][1] = 0.0f;
    fixture.floats[2][65] = 0.0f;
    auto scratch = fixture.scratch();
    mate_scalar::
        stage_ordinary_fixed_mlp_child_scalar_genomes_with_mates_external_scratch_v4(
            fixture.state(),
            fixture.plan(),
            fixture.random_state(),
            scratch,
            {kBirthCapacity, kSlotCount, kTapeCapacity, 3},
            fixture.linked_inheritance(false)
        );
    if (
        fixture.status != mate_scalar::kStatusOk
        || fixture.committed_count != 1
        || fixture.semantic_added != 7
        || fixture.output_semantic_draws
            != fixture.input_semantic_draws + 7
    ) {
        return false;
    }
    for (int32_t group = 0; group < 7; ++group) {
        if (fixture.donor_ordinals[group] > 1) {
            return false;
        }
    }
    const auto donor_slot = [&](const int32_t group) {
        return fixture.donor_ordinals[group] == 0 ? 65 : 1;
    };
    for (const int32_t field : {0, 1, 4}) {
        if (
            float_bits(fixture.staged_float[field][0])
            != float_bits(mate_scalar::scalar::canon_genome_float(
                fixture.floats[field][donor_slot(0)]
            ))
        ) {
            return false;
        }
    }
    for (int32_t field = 5; field <= 15; ++field) {
        if (
            float_bits(fixture.staged_float[field][0])
            != float_bits(mate_scalar::scalar::canon_genome_float(
                fixture.floats[field][donor_slot(1)]
            ))
        ) {
            return false;
        }
    }
    if (
        float_bits(fixture.staged_float[3][0])
            != float_bits(mate_scalar::scalar::canon_genome_float(
                fixture.floats[3][donor_slot(2)]
            ))
        || float_bits(fixture.staged_float[2][0])
            != float_bits(mate_scalar::scalar::canon_genome_float(
                fixture.floats[2][donor_slot(3)]
            ))
        || fixture.staged_int32[0][0]
            != fixture.hidden[donor_slot(4)]
    ) {
        return false;
    }

    Fixture invalid = accepted;
    invalid.planned_count = 1;
    invalid.parents[0] = 1;
    invalid.mates[0] = 65;
    invalid.children[0] = 128;
    invalid.canonical_parent_0[0] = 1;
    invalid.canonical_parent_1[0] = 1;
    const uint32_t word_before = invalid.output_words[0];
    auto invalid_scratch = invalid.scratch();
    mate_scalar::
        stage_ordinary_fixed_mlp_child_scalar_genomes_with_mates_external_scratch_v4(
            invalid.state(),
            invalid.plan(),
            invalid.random_state(),
            invalid_scratch,
            {kBirthCapacity, kSlotCount, kTapeCapacity, 3},
            invalid.linked_inheritance(false)
        );
    return (
        invalid.status == mate_scalar::kStatusInvalidInheritancePlan
        && invalid.committed_count == 0
        && invalid.semantic_added == 0
        && invalid.output_words[0] == word_before
    );
}

}  // namespace

int main() {
    Fixture fixture;
    int64_t tape_entries = 0;
    int32_t has_gauss = 0;
    if (!(std::cin >> fixture.planned_count >> fixture.input_index >> has_gauss
          >> fixture.input_gauss_next >> fixture.input_raw_draws
          >> fixture.input_semantic_draws)) {
        return 2;
    }
    fixture.input_has_gauss = has_gauss != 0;
    for (auto& word : fixture.input_words) std::cin >> word;
    std::cin >> tape_entries;
    for (int64_t entry = 0; entry < tape_entries; ++entry) {
        int64_t index = -1;
        double primary = 0.0;
        double secondary = 0.0;
        std::cin >> index >> primary >> secondary;
        if (index < 0 || index >= kTapeCapacity) return 3;
        fixture.primary_tape[index] = primary;
        fixture.secondary_tape[index] = secondary;
    }

    auto scratch = fixture.scratch();
    mate_scalar::
        stage_ordinary_fixed_mlp_child_scalar_genomes_with_mates_external_scratch_v3(
            fixture.state(), fixture.plan(), fixture.random_state(), scratch,
            {kBirthCapacity, kSlotCount, kTapeCapacity, 3}
        );
    if (!rejection_contracts_hold(fixture)) return 4;
    if (!linked_inheritance_contracts_hold(fixture)) return 5;

    std::cout << fixture.status << ' ' << fixture.committed_count << ' '
              << fixture.semantic_added << ' ' << fixture.output_index << ' '
              << static_cast<int32_t>(fixture.output_has_gauss) << ' '
              << std::setprecision(17) << fixture.output_gauss_next << ' '
              << fixture.output_raw_draws << ' '
              << fixture.output_semantic_draws << '\n';
    for (int64_t birth = 0; birth < fixture.planned_count; ++birth) {
        for (const auto& field : fixture.staged_int32) {
            std::cout << field[birth] << ' ';
        }
        std::cout << fixture.staged_policy_memory[birth] << ' '
                  << static_cast<int32_t>(fixture.staged_policy_gru[birth]);
        for (const auto& field : fixture.staged_float) {
            std::cout << ' ' << float_bits(field[birth]);
        }
        std::cout << '\n';
    }
    for (const auto word : fixture.output_words) std::cout << word << ' ';
    std::cout << '\n';
    return 0;
}
