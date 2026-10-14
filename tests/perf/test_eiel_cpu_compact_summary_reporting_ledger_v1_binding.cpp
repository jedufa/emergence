#include "cpu_compact_summary_reporting_ledger_v1.hpp"

#include <pybind11/pybind11.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

namespace py = pybind11;
namespace authority =
    eiel::cuda_architecture_b::compact_summary_reporting_authority_v1;
namespace canonical =
    eiel::cuda_architecture_b::compact_summary_host_canonical_v1;
namespace ledger =
    eiel::native::cpu_compact_summary_reporting_ledger_v1;

namespace {

using LedgerV1 = ledger::CpuCompactSummaryReportingLedgerV1;

inline constexpr std::size_t kFixtureRows = 4;
inline constexpr std::size_t kFixtureSlots = 4;
inline constexpr std::size_t kAgentCapacity = 32;
inline constexpr std::size_t kRetainedRowBytes =
    sizeof(authority::GenerationAuthorityHeaderV1)
    + kFixtureRows
        * (sizeof(authority::BehaviorAuthorityRowV1)
           + sizeof(authority::LineageAuthorityRowV1));

static_assert(std::is_same_v<
    decltype(std::declval<const LedgerV1&>().behavior_rows()),
    const authority::BehaviorAuthorityRowV1*
>);
static_assert(std::is_same_v<
    decltype(std::declval<const LedgerV1&>().lineage_rows()),
    const authority::LineageAuthorityRowV1*
>);
static_assert(std::is_same_v<
    decltype(std::declval<const LedgerV1&>().canonical_result()),
    const canonical::HostCanonicalResultV1&
>);
static_assert(std::is_same_v<
    decltype(std::declval<const LedgerV1&>().lineage_metrics_state()),
    const canonical::LineageMetricsStateV1&
>);

struct ResidentSourcesStorageV1 {
    std::array<std::int64_t, kFixtureSlots> slot_to_agent_id{};
    std::array<double, kFixtureSlots> energy_collected_exact{};
    std::array<double, kFixtureSlots> early_energy_collected_exact{};
    std::array<float, kFixtureSlots> injury{};
    std::array<float, kFixtureSlots> aggression_bias{};
    std::array<float, kFixtureSlots> early_injury_gained{};
    std::array<float, kFixtureSlots> early_life_weight{};
    std::array<float, kFixtureSlots> drive_jitter{};
    std::array<float, kFixtureSlots> pred_err_sum{};
    std::array<float, kFixtureSlots> birth_reset_shock{};
    std::array<float, kFixtureSlots> episode0_pred_err_sum{};
    std::array<float, kFixtureSlots> late_pred_err_sum{};
    std::array<std::int32_t, kFixtureSlots> lifespan_steps{};
    std::array<std::int32_t, kFixtureSlots> offspring_count{};
    std::array<std::int32_t, kFixtureSlots> attacks_made{};
    std::array<std::int32_t, kFixtureSlots> early_survival_steps{};
    std::array<std::int32_t, kFixtureSlots> action_switches{};
    std::array<std::int32_t, kFixtureSlots> action_oscillations{};
    std::array<std::int32_t, kFixtureSlots> selfharm_events{};
    std::array<std::int32_t, kFixtureSlots> pred_err_count{};
    std::array<std::int32_t, kFixtureSlots> episode0_pred_err_count{};
    std::array<std::int32_t, kFixtureSlots> episode0_selfharm_events{};
    std::array<std::int32_t, kFixtureSlots> late_pred_err_count{};
    std::array<double, kFixtureSlots> policy_energy_observation{};
    std::array<float, kFixtureSlots> health{};
    std::array<float, kFixtureSlots * 4U> drives{};
    std::array<float, kFixtureSlots> identity_drift_ema{};
    std::array<float, kFixtureSlots> exploration_bias{};
    std::array<std::int32_t, kFixtureSlots> pos_y{};
    std::array<std::int32_t, kFixtureSlots> pos_x{};
    std::array<std::int32_t, kFixtureSlots> lineage_founder_id{};
    std::array<std::int32_t, kFixtureSlots> lineage_birth_generation{};
    std::array<std::int32_t, kFixtureSlots> lineage_birth_tick{};
    std::array<std::int32_t, kFixtureSlots> lineage_depth{};
    std::array<std::int32_t, kFixtureSlots> first_repro_step{};
    std::array<std::int32_t, kFixtureSlots> patch_habitat_lineage_id{};
    std::array<std::int32_t, kFixtureSlots> genome_hidden_size{};
    std::array<std::int32_t, kFixtureSlots> genome_memory_dim{};
    std::array<std::uint8_t, kFixtureSlots> policy_use_gru{};

    [[nodiscard]] authority::ResidentGenerationSourcesV1 view() const noexcept {
        return {
            slot_to_agent_id.data(),
            energy_collected_exact.data(),
            early_energy_collected_exact.data(),
            injury.data(),
            aggression_bias.data(),
            early_injury_gained.data(),
            early_life_weight.data(),
            drive_jitter.data(),
            pred_err_sum.data(),
            birth_reset_shock.data(),
            episode0_pred_err_sum.data(),
            late_pred_err_sum.data(),
            lifespan_steps.data(),
            offspring_count.data(),
            attacks_made.data(),
            early_survival_steps.data(),
            action_switches.data(),
            action_oscillations.data(),
            selfharm_events.data(),
            pred_err_count.data(),
            episode0_pred_err_count.data(),
            episode0_selfharm_events.data(),
            late_pred_err_count.data(),
            policy_energy_observation.data(),
            health.data(),
            drives.data(),
            identity_drift_ema.data(),
            exploration_bias.data(),
            pos_y.data(),
            pos_x.data(),
            lineage_founder_id.data(),
            lineage_birth_generation.data(),
            lineage_birth_tick.data(),
            lineage_depth.data(),
            first_repro_step.data(),
            patch_habitat_lineage_id.data(),
            genome_hidden_size.data(),
            genome_memory_dim.data(),
            policy_use_gru.data(),
            1U,
            static_cast<std::int32_t>(kFixtureSlots),
        };
    }
};

ResidentSourcesStorageV1 empty_sources() {
    ResidentSourcesStorageV1 sources{};
    sources.slot_to_agent_id.fill(-1);
    sources.early_life_weight.fill(1.0F);
    sources.health.fill(1.0F);
    sources.first_repro_step.fill(-1);
    return sources;
}

void set_live_slot(
    ResidentSourcesStorageV1& sources,
    const std::size_t slot,
    const std::int64_t agent_id,
    const std::int32_t founder_id,
    const std::int32_t lineage_depth,
    const std::int32_t offspring_count,
    const std::int32_t lifespan_steps,
    const double energy_collected,
    const float injury,
    const float aggression_bias
) {
    sources.slot_to_agent_id[slot] = agent_id;
    sources.energy_collected_exact[slot] = energy_collected;
    sources.early_energy_collected_exact[slot] = energy_collected * 0.25;
    sources.injury[slot] = injury;
    sources.aggression_bias[slot] = aggression_bias;
    sources.early_injury_gained[slot] = injury * 0.5F;
    sources.early_life_weight[slot] = 0.5F;
    sources.drive_jitter[slot] = 0.0625F;
    sources.pred_err_sum[slot] = 1.0F;
    sources.birth_reset_shock[slot] = 0.125F;
    sources.episode0_pred_err_sum[slot] = 0.5F;
    sources.late_pred_err_sum[slot] = 0.25F;
    sources.lifespan_steps[slot] = lifespan_steps;
    sources.offspring_count[slot] = offspring_count;
    sources.attacks_made[slot] = offspring_count > 1 ? 1 : 0;
    sources.early_survival_steps[slot] = lifespan_steps > 3 ? 3 : lifespan_steps;
    sources.action_switches[slot] = 1;
    sources.action_oscillations[slot] = 0;
    sources.selfharm_events[slot] = 0;
    sources.pred_err_count[slot] = 2;
    sources.episode0_pred_err_count[slot] = 2;
    sources.episode0_selfharm_events[slot] = 0;
    sources.late_pred_err_count[slot] = 1;
    sources.policy_energy_observation[slot] = 40.0 + static_cast<double>(slot);
    sources.health[slot] = 1.0F;
    for (std::size_t drive = 0; drive < 4U; ++drive) {
        sources.drives[slot * 4U + drive] =
            static_cast<float>(slot + 1U) * 0.25F
            + static_cast<float>(drive) * 0.125F;
    }
    sources.identity_drift_ema[slot] = 0.03125F;
    sources.exploration_bias[slot] = -0.125F;
    sources.pos_y[slot] = static_cast<std::int32_t>(slot + 2U);
    sources.pos_x[slot] = static_cast<std::int32_t>(slot + 3U);
    sources.lineage_founder_id[slot] = founder_id;
    sources.lineage_birth_generation[slot] = 0;
    sources.lineage_birth_tick[slot] = lineage_depth * 3;
    sources.lineage_depth[slot] = lineage_depth;
    sources.first_repro_step[slot] = offspring_count > 0 ? 4 : -1;
    sources.patch_habitat_lineage_id[slot] = founder_id;
    sources.genome_hidden_size[slot] = 8 + lineage_depth;
    sources.genome_memory_dim[slot] = 4 + lineage_depth;
    sources.policy_use_gru[slot] = lineage_depth % 2 == 0 ? 1U : 0U;
}

authority::BehaviorTerminalValuesV1 death_terminal() {
    authority::BehaviorTerminalValuesV1 terminal{};
    terminal.energy_collected_exact = 0.75;
    terminal.early_energy_collected_exact = 0.25;
    terminal.injury_end = 0.5F;
    terminal.aggression_bias_f32 = 0.375F;
    terminal.early_injury_gained = 0.25F;
    terminal.early_life_weight = 0.5F;
    terminal.drive_jitter = 0.0625F;
    terminal.pred_err_sum = 0.5F;
    terminal.birth_reset_shock = 0.125F;
    terminal.episode0_pred_err_sum = 0.25F;
    terminal.late_pred_err_sum = 0.125F;
    terminal.lifespan_steps = 3;
    terminal.offspring = 0;
    terminal.attacks_made = 0;
    terminal.early_survival_steps = 2;
    terminal.action_switches = 1;
    terminal.action_oscillations = 0;
    terminal.selfharm_events = 0;
    terminal.pred_err_count = 1;
    terminal.episode0_pred_err_count = 1;
    terminal.episode0_selfharm_events = 0;
    terminal.late_pred_err_count = 1;
    return terminal;
}

authority::ReportingFitnessConfigV1 fitness_config() {
    authority::ReportingFitnessConfigV1 config{};
    config.parent_fraction = 0.5;
    config.patch_habitat_lineage_available = 1U;
    return config;
}

canonical::LineageMetricsConfigV1 lineage_metrics_config() {
    canonical::LineageMetricsConfigV1 config{};
    config.population_dominance_threshold = 0.80;
    config.entropy_minimum = 0.10;
    config.reproduction_dominance_threshold = 0.80;
    config.population_dominance_streak_required = 5;
    config.minimum_active_lineages = 2;
    config.reproduction_dominance_streak_required = 5;
    config.metrics_defined = 1U;
    return config;
}

struct GenerationCaptureV1 {
    std::int64_t generation = 0;
    authority::GenerationAuthorityHeaderV1 header{};
    std::array<authority::BehaviorAuthorityRowV1, kFixtureRows> behavior{};
    std::array<authority::LineageAuthorityRowV1, kFixtureRows> lineage{};
    canonical::HostCanonicalResultV1 canonical_result{};
    std::array<std::uint8_t, kRetainedRowBytes> retained_row{};
    bool retained_row_copied = false;
};

GenerationCaptureV1 capture_generation(
    const std::int64_t generation,
    const LedgerV1& owner
) {
    GenerationCaptureV1 capture{};
    capture.generation = generation;
    capture.header = owner.header();
    for (std::size_t index = 0; index < kFixtureRows; ++index) {
        capture.behavior[index] = owner.behavior_rows()[index];
        capture.lineage[index] = owner.lineage_rows()[index];
    }
    capture.canonical_result = owner.canonical_result();
    capture.retained_row_copied = owner.retained_row_bytes_v1()
            == capture.retained_row.size()
        && owner.copy_retained_row_v1(
            capture.retained_row.data(), capture.retained_row.size()
        );
    return capture;
}

bool run_generation_zero(LedgerV1& owner, GenerationCaptureV1& capture) {
    std::array<std::int64_t, 2> active_ids{10, 11};
    std::array<std::int32_t, kAgentCapacity> agent_to_slot{};
    agent_to_slot.fill(-1);
    agent_to_slot[10] = 0;
    agent_to_slot[11] = 1;
    std::array<float, kFixtureSlots> injury_by_slot{};
    injury_by_slot[0] = -0.0F;
    injury_by_slot[1] = 0.25F;

    if (!owner.initialize_generation_from_slot_state_v1(
            12, active_ids.data(), agent_to_slot.data(), injury_by_slot.data(),
            static_cast<std::int32_t>(active_ids.size()),
            static_cast<std::int32_t>(agent_to_slot.size())
        )
        || !owner.count_action_v1(0, 10, 0)
        || !owner.count_action_v1(0, 10, 0)
        || !owner.count_action_v1(1, 11, 6)
        || !owner.seal_death_before_slot_reuse_v1(0, 10, death_terminal())
        || !owner.append_ordinary_birth_v1(12, 0, 0.0F)
        || !owner.count_action_v1(0, 12, 4)) {
        return false;
    }

    ResidentSourcesStorageV1 sources = empty_sources();
    set_live_slot(sources, 1, 11, 7, 0, 1, 6, 2.5, 0.125F, -0.25F);
    set_live_slot(sources, 0, 12, 7, 1, 0, 5, 1.5, 0.0F, 0.125F);
    const authority::ResidentGenerationSourcesV1 source_view = sources.view();
    if (!owner.finalize_pre_turnover_v1(source_view, fitness_config())
        || !owner.finalized() || owner.committed()
        || !owner.canonicalize_retained_summary_v1(
            0, lineage_metrics_config()
        )) {
        return false;
    }
    capture = capture_generation(0, owner);
    return owner.committed();
}

bool run_generation_one(LedgerV1& owner, GenerationCaptureV1& capture) {
    std::array<std::int64_t, 2> active_ids{11, 12};
    std::array<std::int32_t, kAgentCapacity> agent_to_slot{};
    agent_to_slot.fill(-1);
    agent_to_slot[11] = 1;
    agent_to_slot[12] = 0;
    std::array<float, kFixtureSlots> injury_by_slot{};
    injury_by_slot[0] = 0.0F;
    injury_by_slot[1] = 0.125F;

    if (!owner.initialize_generation_from_slot_state_v1(
            13, active_ids.data(), agent_to_slot.data(), injury_by_slot.data(),
            static_cast<std::int32_t>(active_ids.size()),
            static_cast<std::int32_t>(agent_to_slot.size())
        )
        || !owner.count_action_v1(1, 11, 5)
        || !owner.count_action_v1(0, 12, 4)
        || !owner.count_action_v1(0, 12, 4)) {
        return false;
    }

    ResidentSourcesStorageV1 sources = empty_sources();
    set_live_slot(sources, 1, 11, 7, 0, 2, 7, 3.0, 0.25F, -0.25F);
    set_live_slot(sources, 0, 12, 7, 1, 1, 6, 2.0, 0.125F, 0.125F);
    const authority::ResidentGenerationSourcesV1 source_view = sources.view();
    if (!owner.finalize_pre_turnover_v1(source_view, fitness_config())
        || !owner.canonicalize_retained_summary_v1(
            1, lineage_metrics_config()
        )) {
        return false;
    }
    capture = capture_generation(1, owner);
    return owner.committed();
}

struct SequenceCaptureV1 {
    bool success = false;
    GenerationCaptureV1 generation_zero{};
    GenerationCaptureV1 generation_one{};
};

SequenceCaptureV1 run_sequence() {
    SequenceCaptureV1 sequence{};
    auto owner = std::make_unique<LedgerV1>(
        static_cast<std::int32_t>(kFixtureRows),
        static_cast<std::int32_t>(kFixtureSlots)
    );
    sequence.success = run_generation_zero(*owner, sequence.generation_zero)
        && run_generation_one(*owner, sequence.generation_one);
    return sequence;
}

bool bytes_equal(const void* left, const void* right, const std::size_t size) {
    return std::memcmp(left, right, size) == 0;
}

bool captures_equal(
    const GenerationCaptureV1& left,
    const GenerationCaptureV1& right
) {
    return left.generation == right.generation
        && bytes_equal(&left.header, &right.header, sizeof(left.header))
        && bytes_equal(
            left.behavior.data(), right.behavior.data(), sizeof(left.behavior)
        )
        && bytes_equal(
            left.lineage.data(), right.lineage.data(), sizeof(left.lineage)
        )
        && bytes_equal(
            &left.canonical_result,
            &right.canonical_result,
            sizeof(left.canonical_result)
        );
}

py::dict lineage_state_to_dict(
    const canonical::LineageMetricsStateV1& state
) {
    py::dict output;
    output["collapse_streak"] = state.collapse_streak;
    output["population_dominance_streak"] =
        state.population_dominance_streak;
    output["reproduction_dominance_streak"] =
        state.reproduction_dominance_streak;
    return output;
}

py::dict canonical_result_to_dict(
    const canonical::HostCanonicalResultV1& result
) {
    py::dict output;
    output["status"] = result.status;
    output["failed_field"] = result.failed_field;
    output["failed_row"] = result.failed_row;
    output["behavior_digest"] = py::bytes(
        reinterpret_cast<const char*>(result.behavior_rows_digest_sha256),
        canonical::kDigestBytesV1
    );
    output["lineage_digest"] = py::bytes(
        reinterpret_cast<const char*>(result.lineage_records_digest_sha256),
        canonical::kDigestBytesV1
    );

    py::dict behavior;
    behavior["row_count"] = result.behavior.row_count;
    behavior["cohort_size"] = result.behavior.cohort_size;
    behavior["cohort_survivors"] = result.behavior.cohort_survivors;
    behavior["action_total"] = result.behavior.action_total;
    behavior["move_total"] = result.behavior.move_total;
    behavior["move_n"] = result.behavior.move_n;
    behavior["move_s"] = result.behavior.move_s;
    behavior["move_e"] = result.behavior.move_e;
    behavior["move_w"] = result.behavior.move_w;
    behavior["gather"] = result.behavior.gather;
    behavior["rest"] = result.behavior.rest;
    behavior["attack"] = result.behavior.attack;
    output["behavior"] = behavior;

    const canonical::LineageSummaryOutputV1& lineage_output = result.lineage;
    py::dict lineage_result;
    lineage_result["defined_mask"] =
        lineage_output.compact_abi_v2_defined_mask;
    lineage_result["record_count"] = lineage_output.record_count;
    lineage_result["active_lineages"] = lineage_output.active_lineages;
    lineage_result["rare_persistent_lineages"] =
        lineage_output.rare_persistent_lineages;
    lineage_result["collapse_streak"] = lineage_output.collapse_streak;
    lineage_result["dominant_population_share"] =
        lineage_output.dominant_population_share;
    lineage_result["top3_population_share"] =
        lineage_output.top3_population_share;
    lineage_result["population_entropy"] = lineage_output.population_entropy;
    if ((lineage_output.compact_abi_v2_defined_mask
         & canonical::kCompactAbiV2DefinedLineageReproductionEntropyV1) != 0U) {
        lineage_result["reproduction_entropy"] =
            lineage_output.reproduction_entropy;
        lineage_result["dominant_reproduction_share"] =
            lineage_output.dominant_reproduction_share;
    } else {
        lineage_result["reproduction_entropy"] = py::none();
        lineage_result["dominant_reproduction_share"] = py::none();
    }
    lineage_result["collapse_warning_code"] =
        lineage_output.collapse_warning_code;
    if ((lineage_output.compact_abi_v2_defined_mask
         & canonical::kCompactAbiV2DefinedLineageDominantTextV1) != 0U) {
        lineage_result["dominant_lineage_id"] = py::str(
            reinterpret_cast<const char*>(
                lineage_output.dominant_lineage_text_record + 1U
            ),
            lineage_output.dominant_lineage_text_record[0]
        );
    } else {
        lineage_result["dominant_lineage_id"] = py::none();
    }
    output["lineage"] = lineage_result;
    output["lineage_state_after"] =
        lineage_state_to_dict(result.lineage_state_after);
    return output;
}

py::list behavior_rows_to_list(const GenerationCaptureV1& capture) {
    py::list output;
    for (std::uint16_t index = 0; index < capture.header.behavior_row_count; ++index) {
        const authority::BehaviorAuthorityRowV1& row = capture.behavior[index];
        py::dict value;
        value["generation"] = capture.generation;
        value["agent_id"] = row.agent_id;
        value["cohort"] = row.agent_id < capture.header.start_id_ceiling
            ? "start" : "born";
        value["move_n"] = row.action_counts[0];
        value["move_s"] = row.action_counts[1];
        value["move_e"] = row.action_counts[2];
        value["move_w"] = row.action_counts[3];
        value["gather"] = row.action_counts[4];
        value["rest"] = row.action_counts[5];
        value["attack"] = row.action_counts[6];
        value["lifespan_steps"] = row.lifespan_steps;
        value["energy_collected"] = row.energy_collected_exact;
        value["offspring"] = row.offspring;
        value["attacks_made"] = row.attacks_made;
        value["fitness"] = row.base_fitness;
        value["alive_end"] =
            (row.flags & authority::kBehaviorAliveEndV1) != 0U ? 1 : 0;
        value["eligible_for_selection"] =
            (row.flags & authority::kBehaviorEligibleForSelectionV1) != 0U
            ? 1 : 0;
        value["kept_next"] =
            (row.flags & authority::kBehaviorKeptNextV1) != 0U ? 1 : 0;
        value["injury_end"] = static_cast<double>(row.injury_end);
        value["injury_delta"] = static_cast<double>(row.injury_end)
            - static_cast<double>(row.injury_start);
        value["aggression_bias"] =
            static_cast<double>(row.aggression_bias_f32);
        output.append(value);
    }
    return output;
}

py::list lineage_rows_to_list(const GenerationCaptureV1& capture) {
    py::list output;
    for (std::uint16_t index = 0; index < capture.header.lineage_record_count; ++index) {
        const authority::LineageAuthorityRowV1& row = capture.lineage[index];
        char lineage_id[32];
        std::snprintf(lineage_id, sizeof(lineage_id), "L%08d", row.founder_id);

        py::dict stats;
        stats["lineage_id"] = lineage_id;
        stats["founder_id"] = row.founder_id;
        stats["parent_ids"] = py::list();
        stats["parent_lineage_ids"] = py::list();
        stats["birth_generation"] = row.birth_generation;
        stats["birth_tick"] = row.birth_tick;
        stats["lineage_depth"] = row.lineage_depth;
        stats["offspring_count"] = row.offspring_count;
        stats["survival_steps"] = static_cast<double>(row.survival_steps);
        stats["energy_collected"] = row.energy_collected_exact;
        stats["first_repro_step"] = row.first_repro_step < 0
            ? py::object(py::none()) : py::cast(row.first_repro_step);
        stats["pred_err_count"] = row.pred_err_count;
        stats["pred_err_sum"] = static_cast<double>(row.pred_err_sum);
        stats["episode0_pred_err_mean"] = row.episode0_pred_err_mean;
        stats["identity_drift_ema"] = row.identity_drift_ema;
        stats["drive_jitter"] = static_cast<double>(row.drive_jitter);
        stats["ecological_role"] = "general_agent";
        stats["role_source"] = "default";
        stats["role_confidence"] = py::none();
        stats["patch_habitat_lineage_id"] = row.patch_habitat_lineage_id;

        py::dict genome;
        genome["hidden_size"] = row.hidden_size;
        genome["memory_dim"] = row.memory_dim;
        genome["use_gru"] = row.use_gru != 0U;
        genome["aggression_bias"] =
            static_cast<double>(row.aggression_bias_f32);
        genome["exploration_bias"] =
            static_cast<double>(row.exploration_bias_f32);

        py::list drives;
        for (const float drive : row.drives) {
            drives.append(static_cast<double>(drive));
        }
        py::dict value;
        value["agent_id"] = row.agent_id;
        value["energy"] = row.energy;
        value["health"] = static_cast<double>(row.health);
        value["injury"] = static_cast<double>(row.injury);
        value["drives"] = drives;
        value["y"] = row.y;
        value["x"] = row.x;
        value["stats"] = stats;
        value["genome"] = genome;
        output.append(value);
    }
    return output;
}

py::dict capture_to_dict(const GenerationCaptureV1& capture) {
    py::dict output;
    output["generation"] = capture.generation;
    output["start_id_ceiling"] = capture.header.start_id_ceiling;
    output["header_status"] = capture.header.status;
    output["behavior_rows"] = behavior_rows_to_list(capture);
    output["lineage_records"] = lineage_rows_to_list(capture);
    output["canonical"] = canonical_result_to_dict(capture.canonical_result);
    output["retained_row_copied"] = capture.retained_row_copied;
    output["retained_row"] = py::bytes(
        reinterpret_cast<const char*>(capture.retained_row.data()),
        capture.retained_row.size()
    );
    return output;
}

bool canonical_payload_is_clear(
    const canonical::HostCanonicalResultV1& result
) {
    const auto bytes_zero = [](const std::uint8_t* data, const std::size_t size) {
        for (std::size_t index = 0; index < size; ++index) {
            if (data[index] != 0U) {
                return false;
            }
        }
        return true;
    };
    return bytes_zero(
            result.behavior_rows_digest_sha256, canonical::kDigestBytesV1
        )
        && bytes_zero(
            result.lineage_records_digest_sha256, canonical::kDigestBytesV1
        )
        && result.behavior.row_count == 0
        && result.lineage.record_count == 0
        && result.lineage_state_after.collapse_streak == 0
        && result.lineage_state_after.population_dominance_streak == 0
        && result.lineage_state_after.reproduction_dominance_streak == 0;
}

bool prepare_simple_sealed_owner(LedgerV1& owner) {
    const std::array<std::int64_t, 1> active_ids{1};
    std::array<std::int32_t, kAgentCapacity> agent_to_slot{};
    agent_to_slot.fill(-1);
    agent_to_slot[1] = 0;
    std::array<float, kFixtureSlots> injury_by_slot{};
    if (!owner.initialize_generation_from_slot_state_v1(
            2, active_ids.data(), agent_to_slot.data(), injury_by_slot.data(),
            1, static_cast<std::int32_t>(agent_to_slot.size())
        )
        || !owner.count_action_v1(0, 1, 0)) {
        return false;
    }
    ResidentSourcesStorageV1 sources = empty_sources();
    set_live_slot(sources, 0, 1, 1, 0, 1, 4, 1.0, 0.0F, 0.125F);
    const authority::ResidentGenerationSourcesV1 source_view = sources.view();
    return owner.finalize_pre_turnover_v1(source_view, fitness_config());
}

py::dict run_g2_fixture() {
    const SequenceCaptureV1 first = run_sequence();
    const SequenceCaptureV1 repeat = run_sequence();
    py::dict output;
    output["success"] = first.success;
    output["b1_repeat_exact"] = first.success && repeat.success
        && captures_equal(first.generation_zero, repeat.generation_zero)
        && captures_equal(first.generation_one, repeat.generation_one);
    py::list generations;
    generations.append(capture_to_dict(first.generation_zero));
    generations.append(capture_to_dict(first.generation_one));
    output["generations"] = generations;
    return output;
}

py::dict run_failure_contracts() {
    py::dict output;

    auto maximum = std::make_unique<LedgerV1>(
        ledger::kMaximumRowCapacityV1,
        ledger::kMaximumSlotCapacityV1
    );
    output["maximum_capacity_valid"] = maximum->capacity_valid()
        && maximum->initialize_generation_from_slot_state_v1(
            0, nullptr, nullptr, nullptr, 0, 1
        );
    output["row_zero_rejected"] =
        !std::make_unique<LedgerV1>(0, 1)->capacity_valid();
    output["row_above_max_rejected"] = !std::make_unique<LedgerV1>(
        ledger::kMaximumRowCapacityV1 + 1, 1
    )->capacity_valid();
    output["slot_zero_rejected"] =
        !std::make_unique<LedgerV1>(1, 0)->capacity_valid();
    output["slot_above_max_rejected"] = !std::make_unique<LedgerV1>(
        1, ledger::kMaximumSlotCapacityV1 + 1
    )->capacity_valid();
    output["negative_lineage_state_rejected"] =
        !std::make_unique<LedgerV1>(1, 1, canonical::LineageMetricsStateV1{-1, 0, 0})
            ->initialize_generation_from_slot_state_v1(
            0, nullptr, nullptr, nullptr, 0, 1
        );

    auto overflow = std::make_unique<LedgerV1>(2, 4);
    const std::array<std::int64_t, 1> overflow_ids{1};
    std::array<std::int32_t, kAgentCapacity> overflow_map{};
    overflow_map.fill(-1);
    overflow_map[1] = 0;
    std::array<float, kFixtureSlots> overflow_injury{};
    const bool overflow_initialized =
        overflow->initialize_generation_from_slot_state_v1(
            2, overflow_ids.data(), overflow_map.data(),
            overflow_injury.data(), 1,
            static_cast<std::int32_t>(overflow_map.size())
        );
    const bool first_birth = overflow->append_ordinary_birth_v1(2, 1, 0.0F);
    const bool overflowing_birth = overflow->append_ordinary_birth_v1(3, 2, 0.0F);
    output["overflow_initialized"] = overflow_initialized;
    output["overflow_first_birth"] = first_birth;
    output["overflow_rejected"] = !overflowing_birth;
    output["overflow_header_status"] = overflow->header().status;
    output["overflow_wrapper_status"] = ledger::ledger_wrapper_status_name(
        overflow->wrapper_status()
    );
    output["overflow_committed"] = overflow->committed();

    auto write_after_seal = std::make_unique<LedgerV1>(2, 4);
    const bool write_prepared = prepare_simple_sealed_owner(*write_after_seal);
    const bool write_canonicalized = write_prepared
        && write_after_seal->canonicalize_retained_summary_v1(
            0, lineage_metrics_config()
        );
    const canonical::LineageMetricsStateV1 state_before_write =
        write_after_seal->lineage_metrics_state();
    const bool committed_before_write = write_after_seal->committed();
    const bool post_seal_action = write_after_seal->count_action_v1(0, 1, 0);
    output["write_after_seal_prepared"] = write_canonicalized;
    output["write_after_seal_committed_before"] = committed_before_write;
    output["write_after_seal_rejected"] = !post_seal_action;
    output["write_after_seal_header_status"] = write_after_seal->header().status;
    output["write_after_seal_committed_after"] = write_after_seal->committed();
    output["write_after_seal_payload_clear"] = canonical_payload_is_clear(
        write_after_seal->canonical_result()
    );
    output["write_after_seal_state_preserved"] = bytes_equal(
        &state_before_write,
        &write_after_seal->lineage_metrics_state(),
        sizeof(state_before_write)
    );

    auto canonical_failure = std::make_unique<LedgerV1>(2, 4);
    const bool canonical_failure_prepared =
        prepare_simple_sealed_owner(*canonical_failure);
    canonical::LineageMetricsConfigV1 invalid_config = lineage_metrics_config();
    invalid_config.population_dominance_threshold =
        std::numeric_limits<double>::quiet_NaN();
    const bool invalid_canonical =
        canonical_failure->canonicalize_retained_summary_v1(0, invalid_config);
    output["canonical_failure_prepared"] = canonical_failure_prepared;
    output["canonical_failure_rejected"] = !invalid_canonical;
    output["canonical_failure_status"] = static_cast<std::uint32_t>(
        canonical_failure->last_canonical_status()
    );
    output["canonical_failure_payload_clear"] = canonical_payload_is_clear(
        canonical_failure->canonical_result()
    );
    output["canonical_failure_not_committed"] = !canonical_failure->committed();
    const bool canonical_retry =
        canonical_failure->canonicalize_retained_summary_v1(
            0, lineage_metrics_config()
        );
    output["canonical_retry_committed"] =
        canonical_retry && canonical_failure->committed();
    output["canonical_retry_state"] = lineage_state_to_dict(
        canonical_failure->lineage_metrics_state()
    );
    return output;
}

py::dict storage_receipt() {
    py::dict output;
    output["maximum_row_capacity"] = ledger::kMaximumRowCapacityV1;
    output["maximum_slot_capacity"] = ledger::kMaximumSlotCapacityV1;
    output["ledger_bytes"] = sizeof(LedgerV1);
    output["header_bytes"] = sizeof(authority::GenerationAuthorityHeaderV1);
    output["behavior_row_bytes"] = sizeof(authority::BehaviorAuthorityRowV1);
    output["lineage_row_bytes"] = sizeof(authority::LineageAuthorityRowV1);
    output["canonical_result_bytes"] =
        sizeof(canonical::HostCanonicalResultV1);
    return output;
}

}  // namespace

PYBIND11_MODULE(eiel_cpu_compact_summary_reporting_ledger_v1_test, module) {
    module.def("run_g2_fixture", &run_g2_fixture);
    module.def("run_failure_contracts", &run_failure_contracts);
    module.def("storage_receipt", &storage_receipt);
}
