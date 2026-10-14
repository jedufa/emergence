#include "cpu_compact_summary_source_composer_v1.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace py = pybind11;
namespace composer =
    eiel::native::cpu_compact_summary_source_composer_v1;
namespace encoder = eiel::native::cpu_compact_summary_v1;
namespace compact_abi = eiel::cuda_architecture_b::compact_summary_v2;
namespace canonical =
    eiel::cuda_architecture_b::compact_summary_host_canonical_v1;

namespace {

template <typename T, std::size_t Size>
py::bytes plane_bytes(const std::array<T, Size>& plane) {
    return py::bytes(
        reinterpret_cast<const char*>(plane.data()),
        static_cast<py::ssize_t>(sizeof(plane))
    );
}

template <typename T>
bool object_is_bit_zero(const T& value) {
    const auto* const bytes =
        reinterpret_cast<const std::uint8_t*>(&value);
    for (std::size_t index = 0; index < sizeof(value); ++index) {
        if (bytes[index] != 0U) {
            return false;
        }
    }
    return true;
}

void dirty_row(encoder::CpuCompactSummaryRowV1& row) {
    row.validity.fill(0xA5U);
    row.defined_mask.fill(0xA5A5A5A5A5A5A5A5ULL);
    row.status_codes.fill(-1);
    row.integer_values.fill(-1);
    row.float_values.fill(1.25);
    row.digest_values.fill(0xA5U);
    row.text_values.fill(0xA5U);
}

void fill_digest(
    composer::digests::DigestV1& digest,
    const std::uint8_t byte
) {
    digest.bytes.fill(byte);
}

void copy_digest_bytes(
    const py::handle value,
    composer::digests::DigestV1& output
) {
    const std::string raw = py::cast<py::bytes>(value);
    if (raw.size() != output.bytes.size()) {
        throw py::value_error("every digest authority must contain 32 bytes");
    }
    std::memcpy(output.bytes.data(), raw.data(), output.bytes.size());
}

composer::ComposerInputV1 fixture_input(
    const bool patch,
    const bool lineage_defined
) {
    composer::ComposerInputV1 input{};
    input.input_version = composer::kComposerInputVersionV1;
    input.generation.summary_status_code = compact_abi::kSummaryStatusComplete;
    input.generation.world_code = patch
        ? compact_abi::kWorldPatch
        : compact_abi::kWorldFlat;
    input.generation.generation_start_generation = 3;
    input.generation.generation_start_tick = 0;
    input.generation.generation_end_generation = 3;
    input.generation.generation_end_tick_exclusive = 80;
    input.generation.next_generation_start_generation = 4;
    input.generation.next_generation_start_tick = 0;
    input.generation.alive = 6;
    input.generation.births = 2;
    input.generation.failed_births = 1;
    input.generation.deaths = 1;

    input.turnover.count = 1;
    input.turnover.status_code = compact_abi::kTurnoverStatusCommitted;

    input.lifecycle_authority.event_cursor_start = 10;
    input.lifecycle_authority.event_cursor_pre_turnover = 14;
    input.lifecycle_authority.event_cursor_end = 16;
    auto& lifecycle = input.lifecycle_authority.reduction;
    lifecycle.event_row_count = 6;
    lifecycle.generation_event_count = 4;
    lifecycle.turnover_event_count = 2;
    lifecycle.generation_counts_by_type = {2, 0, 1, 1};
    lifecycle.turnover_counts_by_type = {0, 2, 0, 0};
    lifecycle.deaths_energy_depleted = 1;
    lifecycle.deaths_health_depleted = 0;
    fill_digest(lifecycle.active_prefix_digest, 0xA0U);

    auto& identity = input.identity_authority;
    identity.population_next_id = 10;
    identity.active_agent_count = 8;
    identity.active_agent_id_min = 1;
    identity.active_agent_id_max = 9;
    identity.free_slot_count = 2;
    identity.active_agent_id_min_defined = 1U;
    identity.active_agent_id_max_defined = 1U;
    fill_digest(identity.active_agent_ids_digest, 0xA1U);

    auto& rng = input.rng_state_authority;
    rng.authority_complete_code = compact_abi::kBooleanStatusTrue;
    rng.export_non_consuming_code = compact_abi::kBooleanStatusTrue;
    rng.policy_mutation_tape_cursor = 1;
    fill_digest(rng.authority_digest, 0xA2U);
    fill_digest(rng.policy_mutation_tape_digest, 0xA3U);
    fill_digest(rng.authoritative_state_digest, 0xA5U);
    fill_digest(rng.canonical_checkpoint_storage_digest, 0xA6U);
    fill_digest(rng.next_generation_start_state_digest, 0xA7U);

    auto& metrics = input.generation_metrics;
    metrics.capacity = 10;
    metrics.free_slots = 4;
    metrics.birth_attempts = 3;
    metrics.native_failed_birth_event_rows = 1;
    metrics.injury_blocked_births = 0;
    metrics.moved = 4;
    metrics.gathered = 2;
    metrics.movement_attempts = 8;
    metrics.movement_blocked_occupied = 3;
    metrics.movement_blocked_bounds = 1;
    metrics.policy_loop_tick_calls = 80;
    metrics.policy_refill_calls = 2;
    metrics.policy_refill_rows = 5;
    metrics.policy_cache_hit_calls = 78;
    metrics.policy_cache_rows_requested = 20;
    metrics.policy_cache_rows_hit = 15;
    metrics.policy_cache_rows_missed = 5;
    metrics.policy_cache_rows_refilled = 5;
    metrics.policy_cache_full_refill_ticks = 1;
    metrics.policy_cache_partial_refill_ticks = 1;
    metrics.peak_injury = 0.5;

    if (patch) {
        auto& value = input.patch_metrics;
        value.present = 1U;
        value.patch_movement_energy_event_count = 4;
        value.patch_spawn_sync_rows = {2, 1U};
        value.typed_death_event_rows = {1, 1U};
        value.env_rng_padding_draws = {0, 1U};
        value.patch_food_harvest_total = 5.0;
        value.patch_scent_emit_total = 2.0;
        value.patch_waste_emit_total = 1.0;
        value.patch_movement_energy_total = 0.5;
        value.energy_collected_delta = {5.0, 1U};
        value.patch_scent_move_queued_total = {1.0, 1U};
        value.patch_waste_move_queued_total = {0.5, 1U};
        value.patch_scent_gather_queued_total = {1.0, 1U};
        value.patch_waste_gather_queued_total = {0.5, 1U};
        value.patch_movement_energy_extra_total = {0.5, 1U};
        value.patch_scent_emitted_total = {2.0, 1U};
        value.patch_waste_emitted_total = {1.0, 1U};
        value.patch_food_depleted_frac = {0.25, 1U};
        value.patch_food_full_frac = {0.5, 1U};
        value.patch_density_hotspot_frac = {0.125, 1U};
    }

    auto& host = input.host_canonical_authority;
    host.status = static_cast<std::uint32_t>(
        canonical::HostCanonicalStatusV1::kSuccess
    );
    host.failed_field = static_cast<std::uint32_t>(
        canonical::HostCanonicalFieldV1::kNone
    );
    host.failed_row = canonical::kNoFailedRowV1;
    std::memset(
        host.behavior_rows_digest_sha256,
        0xA8,
        sizeof(host.behavior_rows_digest_sha256)
    );
    std::memset(
        host.lineage_records_digest_sha256,
        0xA9,
        sizeof(host.lineage_records_digest_sha256)
    );
    host.behavior.row_count = 2;
    host.behavior.cohort_size = 1;
    host.behavior.cohort_survivors = 1;
    host.behavior.action_total = 13;
    host.behavior.move_total = 7;
    host.behavior.move_n = 3;
    host.behavior.move_s = 2;
    host.behavior.move_e = 0;
    host.behavior.move_w = 2;
    host.behavior.gather = 4;
    host.behavior.rest = 2;
    host.behavior.attack = 0;
    host.lineage.record_count = 2;
    if (lineage_defined) {
        host.lineage.compact_abi_v2_defined_mask =
            canonical::kCompactAbiV2DefinedLineageActiveV1
            | canonical::kCompactAbiV2DefinedLineageDominantTextV1
            | canonical::kCompactAbiV2DefinedLineageDominantPopulationShareV1
            | canonical::kCompactAbiV2DefinedLineageTop3ShareV1
            | canonical::kCompactAbiV2DefinedLineagePopulationEntropyV1
            | canonical::kCompactAbiV2DefinedLineageReproductionEntropyV1
            | canonical::kCompactAbiV2DefinedLineageDominantReproductionShareV1
            | canonical::kCompactAbiV2DefinedLineageRarePersistentV1
            | canonical::kCompactAbiV2DefinedLineageCollapseWarningV1
            | canonical::kCompactAbiV2DefinedLineageCollapseStreakV1;
        host.lineage.active_lineages = 2;
        host.lineage.rare_persistent_lineages = 1;
        host.lineage.collapse_streak = 0;
        host.lineage.dominant_population_share = 0.5;
        host.lineage.top3_population_share = 1.0;
        host.lineage.population_entropy = 0.9;
        host.lineage.reproduction_entropy = 0.8;
        host.lineage.dominant_reproduction_share = 0.5;
        host.lineage.collapse_warning_code = compact_abi::kBooleanStatusFalse;
        constexpr char kLineageId[] = "L00000001";
        host.lineage.dominant_lineage_text_record[0] =
            static_cast<std::uint8_t>(sizeof(kLineageId) - 1U);
        std::memcpy(
            host.lineage.dominant_lineage_text_record + 1U,
            kLineageId,
            sizeof(kLineageId) - 1U
        );
    }

    auto& terminal = input.terminal_checkpoint_authority;
    terminal.terminal_status_code = compact_abi::kTerminalStatusNotTerminal;
    terminal.recoverable_error_status_code =
        compact_abi::kRecoverableErrorStatusNone;
    terminal.sticky_failure_code = compact_abi::kBooleanStatusFalse;
    terminal.checkpoint_due_code = compact_abi::kBooleanStatusFalse;
    return input;
}

void apply_digest_authorities(
    const py::sequence digest_values,
    composer::ComposerInputV1& input
) {
    if (py::len(digest_values) != compact_abi::kDigestFieldCountV2) {
        throw py::value_error("digest authority list must match ABI field count");
    }
    copy_digest_bytes(
        digest_values[compact_abi::kDigestEventsActivePrefixDigestSha256],
        input.lifecycle_authority.reduction.active_prefix_digest
    );
    copy_digest_bytes(
        digest_values[
            compact_abi::kDigestIdentityActiveAgentIdsDigestSha256
        ],
        input.identity_authority.active_agent_ids_digest
    );
    copy_digest_bytes(
        digest_values[compact_abi::kDigestRngAuthorityDigestSha256],
        input.rng_state_authority.authority_digest
    );
    copy_digest_bytes(
        digest_values[
            compact_abi::kDigestRngPolicyMutationTapeDigestSha256
        ],
        input.rng_state_authority.policy_mutation_tape_digest
    );
    copy_digest_bytes(
        digest_values[
            compact_abi::kDigestStateDigestsAuthoritativeStateDigestSha256
        ],
        input.rng_state_authority.authoritative_state_digest
    );
    copy_digest_bytes(
        digest_values[
            compact_abi::
                kDigestStateDigestsCanonicalCheckpointStorageDigestSha256
        ],
        input.rng_state_authority.canonical_checkpoint_storage_digest
    );
    copy_digest_bytes(
        digest_values[
            compact_abi::
                kDigestStateDigestsNextGenerationStartStateDigestSha256
        ],
        input.rng_state_authority.next_generation_start_state_digest
    );

    composer::digests::DigestV1 digest{};
    copy_digest_bytes(
        digest_values[compact_abi::kDigestBehaviorMetricsRowsDigestSha256],
        digest
    );
    std::memcpy(
        input.host_canonical_authority.behavior_rows_digest_sha256,
        digest.bytes.data(),
        digest.bytes.size()
    );
    copy_digest_bytes(
        digest_values[compact_abi::kDigestLineageMetricsRecordsDigestSha256],
        digest
    );
    std::memcpy(
        input.host_canonical_authority.lineage_records_digest_sha256,
        digest.bytes.data(),
        digest.bytes.size()
    );
}

py::dict row_to_dict(const encoder::CpuCompactSummaryRowV1& row) {
    py::dict result;
    result["validity"] = plane_bytes(row.validity);
    result["defined_mask"] = plane_bytes(row.defined_mask);
    result["status_codes"] = plane_bytes(row.status_codes);
    result["integer_values"] = plane_bytes(row.integer_values);
    result["float_values"] = plane_bytes(row.float_values);
    result["digest_values"] = plane_bytes(row.digest_values);
    result["text_values"] = plane_bytes(row.text_values);
    return result;
}

py::dict compose_fixture(
    const bool patch,
    const bool lineage_defined,
    const py::sequence digest_values
) {
    composer::ComposerInputV1 input = fixture_input(patch, lineage_defined);
    apply_digest_authorities(digest_values, input);
    encoder::CpuCompactSummaryRowV1 row{};
    const composer::ComposeStatusV1 status =
        composer::compose_and_encode_completed_row(input, row);
    py::dict result;
    result["status"] = composer::compose_status_name(status);
    result["row"] = row_to_dict(row);
    return result;
}

template <typename Mutator>
void append_failure(
    py::list& output,
    const char* const name,
    Mutator mutate
) {
    composer::ComposerInputV1 input = fixture_input(false, true);
    mutate(input);

    encoder::CpuCompactSummarySourceAuthorityV1 source{};
    std::memset(&source, 0xA5, sizeof(source));
    const composer::ComposeStatusV1 source_status =
        composer::compose_source_authority(input, source);

    encoder::CpuCompactSummaryRowV1 row{};
    dirty_row(row);
    const composer::ComposeStatusV1 row_status =
        composer::compose_and_encode_completed_row(input, row);
    py::dict item;
    item["name"] = name;
    item["source_status"] = composer::compose_status_name(source_status);
    item["row_status"] = composer::compose_status_name(row_status);
    item["source_bit_zero"] = object_is_bit_zero(source);
    item["row_bit_zero"] = object_is_bit_zero(row);
    output.append(item);
}

py::list failure_cases() {
    py::list output;
    append_failure(output, "input_version", [](auto& input) {
        input.input_version = 0U;
    });
    append_failure(output, "lifecycle_cursor", [](auto& input) {
        ++input.lifecycle_authority.event_cursor_end;
    });
    append_failure(output, "identity_capacity", [](auto& input) {
        ++input.identity_authority.free_slot_count;
    });
    append_failure(output, "missing_lifecycle_digest", [](auto& input) {
        input.lifecycle_authority.reduction.active_prefix_digest.bytes.fill(0U);
    });
    append_failure(output, "missing_state_digest", [](auto& input) {
        input.rng_state_authority.authoritative_state_digest.bytes.fill(0U);
    });
    append_failure(output, "recoverable_terminal_mismatch", [](auto& input) {
        input.terminal_checkpoint_authority.recoverable_error_status_code =
            compact_abi::kRecoverableErrorStatusPresent;
    });
    append_failure(output, "failed_terminal_mismatch", [](auto& input) {
        input.terminal_checkpoint_authority.terminal_status_code =
            compact_abi::kTerminalStatusFailed;
    });
    append_failure(output, "host_canonical_failure", [](auto& input) {
        input.host_canonical_authority.status = static_cast<std::uint32_t>(
            canonical::HostCanonicalStatusV1::kLineageCounter
        );
    });
    return output;
}

py::dict abi_receipt() {
    py::dict result;
    result["input_bytes"] = sizeof(composer::ComposerInputV1);
    result["input_standard_layout"] =
        std::is_standard_layout_v<composer::ComposerInputV1>;
    result["input_trivially_copyable"] =
        std::is_trivially_copyable_v<composer::ComposerInputV1>;
    result["plane_count"] = static_cast<std::int64_t>(
        compact_abi::kPlaneCountV2
    );
    return result;
}

}  // namespace

PYBIND11_MODULE(
    eiel_cpu_compact_summary_source_composer_v1_test,
    module
) {
    module.def("abi_receipt", &abi_receipt);
    module.def("compose_fixture", &compose_fixture);
    module.def("failure_cases", &failure_cases);
}
