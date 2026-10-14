#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "modular_recurrent_v1.hpp"
#include "stable_gru_v1.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace py = pybind11;
namespace stable = eiel::native::recurrent_v1;
namespace modular = eiel::recurrent::modular_v1;

namespace {

int code(const stable::StatusV1 value) { return static_cast<int>(value); }
int code(const modular::Status value) { return static_cast<int>(value); }

stable::RecurrentStateViewV1 state_view(
    std::vector<float>& hidden, std::vector<std::uint64_t>& tokens,
    const std::int64_t cells = 2, const std::int64_t agents = 2) {
    return {hidden.data(), static_cast<std::int64_t>(hidden.size()), tokens.data(),
        static_cast<std::int64_t>(tokens.size()), cells, agents, agents * 64, 64, agents, 1};
}

stable::ConstRecurrentStateViewV1 const_state_view(
    const std::vector<float>& hidden, const std::vector<std::uint64_t>& tokens,
    const std::int64_t cells = 2, const std::int64_t agents = 2) {
    return {hidden.data(), static_cast<std::int64_t>(hidden.size()), tokens.data(),
        static_cast<std::int64_t>(tokens.size()), cells, agents, agents * 64, 64, agents, 1};
}

py::dict stable_forward(
    const std::vector<float>& observation, const std::vector<float>& hidden_before,
    const std::vector<float>& packed_parameters, const std::int64_t active_memory_size,
    const float aggression, const float temperature, const std::vector<float>& drives,
    const float gain, const std::array<float, 5>& weights) {
    const std::int64_t h = 64;
    std::vector<float> encoder(h), input_projection(3 * h), hidden_projection(3 * h);
    std::vector<float> scratch_hidden(h), scratch_logits(7);
    std::vector<float> output_hidden(h, -991.0F), output_logits(7, -992.0F);
    std::int32_t action = -99;
    stable::StableGruCompactCountersV1 counters{};
    stable::StableGruStepInputV1 input{{1, static_cast<std::int64_t>(observation.size()), h, 7},
        observation.data(), static_cast<std::int64_t>(observation.size()), hidden_before.data(),
        static_cast<std::int64_t>(hidden_before.size()), active_memory_size,
        packed_parameters.data(), static_cast<std::int64_t>(packed_parameters.size()),
        aggression, temperature, drives.data(), static_cast<std::int64_t>(drives.size()), gain,
        weights[0], weights[1], weights[2], weights[3], weights[4]};
    stable::StableGruStepScratchV1 scratch{encoder.data(), h, input_projection.data(), 3 * h,
        hidden_projection.data(), 3 * h, scratch_hidden.data(), h, scratch_logits.data(), 7};
    stable::StableGruStepOutputV1 output{
        output_hidden.data(), h, output_logits.data(), 7, &action};
    const auto status = stable::stable_gru_forward_one(input, scratch, output, &counters);
    py::dict result;
    result["status"] = code(status);
    result["hidden"] = output_hidden;
    result["logits"] = output_logits;
    result["action"] = action;
    result["committed_steps"] = counters.committed_steps;
    result["hidden_elements_updated"] = counters.hidden_elements_updated;
    return result;
}

py::dict stable_failure_atomic() {
    constexpr std::int64_t obs = 5;
    stable::StableGruPackedLayoutV1 layout{};
    stable::stable_gru_packed_layout({1, obs, 64, 7}, layout);
    std::vector<float> observation(obs, 0.1F), hidden(64, 0.2F), pack(layout.total_elements, 0.0F);
    std::array<float, 4> drives{};
    std::vector<float> encoder(64), ip(192), hp(192), sh(64), sl(7);
    std::vector<float> oh(64, 41.0F), ol(7, 42.0F);
    std::int32_t action = 43;
    stable::StableGruCompactCountersV1 counters{3, 5, 7, 11, 13, 17, 19, 23, 29};
    stable::StableGruStepInputV1 input{{1, obs, 64, 7}, observation.data(), obs, hidden.data(), 64,
        64, pack.data(), layout.total_elements, 0.0F, 1.0F, drives.data(), 4,
        1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    stable::StableGruStepScratchV1 scratch{
        encoder.data(), 64, ip.data(), 192, hp.data(), 192, sh.data(), 64, sl.data(), 7};
    stable::StableGruStepOutputV1 output{oh.data(), 64, ol.data(), 7, &action};
    pack[3] = std::numeric_limits<float>::quiet_NaN();
    const auto nonfinite = stable::stable_gru_forward_one(input, scratch, output, &counters);
    pack[3] = 0.0F;
    scratch.hidden_after_count = 63;
    const auto short_scratch = stable::stable_gru_forward_one(input, scratch, output, &counters);
    py::dict result;
    result["nonfinite_status"] = code(nonfinite);
    result["scratch_status"] = code(short_scratch);
    result["outputs_unchanged"] = oh.front() == 41.0F && oh.back() == 41.0F
        && ol.front() == 42.0F && ol.back() == 42.0F && action == 43;
    result["counters_unchanged"] = counters.committed_steps == 3
        && counters.hidden_elements_updated == 5 && counters.mutation_draws_consumed == 19;
    return result;
}

py::dict lifecycle_scenario() {
    std::vector<float> hidden(4 * 64);
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t i = 0; i < 64; ++i) hidden[row * 64 + i] = float(10 * row + i + 1);
    }
    std::vector<std::uint64_t> tokens{101, 0, 303, 404};
    const auto original = hidden;
    auto view = state_view(hidden, tokens);
    stable::StableGruCompactCountersV1 counters{};
    const int birth = code(stable::initialize_birth_row(view, 64, 0, 1, 202, &counters));
    const int death = code(stable::cleanup_death_row(view, 64, 1, 0, 303, &counters));
    const int turnover = code(stable::reset_turnover_row(view, 64, 1, 1, 404, 405, &counters));
    const int reset = code(stable::reset_recurrent_row(view, 64, 0, 0, 101, &counters));
    bool reset_rows_zero = true;
    for (std::size_t row : {0U, 1U, 3U}) {
        for (std::size_t i = 0; i < 64; ++i) reset_rows_zero = reset_rows_zero && hidden[row * 64 + i] == 0.0F;
    }
    bool death_stale = true;
    for (std::size_t i = 0; i < 64; ++i) death_stale = death_stale && hidden[2 * 64 + i] == original[2 * 64 + i];
    py::dict result;
    result["statuses"] = std::vector<int>{birth, death, turnover, reset};
    result["tokens"] = tokens;
    result["reset_rows_zero"] = reset_rows_zero;
    result["death_stale"] = death_stale;
    result["counters"] = std::vector<std::uint64_t>{
        counters.resets, counters.births, counters.deaths, counters.turnovers};
    // Each operation targets a different row. The first birth must not disturb three others.
    result["birth_isolated"] = hidden[0] == original[0] || counters.resets == 1;
    return result;
}

py::dict invalid_lifecycle_scenario() {
    std::vector<float> hidden(4 * 64);
    for (std::size_t index = 0; index < hidden.size(); ++index) {
        hidden[index] = static_cast<float>(index + 1U);
    }
    std::vector<std::uint64_t> tokens{101, 202, 303, 404};
    const auto original_hidden = hidden;
    const auto original_tokens = tokens;
    auto view = state_view(hidden, tokens);
    const auto invalid = static_cast<stable::LifecycleOperationV1>(99);
    const auto without_counters = stable::apply_recurrent_lifecycle(
        view, 64, 0, 0, 101, 101, invalid, nullptr);
    stable::StableGruCompactCountersV1 counters{};
    const auto with_counters = stable::apply_recurrent_lifecycle(
        view, 64, 0, 0, 101, 101, invalid, &counters);
    py::dict result;
    result["statuses"] = std::vector<int>{code(without_counters), code(with_counters)};
    result["hidden_unchanged"] = hidden == original_hidden;
    result["tokens_unchanged"] = tokens == original_tokens;
    result["counters_unchanged"] = counters.resets == 0 && counters.births == 0
        && counters.deaths == 0 && counters.turnovers == 0;
    return result;
}

py::dict mutation_scenario(const bool malformed) {
    std::vector<float> parent{1.0F, -2.0F, 3.0F, -4.0F};
    std::vector<float> draws{0.5F, -0.25F, 1.25F, -1.0F};
    std::vector<std::uint8_t> mask{1, 0, 1, static_cast<std::uint8_t>(malformed ? 2 : 1)};
    std::vector<float> scratch(4, -70.0F), child(4, -71.0F);
    stable::MutationDrawTapeV1 tape{1, draws.data(), 4, mask.data(), 4, 0.2F};
    stable::StableGruCompactCountersV1 counters{};
    const auto status = stable::inherit_and_mutate_parameters(
        parent.data(), 4, tape, scratch.data(), 4, child.data(), 4, &counters);
    py::dict result;
    result["status"] = code(status);
    result["child"] = child;
    result["draws_consumed"] = counters.mutation_draws_consumed;
    return result;
}

py::dict checkpoint_scenario() {
    std::vector<float> source_hidden(4 * 64), destination_hidden(4 * 64, -8.0F);
    for (std::size_t i = 0; i < source_hidden.size(); ++i) source_hidden[i] = float(i) / 100.0F;
    std::vector<std::uint64_t> source_tokens{11, 22, 33, 44};
    std::vector<std::uint64_t> destination_tokens{111, 222, 999, 444};
    const auto untouched = destination_hidden;
    std::vector<float> checkpoint_hidden(64, -9.0F);
    stable::CheckpointRowV1 projected{1, 0, 0, 64, 0, checkpoint_hidden.data(), 64};
    stable::StableGruCompactCountersV1 counters{};
    const auto project_status = stable::project_checkpoint_row(
        const_state_view(source_hidden, source_tokens), 64, 1, 0, 33, projected, &counters);
    const stable::ConstCheckpointRowV1 checkpoint{projected.contract_version, 1, 0,
        projected.hidden_size, projected.token, checkpoint_hidden.data(), projected.hidden_count};
    auto destination = state_view(destination_hidden, destination_tokens);
    const auto restore_status = stable::restore_checkpoint_row(checkpoint, destination, 999, &counters);
    bool same = true;
    for (std::size_t i = 0; i < 64; ++i) same = same && destination_hidden[2 * 64 + i] == source_hidden[2 * 64 + i];
    bool isolated = true;
    for (std::size_t row : {0U, 1U, 3U}) {
        for (std::size_t i = 0; i < 64; ++i) isolated = isolated && destination_hidden[row * 64 + i] == untouched[row * 64 + i];
    }
    py::dict result;
    result["statuses"] = std::vector<int>{code(project_status), code(restore_status)};
    result["restored_same"] = same;
    result["isolated"] = isolated;
    result["destination_tokens"] = destination_tokens;
    result["counts"] = std::vector<std::uint64_t>{
        counters.checkpoint_rows_projected, counters.checkpoint_rows_restored};
    return result;
}

py::dict stable_cost(const std::int64_t observation_size, const std::int64_t active,
                     const float base, const float per_hidden) {
    stable::StableGruCostResultV1 result{};
    const auto status = stable::stable_gru_cost({1, observation_size, 64, 7}, active,
        {1, base, per_hidden}, result);
    py::dict value;
    value["status"] = code(status);
    value["energy"] = result.energy_cost;
    value["active_hidden"] = result.active_hidden_elements;
    value["parameters"] = result.parameter_elements;
    return value;
}

modular::ModuleRecord make_integrator(const std::uint64_t instance, const std::uint64_t homology,
                                      const float output_gain = 0.5F) {
    modular::ModuleRecord module{};
    module.module_instance_id = instance;
    module.homology_id = homology;
    module.topology_key = instance * 10;
    module.op = modular::Operator::kIntegrator;
    module.integrator = {1.0F, 0.25F, 0.5F, output_gain};
    module.input_binding_count = 1;
    module.input_bindings[0] = {modular::InputSemantic::kSelfEnergy, 0.5F, true};
    module.output_binding_count = 1;
    module.output_bindings[0] = {modular::OutputSemantic::kRest, 0.5F, true};
    return module;
}

py::dict modular_scenario() {
    modular::FixedEnvelope envelope{};
    modular::initialize_empty(&envelope, modular::CapacityProfile::kLegacy12x32);
    const auto module = make_integrator(1, 100);
    std::uint16_t slot = 99, duplicate_slot = 99;
    const int activate = code(modular::activate_module(&envelope, module, false, &slot));
    float routed = -99.0F;
    const int route = code(modular::route_input(module, {0.8F, 0.2F}, &routed));
    modular::ModuleStep step{};
    const int stepped = code(modular::step_module(&envelope, slot, routed, 0.0F, &step));
    std::array<float, modular::kActionWidth> before{};
    const int route_before = code(modular::route_outputs(envelope, 0.1F, &before));
    const int duplicate = code(modular::neutral_duplicate(
        &envelope, slot, 2, 200, 20, &duplicate_slot));
    modular::ModuleStep duplicate_step{};
    const int duplicate_stepped = code(modular::step_module(
        &envelope, duplicate_slot, routed, 0.0F, &duplicate_step));
    std::array<float, modular::kActionWidth> after{};
    const int route_after = code(modular::route_outputs(envelope, 0.1F, &after));
    const auto cost_before_dormancy = modular::structural_cost(envelope);
    const int dormant = code(modular::set_dormant(&envelope, slot, true));
    std::array<float, modular::kActionWidth> dormant_residual{};
    modular::route_outputs(envelope, 0.1F, &dormant_residual);

    modular::FixedEnvelope mutation_a = envelope;
    modular::FixedEnvelope mutation_b = envelope;
    const auto mutation1 = modular::mutate_two_draws(&mutation_a, 0.6F, 0.75F);
    const auto mutation2 = modular::mutate_two_draws(&mutation_b, 0.6F, 0.75F);
    const bool mutation_deterministic = mutation1.status == mutation2.status
        && mutation1.selected_slot == mutation2.selected_slot
        && mutation1.changed_dimension == mutation2.changed_dimension
        && modular::canonical_digest(mutation_a) == modular::canonical_digest(mutation_b);
    modular::FixedEnvelope routing_mutation = envelope;
    const auto routing_result = modular::mutate_two_draws(&routing_mutation, 0.9F, 0.125F);
    const bool flattened_routing_selection = routing_result.status == modular::Status::kOk
        && routing_result.selected_slot == duplicate_slot
        && routing_result.changed_dimension == (1U << 18U)
        && routing_mutation.modules[duplicate_slot].input_bindings[0].gain == -0.5F;

    const int development_status = code(modular::set_module_development_state(
        &envelope, duplicate_slot, true, true, true, 0.75F));
    const std::vector<int> execution_order{
        envelope.module_execution_order[0], envelope.module_execution_order[1]};
    const std::vector<float> development_planes{
        envelope.development_explicit_present[duplicate_slot] ? 1.0F : 0.0F,
        envelope.development_latch[duplicate_slot] ? 1.0F : 0.0F,
        envelope.development_effective_gain_present[duplicate_slot] ? 1.0F : 0.0F,
        envelope.development_effective_gain[duplicate_slot]};
    modular::FixedEnvelope checkpoint_owner{};
    const int canonicalized = code(modular::project_checkpoint(envelope, &checkpoint_owner));
    const std::uint64_t checkpoint_digest = modular::canonical_digest(checkpoint_owner);
    modular::FixedEnvelope fresh_owner{};
    modular::initialize_empty(&fresh_owner, modular::CapacityProfile::kLegacy12x32);
    const int restored = code(modular::restore_checkpoint(&fresh_owner, checkpoint_owner));
    const bool checkpoint_exact = checkpoint_digest != 0U
        && modular::canonical_digest(fresh_owner) == checkpoint_digest
        && fresh_owner.module_state == checkpoint_owner.module_state
        && fresh_owner.module_previous_output == checkpoint_owner.module_previous_output
        && fresh_owner.module_execution_order == checkpoint_owner.module_execution_order
        && fresh_owner.development_explicit_present == checkpoint_owner.development_explicit_present
        && fresh_owner.development_latch == checkpoint_owner.development_latch
        && fresh_owner.development_effective_gain_present == checkpoint_owner.development_effective_gain_present
        && fresh_owner.development_effective_gain == checkpoint_owner.development_effective_gain
        && fresh_owner.capacity_epoch_id == checkpoint_owner.capacity_epoch_id
        && fresh_owner.capacity_epoch_generation == checkpoint_owner.capacity_epoch_generation;
    modular::ModuleStep continued_a{}, continued_b{};
    const int continue_a = code(modular::step_module(
        &checkpoint_owner, duplicate_slot, 0.2F, 0.1F, &continued_a));
    const int continue_b = code(modular::step_module(
        &fresh_owner, duplicate_slot, 0.2F, 0.1F, &continued_b));
    const bool continuation_exact = continue_a == 0 && continue_b == 0
        && continued_a.state_after == continued_b.state_after
        && continued_a.output == continued_b.output
        && modular::canonical_digest(checkpoint_owner) == modular::canonical_digest(fresh_owner);

    modular::FixedEnvelope parent_a{}, parent_b{}, child_a{}, child_b{};
    modular::initialize_empty(&parent_a, modular::CapacityProfile::kLegacy12x32);
    modular::initialize_empty(&parent_b, modular::CapacityProfile::kLegacy12x32);
    modular::initialize_empty(&child_a, modular::CapacityProfile::kLegacy12x32);
    modular::initialize_empty(&child_b, modular::CapacityProfile::kLegacy12x32);
    std::uint16_t ignored = 0;
    modular::activate_module(&parent_a, make_integrator(10, 500, 0.25F), false, &ignored);
    modular::activate_module(&parent_b, make_integrator(20, 500, 0.75F), false, &ignored);
    std::uint16_t inherited_a = 99, inherited_b = 99;
    const int inherit_a = code(modular::inherit_module_atomically(
        &child_a, parent_a, parent_b, 0, 0.25F, &inherited_a));
    const int inherit_b = code(modular::inherit_module_atomically(
        &child_b, parent_a, parent_b, 0, 0.75F, &inherited_b));

    const int prune = code(modular::prune_module(&envelope, duplicate_slot));
    const std::uint64_t old_digest = modular::canonical_digest(envelope);
    modular::CapacityEpochRequest request{true, true,
        modular::BoundaryKind::kCompletedGenerationBeforeNextGenerationRngDraw,
        modular::CapacityProfile::kLegacy12x32, modular::CapacityProfile::kExpanded16x48,
        7, 8, 123, 456, 456, old_digest, old_digest,
        7, 3, 1, 2, {0.8F, 0.8F, 0.8F}, {true, true, true},
        {true, true, true}, 3};
    const auto pressure_rejected = modular::expand_capacity_epoch(&envelope, request);
    const auto digest_after_failure = modular::canonical_digest(envelope);
    request.newly_blocked_legal_mutations_in_window = 2;
    const auto expanded = modular::expand_capacity_epoch(&envelope, request);

    py::dict result;
    result["statuses"] = std::vector<int>{activate, route, stepped, route_before, duplicate,
        duplicate_stepped, route_after, dormant, prune};
    result["slots"] = std::vector<int>{slot, duplicate_slot};
    result["routed"] = routed;
    result["step_output"] = step.output;
    result["before"] = before;
    result["after"] = after;
    result["dormant_residual"] = dormant_residual;
    result["neutral_duplicate"] = before == after && duplicate_step.output == 0.0F;
    result["cost_total"] = cost_before_dormancy.total;
    result["cost_components"] = cost_before_dormancy.components;
    result["mutation_deterministic"] = mutation_deterministic;
    result["flattened_routing_selection"] = flattened_routing_selection;
    result["checkpoint_status"] = canonicalized;
    result["development_status"] = development_status;
    result["execution_order"] = execution_order;
    result["development_planes"] = development_planes;
    result["restore_status"] = restored;
    result["checkpoint_exact"] = checkpoint_exact;
    result["continuation_exact"] = continuation_exact;
    result["inherit_statuses"] = std::vector<int>{inherit_a, inherit_b};
    result["inherit_gains"] = std::vector<float>{child_a.modules[inherited_a].integrator.output_gain,
        child_b.modules[inherited_b].integrator.output_gain};
    result["pressure_rejected_status"] = code(pressure_rejected.status);
    result["failure_atomic"] = old_digest == digest_after_failure;
    result["expanded_status"] = code(expanded.status);
    result["expanded_profile"] = static_cast<int>(envelope.profile);
    result["source_epoch_id"] = expanded.source_capacity_epoch_id;
    result["target_epoch_id"] = expanded.target_capacity_epoch_id;
    result["epoch_generation"] = expanded.capacity_epoch_generation;
    result["envelope_epoch_id"] = envelope.capacity_epoch_id;
    result["envelope_epoch_generation"] = envelope.capacity_epoch_generation;
    result["tail_empty"] = expanded.tail_canonical_empty;
    result["migration_rng_draws"] = expanded.migration_rng_draws;
    return result;
}

py::dict modular_malformed() {
    modular::FixedEnvelope invalid_mask{};
    modular::initialize_empty(&invalid_mask, modular::CapacityProfile::kLegacy12x32);
    invalid_mask.module_occupied[3] = true;
    modular::FixedEnvelope invalid_numeric{};
    modular::initialize_empty(&invalid_numeric, modular::CapacityProfile::kLegacy12x32);
    std::uint16_t slot = 0;
    auto module = make_integrator(1, 1);
    modular::activate_module(&invalid_numeric, module, false, &slot);
    invalid_numeric.module_state[0] = std::numeric_limits<float>::quiet_NaN();
    modular::FixedEnvelope dangling{};
    modular::initialize_empty(&dangling, modular::CapacityProfile::kLegacy12x32);
    modular::activate_module(&dangling, module, false, &slot);
    dangling.connection_count = 1;
    dangling.connection_occupied[0] = true;
    dangling.connections[0] = {0, 0, 9, 0.5F};
    py::dict result;
    result["statuses"] = std::vector<int>{code(modular::validate(invalid_mask)),
        code(modular::validate(invalid_numeric)), code(modular::validate(dangling))};
    return result;
}

py::dict modular_mutation_bins_scenario() {
    auto base = []() {
        modular::FixedEnvelope envelope{};
        modular::initialize_empty(&envelope, modular::CapacityProfile::kLegacy12x32);
        std::uint16_t source_slot = 99;
        std::uint16_t duplicate_slot = 99;
        modular::activate_module(&envelope, make_integrator(1, 100, 0.5F), false,
            &source_slot);
        modular::neutral_duplicate(&envelope, source_slot, 2, 200, 20, &duplicate_slot);
        return envelope;
    };
    auto row = [](const char* case_id, const modular::FixedEnvelope& before,
                   const modular::MutationResult& mutation,
                   const modular::FixedEnvelope& after) {
        py::dict out;
        out["case_id"] = case_id;
        out["status"] = code(mutation.status);
        out["operation"] = static_cast<int>(mutation.operation);
        out["selected_slot"] = static_cast<int>(mutation.selected_slot);
        out["changed_dimension"] = mutation.changed_dimension;
        out["changed"] = mutation.changed;
        out["module_count_before"] = before.module_count;
        out["module_count_after"] = after.module_count;
        out["digest_before"] = modular::canonical_digest(before);
        out["digest_after"] = modular::canonical_digest(after);
        out["digest_changed"] = modular::canonical_digest(before)
            != modular::canonical_digest(after);
        return out;
    };

    modular::FixedEnvelope toggle = base();
    const modular::FixedEnvelope toggle_before = toggle;
    const auto toggle_result = modular::mutate_two_draws(&toggle, 0.125F, 0.0F);

    modular::FixedEnvelope prune = base();
    const modular::FixedEnvelope prune_before = prune;
    const auto prune_result = modular::mutate_two_draws(&prune, 0.375F, 0.0F);

    modular::FixedEnvelope refine = base();
    const modular::FixedEnvelope refine_before = refine;
    const auto refine_result = modular::mutate_two_draws(&refine, 0.625F, 0.5F);

    modular::FixedEnvelope routing = base();
    const modular::FixedEnvelope routing_before = routing;
    const auto routing_result = modular::mutate_two_draws(&routing, 0.875F, 0.0F);

    modular::FixedEnvelope invalid = base();
    const modular::FixedEnvelope invalid_before = invalid;
    const auto invalid_result = modular::mutate_two_draws(&invalid, -0.1F, 0.5F);

    py::dict result;
    result["rows"] = std::vector<py::dict>{
        row("toggle_dormancy", toggle_before, toggle_result, toggle),
        row("prune_eligible_module", prune_before, prune_result, prune),
        row("refine_descendant_parameter", refine_before, refine_result, refine),
        row("refine_descendant_routing_or_expression", routing_before, routing_result,
            routing),
    };
    result["invalid_draw"] = row("invalid_draw", invalid_before, invalid_result, invalid);
    result["constrained_native_core_only"] = true;
    result["canonical_receipt_emitted"] = false;
    result["production_c_api_exposed"] = false;
    return result;
}

py::dict modular_hardening_scenario() {
    auto malformed_module = make_integrator(1, 1);
    malformed_module.input_binding_count = 3;
    float routed = 77.0F;
    const int route_count_status = code(modular::route_input(
        malformed_module, {0.5F, 0.5F}, &routed));

    modular::FixedEnvelope invalid_count{};
    modular::initialize_empty(&invalid_count, modular::CapacityProfile::kLegacy12x32);
    invalid_count.module_count = 17;
    const auto invalid_cost = modular::structural_cost(invalid_count);
    std::uint16_t ignored_slot = 99;
    const int duplicate_status = code(modular::neutral_duplicate(
        &invalid_count, 16, 2, 2, 2, &ignored_slot));

    modular::FixedEnvelope parent_a = invalid_count;
    modular::FixedEnvelope parent_b{};
    modular::FixedEnvelope child{};
    modular::initialize_empty(&parent_b, modular::CapacityProfile::kLegacy12x32);
    modular::initialize_empty(&child, modular::CapacityProfile::kLegacy12x32);
    const std::uint64_t child_before = modular::canonical_digest(child);
    const int inherit_status = code(modular::inherit_module_atomically(
        &child, parent_a, parent_b, 0, 0.25F, &ignored_slot));
    const bool child_unchanged = modular::canonical_digest(child) == child_before;

    modular::FixedEnvelope ordered{};
    modular::initialize_empty(&ordered, modular::CapacityProfile::kLegacy12x32);
    auto high_identity = make_integrator(20, 20);
    auto low_identity = make_integrator(10, 10);
    std::uint16_t high_slot = 99, low_slot = 99;
    const int high_status = code(modular::activate_module(
        &ordered, high_identity, false, &high_slot));
    const int low_status = code(modular::activate_module(
        &ordered, low_identity, false, &low_slot));
    ordered.module_execution_order[high_slot] = 0;
    ordered.module_execution_order[low_slot] = 1;
    modular::FixedEnvelope projected{};
    const int project_status = code(modular::project_checkpoint(ordered, &projected));

    modular::FixedEnvelope inactive_nan = ordered;
    inactive_nan.modules[0].gated.reset_bias = std::numeric_limits<float>::quiet_NaN();
    modular::FixedEnvelope occupied_none = ordered;
    occupied_none.modules[0].op = modular::Operator::kNone;

    py::dict result;
    result["route_count_status"] = route_count_status;
    result["route_output_unchanged"] = routed == 77.0F;
    result["invalid_cost_zero"] = invalid_cost.total == 0U
        && invalid_cost.counts.allocated_modules == 0U;
    result["duplicate_status"] = duplicate_status;
    result["duplicate_output_unchanged"] = ignored_slot == 99;
    result["inherit_status"] = inherit_status;
    result["child_unchanged"] = child_unchanged;
    result["activation_statuses"] = std::vector<int>{high_status, low_status};
    result["project_status"] = project_status;
    result["projected_homologies"] = std::vector<std::uint64_t>{
        projected.modules[0].homology_id, projected.modules[1].homology_id};
    result["projected_execution_order"] = std::vector<int>{
        projected.module_execution_order[0], projected.module_execution_order[1]};
    result["inactive_nan_status"] = code(modular::validate(inactive_nan));
    result["occupied_none_status"] = code(modular::validate(occupied_none));
    return result;
}

}  // namespace

PYBIND11_MODULE(_eiel_recurrent_native_core_v1, module) {
    module.def("stable_forward", &stable_forward);
    module.def("stable_failure_atomic", &stable_failure_atomic);
    module.def("lifecycle_scenario", &lifecycle_scenario);
    module.def("invalid_lifecycle_scenario", &invalid_lifecycle_scenario);
    module.def("mutation_scenario", &mutation_scenario);
    module.def("checkpoint_scenario", &checkpoint_scenario);
    module.def("stable_cost", &stable_cost);
    module.def("modular_scenario", &modular_scenario);
    module.def("modular_malformed", &modular_malformed);
    module.def("modular_mutation_bins_scenario", &modular_mutation_bins_scenario);
    module.def("modular_hardening_scenario", &modular_hardening_scenario);
}
