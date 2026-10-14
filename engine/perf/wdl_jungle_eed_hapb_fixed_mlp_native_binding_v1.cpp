#include "wdl_jungle_eed_hapb_fixed_mlp_native_binding_v1.hpp"

#include <torch/extension.h>

#include "cuda_architecture_b_flattened_wdl_controller_v2.hpp"
#include "cuda_architecture_b_flattened_controller_v1.hpp"
#include "cuda_architecture_b_hapb_policy_context_v1.hpp"
#include "cuda_architecture_b_hapb_residual_wall_v1.hpp"
#include "cuda_architecture_b_resident_window_refill_v1.hpp"

#include <stdexcept>

namespace eiel::cuda_architecture_b::hapb_fixed_mlp_native_binding_v1 {

namespace hapb_context = eiel::cuda_architecture_b::hapb_policy_context_v1;
namespace policy_window_identity =
    eiel::cuda_architecture_b::resident_window_refill_v1;

hapb_context::HapbPolicyContextV1 context_from_tensors_v1(
    const torch::Tensor& canonical_host_policy_bank,
    const torch::Tensor& pinned_staging_tile,
    const torch::Tensor& hot_prefix_cache,
    const torch::Tensor& cuda_tile_arena,
    const torch::Tensor& stable_overflow_rows,
    const torch::Tensor& stable_overflow_active_rows,
    const torch::Tensor& jungle_pre_predator_policy_pcg64_state,
    const torch::Tensor& jungle_pre_predator_policy_pcg64_buffer
) {
    return {
        canonical_host_policy_bank,
        pinned_staging_tile,
        hot_prefix_cache,
        cuda_tile_arena,
        stable_overflow_rows,
        stable_overflow_active_rows,
        torch::Tensor(),
        torch::Tensor(),
        jungle_pre_predator_policy_pcg64_state,
        jungle_pre_predator_policy_pcg64_buffer,
        torch::Tensor(),
        false,
    };
}

hapb_context::HapbPolicyContextV1 context_from_tensors_v2(
    const torch::Tensor& canonical_host_policy_bank,
    const torch::Tensor& pinned_staging_tile,
    const torch::Tensor& hot_prefix_cache,
    const torch::Tensor& cuda_tile_arena,
    const torch::Tensor& stable_overflow_rows,
    const torch::Tensor& stable_overflow_active_rows,
    const torch::Tensor& stable_overflow_content_rows,
    const torch::Tensor& content_row_map,
    const torch::Tensor& policy_traffic_counters,
    const bool content_alias_enabled,
    const torch::Tensor& jungle_pre_predator_policy_pcg64_state,
    const torch::Tensor& jungle_pre_predator_policy_pcg64_buffer
) {
    return {
        canonical_host_policy_bank,
        pinned_staging_tile,
        hot_prefix_cache,
        cuda_tile_arena,
        stable_overflow_rows,
        stable_overflow_active_rows,
        stable_overflow_content_rows,
        content_row_map,
        jungle_pre_predator_policy_pcg64_state,
        jungle_pre_predator_policy_pcg64_buffer,
        policy_traffic_counters,
        content_alias_enabled,
    };
}

hapb_context::HapbDirectRouteContextV1 direct_route_context_from_tensors_v1(
    const torch::Tensor& canonical_host_policy_bank,
    const torch::Tensor& pinned_staging_tile,
    const torch::Tensor& hot_prefix_cache,
    const torch::Tensor& cuda_tile_arena,
    const torch::Tensor& content_row_map,
    const torch::Tensor& policy_traffic_counters,
    hapb_context::HapbOrdinaryBirthTransactionV1 ordinary_birth_transaction = {},
    hapb_context::HapbTurnoverTransactionV1 turnover_transaction = {},
    const bool content_alias_enabled = true
) {
    return {
        canonical_host_policy_bank,
        pinned_staging_tile,
        hot_prefix_cache,
        cuda_tile_arena,
        content_row_map,
        policy_traffic_counters,
        content_alias_enabled,
        std::move(ordinary_birth_transaction),
        std::move(turnover_transaction),
    };
}

void bind_hapb_fixed_mlp_native_v1(pybind11::module_& module) {
    module.def(
        "eiel_cuda_architecture_b_hapb_create_untrusted_policy_window_identity_owner_v1",
        &policy_window_identity::create_untrusted_policy_window_identity_owner_v1,
        "Create an invalid transient owner that must hydrate before first consume"
    );
    module.def(
        "eiel_cuda_architecture_b_hapb_commit_policy_window_identity_owner_entry_v1",
        &policy_window_identity::commit_policy_window_identity_owner_entry_v1,
        "Commit one Python-validated native identity-owner result"
    );
    module.def(
        "eiel_cuda_architecture_b_hapb_export_policy_window_identity_owner_checkpoint_v1",
        &policy_window_identity::export_policy_window_identity_owner_checkpoint_v1,
        "Export logical native identity-owner state for exact checkpoint restore"
    );
    module.def(
        "eiel_cuda_architecture_b_hapb_restore_policy_window_identity_owner_checkpoint_v1",
        &policy_window_identity::restore_policy_window_identity_owner_checkpoint_v1,
        "Rebind a fresh native identity owner to an exact checkpoint state"
    );
    module.def(
        "eiel_cuda_architecture_b_hapb_compact_policy_logits_v1",
        &flattened_wdl_controller_v2::hapb_compact_policy_logits_v1,
        "HAPB compact mapped fixed-MLP logits"
    );
    module.def(
        "eiel_cuda_architecture_b_hapb_hot_prefix_policy_logits_v1",
        &flattened_wdl_controller_v2::hapb_hot_prefix_policy_logits_v1,
        "HAPB selected P256 indexed hot-prefix fixed-MLP logits"
    );
    module.def(
        "eiel_cuda_architecture_b_hapb_refill_policy_logits_from_context_v1",
        [](const torch::Tensor& canonical_host_policy_bank,
           const torch::Tensor& pinned_staging_tile,
           const torch::Tensor& hot_prefix_cache,
           const torch::Tensor& cuda_tile_arena,
           const torch::Tensor& stable_overflow_rows,
           const torch::Tensor& stable_overflow_active_rows,
           const torch::Tensor& jungle_pre_predator_policy_pcg64_state,
           const torch::Tensor& jungle_pre_predator_policy_pcg64_buffer,
           const torch::Tensor& observations,
           const torch::Tensor& stable_hot_prefix_rows,
           const torch::Tensor& stable_hot_prefix_active_rows,
           const torch::Tensor& drives,
           const torch::Tensor& attack_bias,
           const torch::Tensor& action_temperature,
           const torch::Tensor& drive_logit_gain,
           const int64_t hidden_size,
           const int64_t action_count,
           const float w_hunger_gather,
           const float w_pain_rest,
           const float w_fatigue_rest,
           const float w_pain_attack,
           const float w_fear_attack) {
            const auto context = context_from_tensors_v1(
                canonical_host_policy_bank,
                pinned_staging_tile,
                hot_prefix_cache,
                cuda_tile_arena,
                stable_overflow_rows,
                stable_overflow_active_rows,
                jungle_pre_predator_policy_pcg64_state,
                jungle_pre_predator_policy_pcg64_buffer
            );
            const hapb_context::HapbPolicyContextGuardV1 guard(context);
            return flattened_wdl_controller_v2::
                hapb_refill_policy_logits_from_context_v1(
                    observations,
                    stable_hot_prefix_rows,
                    stable_hot_prefix_active_rows,
                    drives,
                    attack_bias,
                    action_temperature,
                    drive_logit_gain,
                    hidden_size,
                    action_count,
                    w_hunger_gather,
                    w_pain_rest,
                    w_fatigue_rest,
                    w_pain_attack,
                    w_fear_attack
                );
        },
        "HAPB guarded host-authoritative compact policy refill"
    );
    module.def(
        "eiel_cuda_architecture_b_hapb_refill_policy_logits_from_context_v2",
        [](const torch::Tensor& canonical_host_policy_bank,
           const torch::Tensor& pinned_staging_tile,
           const torch::Tensor& hot_prefix_cache,
           const torch::Tensor& cuda_tile_arena,
           const torch::Tensor& stable_overflow_rows,
           const torch::Tensor& stable_overflow_active_rows,
           const torch::Tensor& stable_overflow_content_rows,
           const torch::Tensor& content_row_map,
           const torch::Tensor& policy_traffic_counters,
           const bool content_alias_enabled,
           const torch::Tensor& jungle_pre_predator_policy_pcg64_state,
           const torch::Tensor& jungle_pre_predator_policy_pcg64_buffer,
           const torch::Tensor& observations,
           const torch::Tensor& stable_hot_prefix_rows,
           const torch::Tensor& stable_hot_prefix_active_rows,
           const torch::Tensor& drives,
           const torch::Tensor& attack_bias,
           const torch::Tensor& action_temperature,
           const torch::Tensor& drive_logit_gain,
           const int64_t hidden_size,
           const int64_t action_count,
           const float w_hunger_gather,
           const float w_pain_rest,
           const float w_fatigue_rest,
           const float w_pain_attack,
           const float w_fear_attack) {
            const auto context = context_from_tensors_v2(
                canonical_host_policy_bank,
                pinned_staging_tile,
                hot_prefix_cache,
                cuda_tile_arena,
                stable_overflow_rows,
                stable_overflow_active_rows,
                stable_overflow_content_rows,
                content_row_map,
                policy_traffic_counters,
                content_alias_enabled,
                jungle_pre_predator_policy_pcg64_state,
                jungle_pre_predator_policy_pcg64_buffer
            );
            const hapb_context::HapbPolicyContextGuardV1 guard(context);
            return flattened_wdl_controller_v2::
                hapb_refill_policy_logits_from_context_v1(
                    observations,
                    stable_hot_prefix_rows,
                    stable_hot_prefix_active_rows,
                    drives,
                    attack_bias,
                    action_temperature,
                    drive_logit_gain,
                    hidden_size,
                    action_count,
                    w_hunger_gather,
                    w_pain_rest,
                    w_fatigue_rest,
                    w_pain_attack,
                    w_fear_attack
                );
        },
        "HAPB guarded policy refill with explicit alias and traffic maps"
    );
    module.def(
        "eiel_cuda_architecture_b_hapb_ordinary_birth_mutate_rows_v1",
        &flattened_wdl_controller_v2::hapb_ordinary_birth_mutate_rows_v1,
        "HAPB bounded exact CUDA ordinary fixed-MLP birth mutation"
    );
    module.def(
        "eiel_cuda_architecture_b_hapb_resident_policy_logits_oracle_v1",
        &flattened_wdl_controller_v2::hapb_resident_policy_logits_oracle_v1,
        "B1-only bounded resident-layout scalar-policy oracle"
    );
    module.def(
        "eiel_cuda_architecture_b_hapb_settle_prepared_policy_rows_v1",
        &flattened_controller_v1::hapb_settle_prepared_policy_rows_v1,
        "Exact resident serial settlement for an HAPB prepared policy transaction"
    );
    module.def(
        "eiel_cuda_architecture_b_hapb_execute_prepared_policy_boundary_v1",
        [](const torch::Tensor& canonical_host_policy_bank,
           const torch::Tensor& pinned_staging_tile,
           const torch::Tensor& hot_prefix_cache,
           const torch::Tensor& cuda_tile_arena,
           const torch::Tensor& stable_overflow_rows,
           const torch::Tensor& stable_overflow_active_rows,
           const torch::Tensor& jungle_pre_predator_policy_pcg64_state,
           const torch::Tensor& jungle_pre_predator_policy_pcg64_buffer,
           const torch::Tensor& observations,
           const torch::Tensor& stable_hot_prefix_rows,
           const torch::Tensor& stable_hot_prefix_active_rows,
           const torch::Tensor& drives,
           const torch::Tensor& attack_bias,
           const torch::Tensor& action_temperature,
           const torch::Tensor& drive_logit_gain,
           const torch::Tensor& ordered_slots,
           const torch::Tensor& slot_tokens,
           const torch::Tensor& action_cache_actions,
           const torch::Tensor& action_cache_token,
           const torch::Tensor& action_cache_cursor,
           const torch::Tensor& action_cache_next,
           const torch::Tensor& row_actions,
           const torch::Tensor& stable_miss_rows,
           const torch::Tensor& active_row_counts,
           const torch::Tensor& miss_row_counts,
           const float w_hunger_gather,
           const float w_pain_rest,
           const float w_fatigue_rest,
           const float w_pain_attack,
           const float w_fear_attack) {
            const auto context = context_from_tensors_v1(
                canonical_host_policy_bank,
                pinned_staging_tile,
                hot_prefix_cache,
                cuda_tile_arena,
                stable_overflow_rows,
                stable_overflow_active_rows,
                jungle_pre_predator_policy_pcg64_state,
                jungle_pre_predator_policy_pcg64_buffer
            );
            const hapb_context::HapbPolicyContextGuardV1 guard(context);
            const auto logits = flattened_wdl_controller_v2::
                hapb_refill_policy_logits_from_context_v1(
                    observations,
                    stable_hot_prefix_rows,
                    stable_hot_prefix_active_rows,
                    drives,
                    attack_bias,
                    action_temperature,
                    drive_logit_gain,
                    64,
                    7,
                    w_hunger_gather,
                    w_pain_rest,
                    w_fatigue_rest,
                    w_pain_attack,
                    w_fear_attack
                );
            return flattened_controller_v1::
                hapb_settle_prepared_policy_rows_v1(
                    ordered_slots,
                    slot_tokens,
                    action_cache_actions,
                    action_cache_token,
                    action_cache_cursor,
                    action_cache_next,
                    logits,
                    row_actions,
                    stable_miss_rows,
                    active_row_counts,
                    miss_row_counts,
                    jungle_pre_predator_policy_pcg64_state,
                    jungle_pre_predator_policy_pcg64_buffer,
                    w_hunger_gather,
                    w_pain_rest,
                    w_fatigue_rest,
                    w_pain_attack,
                    w_fear_attack
                );
        },
        "HAPB guarded native prepared-policy refill and exact settlement boundary"
    );
    module.def(
        "eiel_cuda_architecture_b_hapb_validate_policy_context_v1",
        [](const torch::Tensor& canonical_host_policy_bank,
           const torch::Tensor& pinned_staging_tile,
           const torch::Tensor& hot_prefix_cache,
           const torch::Tensor& cuda_tile_arena,
           const torch::Tensor& stable_overflow_rows,
           const torch::Tensor& stable_overflow_active_rows,
           const torch::Tensor& jungle_pre_predator_policy_pcg64_state,
           const torch::Tensor& jungle_pre_predator_policy_pcg64_buffer) {
            const auto context = context_from_tensors_v1(
                canonical_host_policy_bank,
                pinned_staging_tile,
                hot_prefix_cache,
                cuda_tile_arena,
                stable_overflow_rows,
                stable_overflow_active_rows,
                jungle_pre_predator_policy_pcg64_state,
                jungle_pre_predator_policy_pcg64_buffer
            );
            const hapb_context::HapbPolicyContextGuardV1 guard(context);
            static_cast<void>(hapb_context::current_hapb_policy_context_v1());
            return true;
        },
        "Validate and install one HAPB exact-policy context without resident fallback"
    );
    module.def(
        "eiel_cuda_architecture_b_hapb_validate_direct_route_context_v1",
        [](const torch::Tensor& canonical_host_policy_bank,
           const torch::Tensor& pinned_staging_tile,
           const torch::Tensor& hot_prefix_cache,
           const torch::Tensor& cuda_tile_arena) {
            const auto context = direct_route_context_from_tensors_v1(
                canonical_host_policy_bank,
                pinned_staging_tile,
                hot_prefix_cache,
                cuda_tile_arena,
                torch::Tensor(),
                torch::Tensor(),
                {},
                {},
                true
            );
            const hapb_context::HapbDirectRouteContextGuardV1 guard(context);
            static_cast<void>(hapb_context::current_hapb_direct_route_context_v1());
            return true;
        },
        "Validate one long-lived HAPB direct-route owner context"
    );
    module.def(
        "eiel_cuda_architecture_b_execute_direct_resident_wdl_jungle_eed_hapb_v1",
        [module](const torch::Tensor& canonical_host_policy_bank,
            const torch::Tensor& pinned_staging_tile,
           const torch::Tensor& hot_prefix_cache,
           const torch::Tensor& cuda_tile_arena,
           const torch::Tensor& content_row_map,
           const torch::Tensor& policy_traffic_counters,
           const bool content_alias_enabled,
           pybind11::object ordinary_birth_transaction,
           pybind11::object turnover_transaction,
           pybind11::args arguments,
           pybind11::kwargs keyword_arguments) -> pybind11::object {
            pybind11::object policy_window_identity_owner = pybind11::none();
            if (keyword_arguments.contains("policy_window_identity_owner")) {
                policy_window_identity_owner = keyword_arguments[
                    "policy_window_identity_owner"
                ];
                keyword_arguments.attr("pop")(
                    "policy_window_identity_owner"
                );
            }
            TORCH_CHECK(
                pybind11::len(keyword_arguments) == 0,
                "HAPB direct entry received an unsupported keyword"
            );
            hapb_context::HapbOrdinaryBirthTransactionV1 transaction;
            if (!ordinary_birth_transaction.is_none()) {
                const auto callback = ordinary_birth_transaction.cast<pybind11::function>();
                transaction = [callback](
                                  const torch::Tensor& planned_parent_slots,
                                  const torch::Tensor& planned_child_slots,
                                  const torch::Tensor& planned_count,
                                  const torch::Tensor& policy_delta_tape,
                                  const torch::Tensor& policy_mutation_mask_tape,
                                  const torch::Tensor& policy_semantic_draws_tape,
                                  const torch::Tensor& policy_absolute_cursor,
                                  const torch::Tensor& policy_window_base
                              ) {
                    pybind11::gil_scoped_acquire acquire;
                    callback(
                        planned_parent_slots,
                        planned_child_slots,
                        planned_count,
                        policy_delta_tape,
                        policy_mutation_mask_tape,
                        policy_semantic_draws_tape,
                        policy_absolute_cursor,
                        policy_window_base
                    );
                };
            }
            hapb_context::HapbTurnoverTransactionV1 turnover;
            if (!turnover_transaction.is_none()) {
                const auto callback = turnover_transaction.cast<pybind11::function>();
                turnover = [callback](
                               const torch::Tensor& planned_parent_slots,
                               const torch::Tensor& planned_mate_slots,
                               const torch::Tensor& planned_child_slots,
                               const torch::Tensor& planned_count,
                               const torch::Tensor& policy_delta_tape,
                               const torch::Tensor& policy_mutation_mask_tape,
                               const torch::Tensor& policy_crossover_mask_packed_tape,
                               const torch::Tensor& policy_absolute_cursor,
                               const torch::Tensor& policy_window_base
                           ) {
                    pybind11::gil_scoped_acquire acquire;
                    callback(
                        planned_parent_slots,
                        planned_mate_slots,
                        planned_child_slots,
                        planned_count,
                        policy_delta_tape,
                        policy_mutation_mask_tape,
                        policy_crossover_mask_packed_tape,
                        policy_absolute_cursor,
                        policy_window_base
                    );
                };
            }
            const auto context = direct_route_context_from_tensors_v1(
                canonical_host_policy_bank,
                pinned_staging_tile,
                hot_prefix_cache,
                cuda_tile_arena,
                content_row_map,
                policy_traffic_counters,
                std::move(transaction),
                std::move(turnover),
                content_alias_enabled
            );
            const hapb_context::HapbDirectRouteContextGuardV1 guard(context);
            const auto resident_direct = module.attr(
                "eiel_cuda_architecture_b_execute_direct_resident_wdl_jungle_eed_v1"
            ).cast<pybind11::function>();
            if (!policy_window_identity_owner.is_none()) {
                auto* const identity_owner = policy_window_identity::
                    policy_window_identity_owner_from_capsule_v1(
                        policy_window_identity_owner
                    );
                policy_window_identity::
                    PolicyWindowIdentityOwnerExecutionLeaseV1 identity_lease(
                        identity_owner
                    );
                const policy_window_identity::ScopedPolicyWindowIdentityStateV1
                    identity_scope(
                        identity_lease.state(), identity_lease.boundary()
                    );
                auto result = resident_direct(*arguments);
                TORCH_CHECK(
                    PyMapping_Check(result.ptr()),
                    "HAPB direct identity result must be a mapping"
                );
                const int64_t validation_token = identity_lease.
                    mark_native_returned_pending_python_validation();
                result.attr("__setitem__")(
                    "_hapb_policy_window_identity_validation_token_v1",
                    validation_token
                );
                return result;
            }
            return resident_direct(*arguments);
        },
        "Guarded HAPB direct trajectory with an optional host-authoritative birth boundary"
    );
    module.def(
        "eiel_cuda_architecture_b_execute_direct_resident_wdl_jungle_eed_hapb_tick_span_v1",
        [module](const torch::Tensor& canonical_host_policy_bank,
            const torch::Tensor& pinned_staging_tile,
           const torch::Tensor& hot_prefix_cache,
           const torch::Tensor& cuda_tile_arena,
           const torch::Tensor& content_row_map,
           const torch::Tensor& policy_traffic_counters,
           const bool content_alias_enabled,
           pybind11::object ordinary_birth_transaction,
           pybind11::object turnover_transaction,
           pybind11::args arguments,
           pybind11::kwargs keyword_arguments) -> pybind11::object {
            pybind11::object policy_window_identity_owner = pybind11::none();
            if (keyword_arguments.contains("policy_window_identity_owner")) {
                policy_window_identity_owner = keyword_arguments[
                    "policy_window_identity_owner"
                ];
                keyword_arguments.attr("pop")(
                    "policy_window_identity_owner"
                );
            }
            TORCH_CHECK(
                pybind11::len(keyword_arguments) == 0,
                "HAPB tick-span entry received an unsupported keyword"
            );
            hapb_context::HapbOrdinaryBirthTransactionV1 transaction;
            if (!ordinary_birth_transaction.is_none()) {
                const auto callback = ordinary_birth_transaction.cast<pybind11::function>();
                transaction = [callback](
                                  const torch::Tensor& planned_parent_slots,
                                  const torch::Tensor& planned_child_slots,
                                  const torch::Tensor& planned_count,
                                  const torch::Tensor& policy_delta_tape,
                                  const torch::Tensor& policy_mutation_mask_tape,
                                  const torch::Tensor& policy_semantic_draws_tape,
                                  const torch::Tensor& policy_absolute_cursor,
                                  const torch::Tensor& policy_window_base
                              ) {
                    pybind11::gil_scoped_acquire acquire;
                    callback(
                        planned_parent_slots,
                        planned_child_slots,
                        planned_count,
                        policy_delta_tape,
                        policy_mutation_mask_tape,
                        policy_semantic_draws_tape,
                        policy_absolute_cursor,
                        policy_window_base
                    );
                };
            }
            hapb_context::HapbTurnoverTransactionV1 turnover;
            if (!turnover_transaction.is_none()) {
                const auto callback = turnover_transaction.cast<pybind11::function>();
                turnover = [callback](
                               const torch::Tensor& planned_parent_slots,
                               const torch::Tensor& planned_mate_slots,
                               const torch::Tensor& planned_child_slots,
                               const torch::Tensor& planned_count,
                               const torch::Tensor& policy_delta_tape,
                               const torch::Tensor& policy_mutation_mask_tape,
                               const torch::Tensor& policy_crossover_mask_packed_tape,
                               const torch::Tensor& policy_absolute_cursor,
                               const torch::Tensor& policy_window_base
                           ) {
                    pybind11::gil_scoped_acquire acquire;
                    callback(
                        planned_parent_slots,
                        planned_mate_slots,
                        planned_child_slots,
                        planned_count,
                        policy_delta_tape,
                        policy_mutation_mask_tape,
                        policy_crossover_mask_packed_tape,
                        policy_absolute_cursor,
                        policy_window_base
                    );
                };
            }
            const auto context = direct_route_context_from_tensors_v1(
                canonical_host_policy_bank,
                pinned_staging_tile,
                hot_prefix_cache,
                cuda_tile_arena,
                content_row_map,
                policy_traffic_counters,
                std::move(transaction),
                std::move(turnover),
                content_alias_enabled
            );
            const hapb_context::HapbDirectRouteContextGuardV1 guard(context);
            const auto resident_tick_dispatch = module.attr(
                "eiel_cuda_architecture_b_execute_direct_resident_wdl_jungle_eed_tick_dispatch_v2"
            ).cast<pybind11::function>();
            constexpr pybind11::ssize_t kOverlayArgumentCountV2 = 9;
            TORCH_CHECK(
                pybind11::len(arguments) > kOverlayArgumentCountV2,
                "HAPB tick dispatch lacks physical trajectory arguments"
            );
            pybind11::tuple source_window(
                pybind11::len(arguments) - kOverlayArgumentCountV2
            );
            for (
                pybind11::ssize_t index = kOverlayArgumentCountV2;
                index < pybind11::len(arguments);
                ++index
            ) {
                source_window[index - kOverlayArgumentCountV2] = arguments[index];
            }
            pybind11::list source_windows;
            source_windows.append(std::move(source_window));
            pybind11::tuple dispatch_arguments(kOverlayArgumentCountV2 + 1);
            for (
                pybind11::ssize_t index = 0;
                index < kOverlayArgumentCountV2;
                ++index
            ) {
                dispatch_arguments[index] = arguments[index];
            }
            dispatch_arguments[kOverlayArgumentCountV2] = std::move(source_windows);
            if (!policy_window_identity_owner.is_none()) {
                auto* const identity_owner = policy_window_identity::
                    policy_window_identity_owner_from_capsule_v1(
                        policy_window_identity_owner
                    );
                policy_window_identity::
                    PolicyWindowIdentityOwnerExecutionLeaseV1 identity_lease(
                        identity_owner
                    );
                const policy_window_identity::ScopedPolicyWindowIdentityStateV1
                    identity_scope(
                        identity_lease.state(), identity_lease.boundary()
                    );
                auto result = resident_tick_dispatch(*dispatch_arguments);
                TORCH_CHECK(
                    PyMapping_Check(result.ptr()),
                    "HAPB tick-span identity result must be a mapping"
                );
                const int64_t validation_token = identity_lease.
                    mark_native_returned_pending_python_validation();
                result.attr("__setitem__")(
                    "_hapb_policy_window_identity_validation_token_v1",
                    validation_token
                );
                return result;
            }
            return resident_tick_dispatch(*dispatch_arguments);
        },
        "Guarded HAPB exact partial-tick trajectory with host-authoritative births"
    );
    module.def(
        "eiel_cuda_architecture_b_hapb_residual_wall_reset_v1",
        []() {
            eiel::cuda_architecture_b::hapb_residual_wall_v1::reset_v1();
        },
        "Reset residual-wall phase timers"
    );
    module.def(
        "eiel_cuda_architecture_b_hapb_residual_wall_snapshot_v1",
        []() {
            pybind11::dict out;
            for (const auto& [key, value] :
                 eiel::cuda_architecture_b::hapb_residual_wall_v1::snapshot_v1()) {
                out[pybind11::str(key)] = value;
            }
            return out;
        },
        "Read residual-wall phase timers as nanoseconds"
    );
}

}  // namespace eiel::cuda_architecture_b::hapb_fixed_mlp_native_binding_v1
