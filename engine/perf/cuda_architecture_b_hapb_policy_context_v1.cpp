#include "cuda_architecture_b_hapb_policy_context_v1.hpp"

#include <torch/extension.h>

#include "cuda_architecture_b_flattened_controller_v1.hpp"
#include "cuda_architecture_b_flattened_wdl_controller_v2.hpp"
#include "cuda_architecture_b_hapb_residual_wall_v1.hpp"

#include <chrono>
#include <cstring>
#include <vector>

namespace eiel::cuda_architecture_b::hapb_policy_context_v1 {
namespace {

thread_local const HapbPolicyContextV1* active_context_v1 = nullptr;
thread_local const HapbDirectRouteContextV1* active_direct_route_context_v1 =
    nullptr;

void check_tensor(
    const torch::Tensor& value,
    const char* const name,
    const bool cuda,
    const torch::ScalarType dtype,
    const torch::IntArrayRef sizes
) {
    TORCH_CHECK(value.defined() && value.is_contiguous(), "HAPB ", name,
        " must be defined contiguous storage");
    TORCH_CHECK(value.is_cuda() == cuda, "HAPB ", name,
        cuda ? " must be CUDA storage" : " must be CPU storage");
    TORCH_CHECK(value.scalar_type() == dtype && value.sizes() == sizes,
        "HAPB ", name, " has an incompatible dtype or shape");
}

}  // namespace

std::int64_t physical_cells_v1(const torch::Tensor& bank) {
    TORCH_CHECK(
        bank.defined() && bank.dim() >= 1,
        "HAPB physical cell count requires a batched tensor"
    );
    const auto cells = bank.size(0);
    TORCH_CHECK(
        cells >= kPhysicalCellsMinV1 && cells <= kPhysicalCellsMaxV1,
        "HAPB physical cell count is outside [1,256]"
    );
    return cells;
}

namespace {

void validate_content_row_map_v1(const torch::Tensor& content_row_map) {
    if (!content_row_map.defined()) return;
    const auto cells = physical_cells_v1(content_row_map);
    check_tensor(
        content_row_map, "content row map", false, torch::kInt64,
        {cells, kCapacityV1}
    );
    const auto* values = content_row_map.data_ptr<std::int64_t>();
    const auto limit = cells * kHotPrefixRowsV1;
    for (std::int64_t cell = 0; cell < cells; ++cell) {
        for (std::int64_t slot = 0; slot < kCapacityV1; ++slot) {
            const auto value = values[cell * kCapacityV1 + slot];
            TORCH_CHECK(
                value == -1 || (value >= 0 && value < limit
                    && value / kHotPrefixRowsV1 == cell),
                "HAPB content row map contains an invalid or cross-cell pool row"
            );
            if (slot < kHotPrefixRowsV1) {
                TORCH_CHECK(
                    value == cell * kHotPrefixRowsV1 + slot,
                    "HAPB prefix content row map must remain stable"
                );
            }
        }
    }
}

void validate_policy_traffic_counters_v1(const torch::Tensor& counters) {
    if (!counters.defined()) return;
    check_tensor(
        counters, "policy traffic counters", false, torch::kInt64, {8}
    );
}

torch::Tensor cpu_int64_vector(const std::vector<std::int64_t>& values) {
    auto result = torch::empty(
        {static_cast<std::int64_t>(values.size())},
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)
    );
    if (!values.empty()) {
        std::memcpy(
            result.data_ptr<std::int64_t>(), values.data(),
            values.size() * sizeof(std::int64_t)
        );
    }
    return result;
}

}  // namespace

void validate_hapb_direct_route_context_v1(
    const HapbDirectRouteContextV1& context
) {
    const auto cells = physical_cells_v1(context.canonical_host_policy_bank);
    check_tensor(
        context.canonical_host_policy_bank, "canonical host policy bank", false,
        torch::kFloat32, {cells, kCapacityV1, kPolicyWidthV1}
    );
    TORCH_CHECK(!context.canonical_host_policy_bank.is_pinned(),
        "HAPB canonical policy bank must remain pageable host authority");
    const auto tile_rows = context.pinned_staging_tile.size(0);
    TORCH_CHECK(tile_rows == 512 || tile_rows == 4096,
        "HAPB tile rows must be one admitted H1 tile size");
    check_tensor(
        context.pinned_staging_tile, "pinned staging tile", false,
        torch::kFloat32, {tile_rows, kPolicyWidthV1}
    );
    TORCH_CHECK(context.pinned_staging_tile.is_pinned(),
        "HAPB staging tile must be explicitly pinned");
    check_tensor(
        context.hot_prefix_cache, "hot prefix cache", true, torch::kFloat32,
        {cells, kHotPrefixRowsV1, kPolicyWidthV1}
    );
    check_tensor(
        context.cuda_tile_arena, "CUDA tile arena", true, torch::kFloat32,
        {tile_rows, kPolicyWidthV1}
    );
    TORCH_CHECK(
        context.hot_prefix_cache.device() == context.cuda_tile_arena.device(),
        "HAPB prefix and tile must share one CUDA device"
    );
    validate_content_row_map_v1(context.content_row_map);
    validate_policy_traffic_counters_v1(context.policy_traffic_counters);
}

void validate_hapb_policy_context_v1(const HapbPolicyContextV1& context) {
    validate_hapb_direct_route_context_v1({
        context.canonical_host_policy_bank,
        context.pinned_staging_tile,
        context.hot_prefix_cache,
        context.cuda_tile_arena,
    });
    check_tensor(
        context.stable_overflow_rows, "stable overflow rows", false,
        torch::kInt64, {context.stable_overflow_rows.numel()}
    );
    check_tensor(
        context.stable_overflow_active_rows, "stable overflow active rows", false,
        torch::kInt64, {context.stable_overflow_rows.numel()}
    );
    if (context.content_alias_enabled) {
        check_tensor(
            context.stable_overflow_content_rows,
            "stable overflow content rows", false, torch::kInt64,
            {context.stable_overflow_rows.numel()}
        );
    }
    const auto* rows = context.stable_overflow_rows.data_ptr<std::int64_t>();
    const auto* active_rows =
        context.stable_overflow_active_rows.data_ptr<std::int64_t>();
    const auto cells = physical_cells_v1(context.canonical_host_policy_bank);
    std::vector<bool> seen(static_cast<std::size_t>(cells * kCapacityV1), false);
    std::vector<bool> active_seen(
        static_cast<std::size_t>(cells * kCapacityV1), false
    );
    for (std::int64_t index = 0; index < context.stable_overflow_rows.numel(); ++index) {
        const auto row = rows[index];
        TORCH_CHECK(row >= 0 && row < cells * kCapacityV1
                && row % kCapacityV1 >= kHotPrefixRowsV1,
            "HAPB stable overflow rows must be valid overflow rows");
        TORCH_CHECK(!seen.at(static_cast<std::size_t>(row)),
            "HAPB stable overflow rows must be unique in settlement order");
        seen.at(static_cast<std::size_t>(row)) = true;
        const auto active_row = active_rows[index];
        TORCH_CHECK(active_row >= 0 && active_row < cells * kCapacityV1
                && active_row / kCapacityV1 == row / kCapacityV1,
            "HAPB stable active rows must address the same resident cell");
        TORCH_CHECK(!active_seen.at(static_cast<std::size_t>(active_row)),
            "HAPB stable active rows must be unique in settlement order");
        active_seen.at(static_cast<std::size_t>(active_row)) = true;
        if (context.content_alias_enabled) {
            const auto content_row =
                context.stable_overflow_content_rows.data_ptr<std::int64_t>()[index];
            TORCH_CHECK(
                content_row == -1
                    || (content_row >= 0
                        && content_row < cells * kHotPrefixRowsV1
                        && content_row / kHotPrefixRowsV1 == row / kCapacityV1),
                "HAPB active overflow slot has an invalid content alias"
            );
        }
    }
    const auto device = context.cuda_tile_arena.device();
    for (const auto* entry : {
             &context.jungle_pre_predator_policy_pcg64_state,
             &context.jungle_pre_predator_policy_pcg64_buffer,
         }) {
        TORCH_CHECK(entry->defined() && entry->is_cuda() && entry->is_contiguous()
                && entry->device() == device
                && (entry->scalar_type() == torch::kInt64
                    || entry->scalar_type() == torch::kUInt64),
            "HAPB Jungle policy cursor must be contiguous CUDA int64/uint64");
    }
    TORCH_CHECK(
        context.jungle_pre_predator_policy_pcg64_state.sizes()
            == torch::IntArrayRef({cells, 4})
            && context.jungle_pre_predator_policy_pcg64_buffer.sizes()
                == torch::IntArrayRef({cells, 2}),
        "HAPB Jungle pre-predator cursor snapshot has an incompatible shape"
    );
}

void install_hapb_direct_route_context_v1(
    const HapbDirectRouteContextV1& context
) {
    TORCH_CHECK(active_direct_route_context_v1 == nullptr,
        "HAPB direct route contexts must not nest");
    validate_hapb_direct_route_context_v1(context);
    active_direct_route_context_v1 = &context;
}

void clear_hapb_direct_route_context_v1() noexcept {
    active_direct_route_context_v1 = nullptr;
}

const HapbDirectRouteContextV1& current_hapb_direct_route_context_v1() {
    TORCH_CHECK(active_direct_route_context_v1 != nullptr,
        "HAPB direct route context is not installed");
    return *active_direct_route_context_v1;
}

bool hapb_direct_route_context_active_v1() noexcept {
    return active_direct_route_context_v1 != nullptr;
}

HapbPolicyContextV1 materialize_hapb_policy_context_v1(
    const torch::Tensor& stable_overflow_rows,
    const torch::Tensor& stable_overflow_active_rows,
    const torch::Tensor& jungle_pre_predator_policy_pcg64_state,
    const torch::Tensor& jungle_pre_predator_policy_pcg64_buffer,
    const torch::Tensor& stable_overflow_content_rows
) {
    const auto& route = current_hapb_direct_route_context_v1();
    HapbPolicyContextV1 context{
        route.canonical_host_policy_bank,
        route.pinned_staging_tile,
        route.hot_prefix_cache,
        route.cuda_tile_arena,
        stable_overflow_rows,
        stable_overflow_active_rows,
        stable_overflow_content_rows,
        route.content_row_map,
        jungle_pre_predator_policy_pcg64_state,
        jungle_pre_predator_policy_pcg64_buffer,
        route.policy_traffic_counters,
        route.content_alias_enabled,
    };
    validate_hapb_policy_context_v1(context);
    return context;
}

torch::Tensor execute_hapb_prepared_policy_boundary_v1(
    const HapbPreparedPolicyBoundaryV1& boundary
) {
    TORCH_CHECK(
        hapb_direct_route_context_active_v1(),
        "HAPB prepared policy boundary requires an installed direct-route context"
    );
    const auto require_cuda = [](const torch::Tensor& value, const char* name,
                                 const torch::ScalarType dtype, const int64_t rank) {
        TORCH_CHECK(
            value.defined() && value.is_cuda() && value.is_contiguous()
                && value.scalar_type() == dtype && value.dim() == rank,
            "HAPB prepared policy boundary ", name, " has incompatible CUDA storage"
        );
    };
    require_cuda(boundary.ordered_slots, "ordered slots", torch::kInt32, 2);
    require_cuda(boundary.observations, "observations", torch::kFloat32, 3);
    require_cuda(boundary.stable_miss_rows, "stable miss rows", torch::kInt64, 2);
    require_cuda(boundary.active_row_counts, "active row counts", torch::kInt64, 1);
    require_cuda(boundary.miss_row_counts, "miss row counts", torch::kInt64, 1);
    require_cuda(boundary.status, "status", torch::kInt32, 1);
    require_cuda(boundary.drives, "drives", torch::kFloat32, 3);
    require_cuda(boundary.attack_bias, "attack bias", torch::kFloat32, 2);
    require_cuda(boundary.action_temperature, "action temperature", torch::kFloat32, 2);
    require_cuda(boundary.drive_logit_gain, "drive logit gain", torch::kFloat32, 2);
    require_cuda(boundary.slot_tokens, "slot tokens", torch::kInt64, 2);
    require_cuda(boundary.action_cache_actions, "action cache actions", torch::kInt32, 3);
    require_cuda(boundary.action_cache_token, "action cache token", torch::kInt64, 2);
    require_cuda(boundary.action_cache_cursor, "action cache cursor", torch::kInt32, 2);
    require_cuda(boundary.action_cache_next, "action cache next", torch::kInt32, 2);
    require_cuda(boundary.logits, "logits", torch::kFloat32, 3);
    require_cuda(boundary.row_actions, "row actions", torch::kInt32, 2);
    require_cuda(boundary.actions_by_slot, "actions by slot", torch::kInt32, 2);
    require_cuda(boundary.policy_pcg64_state_words, "policy PCG64 state", torch::kUInt64, 2);
    require_cuda(boundary.policy_pcg64_buffer_words, "policy PCG64 buffer", torch::kInt64, 2);
    require_cuda(boundary.policy_np_semantic_draws, "policy semantic draws", torch::kInt64, 1);
    const auto cells = physical_cells_v1(
        current_hapb_direct_route_context_v1().canonical_host_policy_bank
    );
    const auto require_shape = [](const torch::Tensor& value,
                                  const std::initializer_list<int64_t> expected,
                                  const char* name) {
        const auto expected_sizes = torch::IntArrayRef(expected);
        TORCH_CHECK(
            value.sizes() == expected_sizes,
            "HAPB prepared policy boundary ", name,
            " requires the physical B/C512 WDL layout; got ", value.sizes(),
            " expected ", expected_sizes
        );
    };
    require_shape(boundary.ordered_slots, {cells, kCapacityV1}, "ordered slots");
    require_shape(boundary.stable_miss_rows, {cells, kCapacityV1}, "stable miss rows");
    require_shape(boundary.slot_tokens, {cells, kCapacityV1}, "slot tokens");
    require_shape(boundary.action_cache_token, {cells, kCapacityV1}, "action cache token");
    require_shape(boundary.action_cache_cursor, {cells, kCapacityV1}, "action cache cursor");
    require_shape(boundary.action_cache_next, {cells, kCapacityV1}, "action cache next");
    require_shape(boundary.row_actions, {cells, kCapacityV1}, "row actions");
    require_shape(boundary.actions_by_slot, {cells, kCapacityV1}, "actions by slot");
    require_shape(boundary.observations, {cells, kCapacityV1, 83}, "observations");
    require_shape(boundary.drives, {cells, kCapacityV1, 4}, "drives");
    require_shape(boundary.attack_bias, {cells, kCapacityV1}, "attack bias");
    require_shape(boundary.action_temperature, {cells, kCapacityV1}, "action temperature");
    require_shape(boundary.drive_logit_gain, {cells, kCapacityV1}, "drive logit gain");
    require_shape(boundary.action_cache_actions, {cells, kCapacityV1, 2}, "action cache actions");
    require_shape(boundary.logits, {cells, kCapacityV1, 7}, "logits");
    require_shape(boundary.active_row_counts, {cells}, "active row counts");
    require_shape(boundary.miss_row_counts, {cells}, "miss row counts");
    require_shape(boundary.status, {cells}, "status");
    require_shape(boundary.policy_pcg64_state_words, {cells, 4}, "policy PCG64 state");
    require_shape(boundary.policy_pcg64_buffer_words, {cells, 2}, "policy PCG64 buffer");
    require_shape(boundary.policy_np_semantic_draws, {cells}, "policy semantic draws");
    if (boundary.policy_service_bypass_by_cell.defined()) {
        require_cuda(
            boundary.policy_service_bypass_by_cell, "policy service bypass",
            torch::kBool, 1
        );
        TORCH_CHECK(
            boundary.policy_service_bypass_by_cell.sizes()
                == torch::IntArrayRef({cells}),
            "HAPB prepared policy bypass has an incompatible shape"
        );
    }

    namespace residual = ::eiel::cuda_architecture_b::hapb_residual_wall_v1;
    const auto classify_started = std::chrono::steady_clock::now();
    const auto status_cpu = boundary.status.cpu().contiguous();
    const auto active_cpu = boundary.active_row_counts.cpu().contiguous();
    const auto misses_cpu = boundary.miss_row_counts.cpu().contiguous();
    const auto ordered_cpu = boundary.ordered_slots.cpu().contiguous();
    const auto miss_rows_cpu = boundary.stable_miss_rows.cpu().contiguous();
    const auto bypass_cpu = boundary.policy_service_bypass_by_cell.defined()
        ? boundary.policy_service_bypass_by_cell.cpu().contiguous()
        : torch::zeros({cells}, torch::TensorOptions().dtype(torch::kBool));
    const auto status = status_cpu.accessor<int32_t, 1>();
    const auto active = active_cpu.accessor<int64_t, 1>();
    const auto misses = misses_cpu.accessor<int64_t, 1>();
    const auto ordered = ordered_cpu.accessor<int32_t, 2>();
    const auto miss_rows = miss_rows_cpu.accessor<int64_t, 2>();
    const auto bypass = bypass_cpu.accessor<bool, 1>();
    std::vector<std::int64_t> hot_rows;
    std::vector<std::int64_t> hot_active_rows;
    std::vector<std::int64_t> overflow_rows;
    std::vector<std::int64_t> overflow_active_rows;
    std::vector<std::int64_t> overflow_content_rows;
    std::vector<std::int64_t> draw_increments(cells, 0);
    {
        const auto& route = current_hapb_direct_route_context_v1();
        const auto* content_map = route.content_alias_enabled
            && route.content_row_map.defined()
            ? route.content_row_map.data_ptr<std::int64_t>()
            : nullptr;
        for (std::int64_t cell = 0; cell < cells; ++cell) {
            TORCH_CHECK(
                status[cell] == 0 && !bypass[cell],
                "HAPB prepared policy boundary refuses failed or cross-path-bypassed cells"
            );
            TORCH_CHECK(
                active[cell] >= 0 && active[cell] <= kCapacityV1
                    && misses[cell] >= 0 && misses[cell] <= active[cell],
                "HAPB prepared policy counts are invalid"
            );
            std::vector<bool> seen(static_cast<std::size_t>(active[cell]), false);
            for (std::int64_t ordinal = 0; ordinal < misses[cell]; ++ordinal) {
                const auto active_row = miss_rows[cell][ordinal];
                TORCH_CHECK(
                    active_row >= 0 && active_row < active[cell]
                        && !seen.at(static_cast<std::size_t>(active_row)),
                    "HAPB prepared policy miss rows are not a stable active-row set"
                );
                seen.at(static_cast<std::size_t>(active_row)) = true;
                const auto slot = ordered[cell][active_row];
                TORCH_CHECK(
                    slot >= 0 && slot < kCapacityV1,
                    "HAPB prepared policy ordered slot is invalid"
                );
                const auto global_slot = cell * kCapacityV1 + slot;
                const auto global_active_row = cell * kCapacityV1 + active_row;
                const auto content_row = content_map == nullptr
                    ? global_slot
                    : content_map[global_slot];
                if (content_map != nullptr) {
                    TORCH_CHECK(
                        content_row == -1
                            || (content_row >= 0
                                && content_row < cells * kHotPrefixRowsV1
                                && content_row / kHotPrefixRowsV1 == cell),
                        "HAPB active logical slot has an invalid content alias"
                    );
                }
                if (slot < kHotPrefixRowsV1) {
                    hot_rows.push_back(global_slot);
                    hot_active_rows.push_back(global_active_row);
                } else {
                    overflow_rows.push_back(global_slot);
                    overflow_active_rows.push_back(global_active_row);
                    overflow_content_rows.push_back(content_row);
                }
            }
            draw_increments.at(static_cast<std::size_t>(cell)) =
                misses[cell] * 2 * 7;
        }
    }
    residual::add_ns(
        residual::classify_ns,
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - classify_started
        ).count()
    );
    const auto context = materialize_hapb_policy_context_v1(
        cpu_int64_vector(overflow_rows),
        cpu_int64_vector(overflow_active_rows),
        boundary.policy_pcg64_state_words,
        boundary.policy_pcg64_buffer_words,
        cpu_int64_vector(overflow_content_rows)
    );
    const HapbPolicyContextGuardV1 guard(context);
    torch::Tensor dense_logits;
    {
        residual::ScopeTimerV1 refill_timer(residual::refill_total_ns);
        dense_logits =
            flattened_wdl_controller_v2::hapb_refill_policy_logits_from_context_v1(
                boundary.observations.reshape({-1, 83}),
                cpu_int64_vector(hot_rows),
                cpu_int64_vector(hot_active_rows),
                boundary.drives.reshape({-1, 4}),
                boundary.attack_bias.reshape({-1}),
                boundary.action_temperature.reshape({-1}),
                boundary.drive_logit_gain.reshape({-1}),
                64, 7,
                boundary.w_hunger_gather,
                boundary.w_pain_rest,
                boundary.w_fatigue_rest,
                boundary.w_pain_attack,
                boundary.w_fear_attack
            ).reshape({cells, kCapacityV1, 7});
    }
    torch::Tensor counters;
    {
        residual::ScopeTimerV1 settle_timer(residual::settlement_ns);
        counters = flattened_controller_v1::hapb_settle_prepared_policy_rows_v1(
            boundary.ordered_slots,
            boundary.slot_tokens,
            boundary.action_cache_actions,
            boundary.action_cache_token,
            boundary.action_cache_cursor,
            boundary.action_cache_next,
            dense_logits,
            boundary.row_actions,
            boundary.stable_miss_rows,
            boundary.active_row_counts,
            boundary.miss_row_counts,
            boundary.policy_pcg64_state_words,
            boundary.policy_pcg64_buffer_words,
            boundary.w_hunger_gather,
            boundary.w_pain_rest,
            boundary.w_fatigue_rest,
            boundary.w_pain_attack,
            boundary.w_fear_attack
        );
        for (std::int64_t cell = 0; cell < cells; ++cell) {
            const auto active_rows = active[cell];
            const auto miss_count = misses[cell];
            const auto slots = boundary.ordered_slots.select(0, cell).narrow(0, 0, active_rows).to(torch::kLong);
            const auto actions = boundary.row_actions.select(0, cell).narrow(0, 0, active_rows);
            boundary.actions_by_slot.select(0, cell).index_copy_(0, slots, actions);
            if (miss_count > 0) {
                const auto miss_indices = boundary.stable_miss_rows.select(0, cell).narrow(0, 0, miss_count);
                const auto miss_logits = dense_logits.select(0, cell).index_select(0, miss_indices);
                boundary.logits.select(0, cell).index_copy_(0, miss_indices, miss_logits);
            }
        }
    }
    const auto draw_increments_cpu = cpu_int64_vector(draw_increments);
    boundary.policy_np_semantic_draws.add_(
        draw_increments_cpu.to(boundary.policy_np_semantic_draws.device(), true)
    );
    return counters;
}

void install_hapb_policy_context_v1(const HapbPolicyContextV1& context) {
    TORCH_CHECK(active_context_v1 == nullptr,
        "HAPB policy contexts must not nest");
    validate_hapb_policy_context_v1(context);
    active_context_v1 = &context;
}

void clear_hapb_policy_context_v1() noexcept {
    active_context_v1 = nullptr;
}

const HapbPolicyContextV1& current_hapb_policy_context_v1() {
    TORCH_CHECK(active_context_v1 != nullptr,
        "HAPB policy context is not installed");
    return *active_context_v1;
}

HapbPolicyContextGuardV1::HapbPolicyContextGuardV1(
    const HapbPolicyContextV1& context
) {
    install_hapb_policy_context_v1(context);
    installed_ = true;
}

HapbPolicyContextGuardV1::~HapbPolicyContextGuardV1() {
    if (installed_) clear_hapb_policy_context_v1();
}

HapbDirectRouteContextGuardV1::HapbDirectRouteContextGuardV1(
    const HapbDirectRouteContextV1& context
) {
    install_hapb_direct_route_context_v1(context);
    installed_ = true;
}

HapbDirectRouteContextGuardV1::~HapbDirectRouteContextGuardV1() {
    if (installed_) clear_hapb_direct_route_context_v1();
}

}  // namespace eiel::cuda_architecture_b::hapb_policy_context_v1
