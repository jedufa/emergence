#include "cuda_architecture_b_modular_tick.hpp"

#include <ATen/ops/_assert_async.h>

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {

using torch::Tensor;

constexpr int64_t kObservationWidth = 27;
constexpr int64_t kLocalObservationWidth = 25;
constexpr int64_t kActionWidth = 7;
constexpr int64_t kModuleSlots = 16;
constexpr int64_t kConnectionSlots = 48;
constexpr int64_t kRestAction = 5;
constexpr int64_t kGenericModule = 1;
constexpr int64_t kIntegratorOperator = 1;
constexpr int64_t kGatedOperator = 2;

enum Plane : int64_t {
    AgentOccupied = 0,
    AgentId,
    AgentSlotToken,
    ModuleCount,
    ConnectionCount,
    ModuleOccupied,
    ModuleLocalIdIndex,
    ModuleInstanceIdIndex,
    ModuleHomologyIdIndex,
    ModuleTypeIndex,
    ModuleTypeCode,
    IdentityOriginCode,
    TopologyPosition,
    ModuleExecutionOrder,
    DependencyTopologicalOrder,
    OperatorCode,
    IntegratorParameters,
    GatedParameters,
    ModuleDormant,
    ModuleStatePresent,
    ModuleState,
    ModuleOutputPresent,
    ModulePreviousOutput,
    InputBindingCount,
    InputSemanticCode,
    InputSemanticIdIndex,
    InputGain,
    InputEnabled,
    OutputBindingCount,
    OutputSemanticCode,
    OutputSemanticIdIndex,
    OutputActionIndex,
    OutputGain,
    OutputEnabled,
    DevelopmentExplicit,
    DevelopmentStageGains,
    DevelopmentPrerequisiteCount,
    DevelopmentPrerequisiteHomologyIndex,
    DevelopmentPrerequisiteModuleSlot,
    DevelopmentLatchEnabled,
    DevelopmentLatchPresent,
    DevelopmentLatchState,
    DevelopmentEffectiveGainPresent,
    DevelopmentEffectiveGain,
    ConnectionOccupied,
    ConnectionLocalIdIndex,
    ConnectionSrcModuleSlot,
    ConnectionDstModuleSlot,
    ConnectionSrcSlotIndex,
    ConnectionDstSlotIndex,
    ConnectionValueTypeIndex,
    ConnectionChannelIndex,
    ConnectionWeight,
    ConnectionExecutionOrder,
    StructuralCostCounts,
    StructuralCostComponents,
    StructuralCostTotal,
};

const std::array<const char*, eiel::cuda_architecture_b::kModularTickPlaneCount>
    kPlaneNames = {{
        "agent_occupied",
        "agent_id",
        "agent_slot_token",
        "module_count",
        "connection_count",
        "module_occupied",
        "module_local_id_index",
        "module_instance_id_index",
        "module_homology_id_index",
        "module_type_index",
        "module_type_code",
        "identity_origin_code",
        "topology_position",
        "module_execution_order",
        "dependency_topological_order",
        "operator_code",
        "integrator_parameters",
        "gated_parameters",
        "module_dormant",
        "module_state_present",
        "module_state",
        "module_output_present",
        "module_previous_output",
        "input_binding_count",
        "input_semantic_code",
        "input_semantic_id_index",
        "input_gain",
        "input_enabled",
        "output_binding_count",
        "output_semantic_code",
        "output_semantic_id_index",
        "output_action_index",
        "output_gain",
        "output_enabled",
        "development_explicit",
        "development_stage_gains",
        "development_prerequisite_count",
        "development_prerequisite_homology_index",
        "development_prerequisite_module_slot",
        "development_latch_enabled",
        "development_latch_present",
        "development_latch_state",
        "development_effective_gain_present",
        "development_effective_gain",
        "connection_occupied",
        "connection_local_id_index",
        "connection_src_module_slot",
        "connection_dst_module_slot",
        "connection_src_slot_index",
        "connection_dst_slot_index",
        "connection_value_type_index",
        "connection_channel_index",
        "connection_weight",
        "connection_execution_order",
        "structural_cost_counts",
        "structural_cost_components",
        "structural_cost_total",
    }};

Tensor gather_module_row(const Tensor& values, const Tensor& raw_slots) {
    auto slots = torch::clamp(raw_slots, 0, kModuleSlots - 1);
    if (values.dim() == 2) {
        return values.gather(1, slots.unsqueeze(1)).squeeze(1);
    }
    std::vector<int64_t> shape;
    shape.reserve(static_cast<std::size_t>(values.dim()));
    shape.push_back(slots.size(0));
    shape.push_back(1);
    for (int64_t axis = 2; axis < values.dim(); ++axis) {
        shape.push_back(values.size(axis));
    }
    auto gather_index = slots.reshape({slots.size(0), 1, 1});
    if (values.dim() > 3) {
        std::vector<int64_t> index_shape(
            static_cast<std::size_t>(values.dim()), 1
        );
        index_shape[0] = slots.size(0);
        gather_index = slots.reshape(index_shape);
    }
    return values.gather(1, gather_index.expand(shape)).squeeze(1);
}

Tensor one_hot(
    const Tensor& raw_indices,
    const int64_t classes,
    const c10::ScalarType dtype
) {
    auto indices = torch::clamp(raw_indices, 0, classes - 1).to(torch::kLong);
    auto out = torch::zeros(
        {indices.size(0), classes},
        indices.options().dtype(dtype)
    );
    return out.scatter_(1, indices.unsqueeze(1), 1);
}

Tensor module_one_hot(
    const Tensor& slots,
    const Tensor& valid,
    const c10::ScalarType dtype
) {
    return one_hot(slots, kModuleSlots, dtype)
        * valid.unsqueeze(1).to(dtype);
}

Tensor stable_sigmoid_f32(const Tensor& value) {
    auto clipped = torch::clamp(value, -16.0, 16.0);
    auto positive_exp = torch::exp(-clipped);
    auto positive_denominator = torch::add(
        torch::ones_like(positive_exp), positive_exp
    );
    auto positive = torch::div(
        torch::ones_like(positive_denominator), positive_denominator
    );
    auto negative_exp = torch::exp(clipped);
    auto negative_denominator = torch::add(
        torch::ones_like(negative_exp), negative_exp
    );
    auto negative = torch::div(negative_exp, negative_denominator);
    return torch::where(clipped >= 0.0, positive, negative);
}

std::pair<Tensor, Tensor> gated_step(
    const Tensor& external,
    const Tensor& routed_previous,
    const Tensor& previous_state,
    const Tensor& expression_scale,
    const Tensor& developmental_gain,
    const Tensor& parameters
) {
    auto aggregate_sum = torch::add(external, routed_previous);
    auto aggregate = torch::clamp(aggregate_sum, -1.0, 1.0);

    auto reset_input = torch::mul(parameters.select(1, 1), aggregate);
    auto reset_state = torch::mul(parameters.select(1, 2), previous_state);
    auto reset_sum_1 = torch::add(parameters.select(1, 0), reset_input);
    auto reset_pre = torch::clamp(torch::add(reset_sum_1, reset_state), -16.0, 16.0);
    auto reset = stable_sigmoid_f32(reset_pre);

    auto update_input = torch::mul(parameters.select(1, 4), aggregate);
    auto update_state = torch::mul(parameters.select(1, 5), previous_state);
    auto update_sum_1 = torch::add(parameters.select(1, 3), update_input);
    auto update_pre = torch::clamp(torch::add(update_sum_1, update_state), -16.0, 16.0);
    auto update = stable_sigmoid_f32(update_pre);

    auto reset_previous = torch::mul(reset, previous_state);
    auto candidate_input = torch::mul(parameters.select(1, 7), aggregate);
    auto candidate_state = torch::mul(parameters.select(1, 8), reset_previous);
    auto candidate_sum_1 = torch::add(parameters.select(1, 6), candidate_input);
    auto candidate_pre = torch::clamp(
        torch::add(candidate_sum_1, candidate_state), -16.0, 16.0
    );
    auto candidate = torch::tanh(candidate_pre);

    auto keep = torch::mul(torch::sub(torch::ones_like(update), update), previous_state);
    auto write = torch::mul(update, candidate);
    auto next_state = torch::clamp(torch::add(keep, write), -1.0, 1.0);
    auto base_output = torch::clamp(
        torch::mul(parameters.select(1, 9), next_state), -1.0, 1.0
    );
    auto expressed = torch::clamp(
        torch::mul(expression_scale, base_output), -1.0, 1.0
    );
    auto effective = torch::where(
        developmental_gain == 1.0,
        expressed,
        torch::clamp(torch::mul(developmental_gain, expressed), -1.0, 1.0)
    );
    return {next_state, effective};
}

void validate_plane_metadata(const std::vector<Tensor>& planes) {
    TORCH_CHECK(
        planes.size() == eiel::cuda_architecture_b::kModularTickPlaneCount,
        "native modular tick requires exactly 57 ordered planes"
    );
    const auto& occupied = planes.at(AgentOccupied);
    TORCH_CHECK(occupied.dim() == 1 && occupied.size(0) > 0, "agent_occupied must be rank one");
    const int64_t capacity = occupied.size(0);
    const auto device = occupied.device();
    for (int64_t index = 0; index < static_cast<int64_t>(planes.size()); ++index) {
        const auto& tensor = planes.at(static_cast<std::size_t>(index));
        TORCH_CHECK(
            tensor.device() == device,
            "native modular tick plane device mismatch: ",
            kPlaneNames.at(static_cast<std::size_t>(index))
        );
        TORCH_CHECK(
            tensor.is_contiguous(),
            "native modular tick plane must be contiguous: ",
            kPlaneNames.at(static_cast<std::size_t>(index))
        );
        TORCH_CHECK(
            !tensor.requires_grad(),
            "native modular tick plane must not require gradients: ",
            kPlaneNames.at(static_cast<std::size_t>(index))
        );
    }
    const auto expect = [&](const Plane index, const c10::ScalarType dtype, std::vector<int64_t> shape) {
        const auto& tensor = planes.at(index);
        TORCH_CHECK(
            tensor.scalar_type() == dtype && tensor.sizes().vec() == shape,
            "invalid native modular tick plane: ", kPlaneNames.at(index)
        );
    };
    expect(AgentOccupied, torch::kBool, {capacity});
    expect(AgentId, torch::kInt64, {capacity});
    expect(AgentSlotToken, torch::kInt64, {capacity});
    expect(ModuleCount, torch::kInt32, {capacity});
    expect(ConnectionCount, torch::kInt32, {capacity});
    expect(ModuleOccupied, torch::kBool, {capacity, kModuleSlots});
    expect(ModuleLocalIdIndex, torch::kInt32, {capacity, kModuleSlots});
    expect(ModuleInstanceIdIndex, torch::kInt32, {capacity, kModuleSlots});
    expect(ModuleHomologyIdIndex, torch::kInt32, {capacity, kModuleSlots});
    expect(ModuleTypeIndex, torch::kInt32, {capacity, kModuleSlots});
    expect(ModuleTypeCode, torch::kInt8, {capacity, kModuleSlots});
    expect(IdentityOriginCode, torch::kInt8, {capacity, kModuleSlots});
    expect(TopologyPosition, torch::kFloat32, {capacity, kModuleSlots, 2});
    expect(ModuleExecutionOrder, torch::kInt32, {capacity, kModuleSlots});
    expect(DependencyTopologicalOrder, torch::kInt32, {capacity, kModuleSlots});
    expect(OperatorCode, torch::kInt8, {capacity, kModuleSlots});
    expect(IntegratorParameters, torch::kFloat32, {capacity, kModuleSlots, 4});
    expect(GatedParameters, torch::kFloat32, {capacity, kModuleSlots, 10});
    expect(ModuleDormant, torch::kBool, {capacity, kModuleSlots});
    expect(ModuleStatePresent, torch::kBool, {capacity, kModuleSlots});
    expect(ModuleState, torch::kFloat32, {capacity, kModuleSlots, 1});
    expect(ModuleOutputPresent, torch::kBool, {capacity, kModuleSlots});
    expect(ModulePreviousOutput, torch::kFloat32, {capacity, kModuleSlots, 1});
    expect(InputBindingCount, torch::kInt8, {capacity, kModuleSlots});
    expect(InputSemanticCode, torch::kInt8, {capacity, kModuleSlots, 2});
    expect(InputSemanticIdIndex, torch::kInt32, {capacity, kModuleSlots, 2});
    expect(InputGain, torch::kFloat32, {capacity, kModuleSlots, 2});
    expect(InputEnabled, torch::kBool, {capacity, kModuleSlots, 2});
    expect(OutputBindingCount, torch::kInt8, {capacity, kModuleSlots});
    expect(OutputSemanticCode, torch::kInt8, {capacity, kModuleSlots, 2});
    expect(OutputSemanticIdIndex, torch::kInt32, {capacity, kModuleSlots, 2});
    expect(OutputActionIndex, torch::kInt8, {capacity, kModuleSlots, 2});
    expect(OutputGain, torch::kFloat32, {capacity, kModuleSlots, 2});
    expect(OutputEnabled, torch::kBool, {capacity, kModuleSlots, 2});
    expect(DevelopmentExplicit, torch::kBool, {capacity, kModuleSlots});
    expect(DevelopmentStageGains, torch::kFloat32, {capacity, kModuleSlots, 3});
    expect(DevelopmentPrerequisiteCount, torch::kInt8, {capacity, kModuleSlots});
    expect(
        DevelopmentPrerequisiteHomologyIndex,
        torch::kInt32,
        {capacity, kModuleSlots, 2}
    );
    expect(DevelopmentPrerequisiteModuleSlot, torch::kInt32, {capacity, kModuleSlots, 2});
    expect(DevelopmentLatchEnabled, torch::kBool, {capacity, kModuleSlots});
    expect(DevelopmentLatchPresent, torch::kBool, {capacity, kModuleSlots});
    expect(DevelopmentLatchState, torch::kBool, {capacity, kModuleSlots});
    expect(DevelopmentEffectiveGainPresent, torch::kBool, {capacity, kModuleSlots});
    expect(DevelopmentEffectiveGain, torch::kFloat32, {capacity, kModuleSlots});
    expect(ConnectionOccupied, torch::kBool, {capacity, kConnectionSlots});
    expect(ConnectionLocalIdIndex, torch::kInt32, {capacity, kConnectionSlots});
    expect(ConnectionSrcModuleSlot, torch::kInt32, {capacity, kConnectionSlots});
    expect(ConnectionDstModuleSlot, torch::kInt32, {capacity, kConnectionSlots});
    expect(ConnectionSrcSlotIndex, torch::kInt32, {capacity, kConnectionSlots});
    expect(ConnectionDstSlotIndex, torch::kInt32, {capacity, kConnectionSlots});
    expect(ConnectionValueTypeIndex, torch::kInt32, {capacity, kConnectionSlots});
    expect(ConnectionChannelIndex, torch::kInt32, {capacity, kConnectionSlots});
    expect(ConnectionWeight, torch::kFloat32, {capacity, kConnectionSlots});
    expect(ConnectionExecutionOrder, torch::kInt32, {capacity, kConnectionSlots});
    expect(StructuralCostCounts, torch::kInt64, {capacity, 10});
    expect(StructuralCostComponents, torch::kFloat64, {capacity, 10});
    expect(StructuralCostTotal, torch::kFloat64, {capacity});
}

void validate_tick_inputs(
    const std::vector<Tensor>& planes,
    const Tensor& observations,
    const Tensor& fatigue,
    const Tensor& stage_indices,
    const int64_t action_width
) {
    TORCH_CHECK(
        action_width == kActionWidth,
        "native modular tick action width must equal 7"
    );
    const int64_t capacity = planes.at(AgentOccupied).size(0);
    const auto device = planes.at(AgentOccupied).device();
    TORCH_CHECK(
        observations.device() == device
            && observations.scalar_type() == torch::kFloat32
            && observations.dim() == 2
            && observations.size(0) == capacity
            && observations.size(1) == kObservationWidth
            && observations.is_contiguous()
            && !observations.requires_grad(),
        "observations must be contiguous float32[capacity,27]"
    );
    TORCH_CHECK(
        fatigue.device() == device
            && fatigue.scalar_type() == torch::kFloat32
            && fatigue.dim() == 1
            && fatigue.size(0) == capacity
            && fatigue.is_contiguous()
            && !fatigue.requires_grad(),
        "fatigue must be contiguous float32[capacity]"
    );
    TORCH_CHECK(
        stage_indices.device() == device
            && stage_indices.scalar_type() == torch::kInt64
            && stage_indices.dim() == 1
            && stage_indices.size(0) == capacity
            && stage_indices.is_contiguous()
            && !stage_indices.requires_grad(),
        "stage_indices must be contiguous int64[capacity]"
    );
    at::_assert_async(
        torch::all(torch::isfinite(observations)),
        "observations must be finite"
    );
    at::_assert_async(
        torch::all(torch::isfinite(fatigue)),
        "fatigue must be finite"
    );
    at::_assert_async(
        torch::all(torch::logical_and(stage_indices >= 0, stage_indices < 3)),
        "stage_indices must be in [0, 2]"
    );
    at::_assert_async(
        torch::all(torch::logical_and(fatigue >= 0.0, fatigue <= 1.0)),
        "fatigue must be in [0, 1]"
    );
}

void validate_module_order(
    const Tensor& order_plane,
    const Tensor& generic,
    const char* membership_message,
    const char* completeness_message
) {
    const int64_t capacity = generic.size(0);
    auto seen = torch::zeros(
        {capacity, kModuleSlots},
        order_plane.options().dtype(torch::kInt32)
    );
    for (int64_t order = 0; order < kModuleSlots; ++order) {
        auto slots = order_plane.select(1, order).to(torch::kLong);
        auto valid = slots >= 0;
        auto in_range = torch::logical_and(slots >= 0, slots < kModuleSlots);
        at::_assert_async(
            torch::all(torch::logical_or(torch::logical_not(valid), in_range)),
            "module order slot is outside the fixed module envelope"
        );
        auto selected_generic = gather_module_row(generic, slots);
        at::_assert_async(
            torch::all(torch::logical_or(
                torch::logical_not(valid), selected_generic
            )),
            membership_message
        );
        auto step_valid = torch::logical_and(valid, selected_generic);
        seen = torch::add(
            seen,
            module_one_hot(slots, step_valid, torch::kBool).to(torch::kInt32)
        );
    }
    at::_assert_async(
        torch::all(seen == generic.to(torch::kInt32)),
        completeness_message
    );
}

void validate_plane_contents(const std::vector<Tensor>& planes) {
    const int64_t capacity = planes.at(AgentOccupied).size(0);
    auto module_occupied = planes.at(ModuleOccupied);
    at::_assert_async(
        torch::all(torch::logical_or(
            torch::logical_not(module_occupied),
            planes.at(AgentOccupied).unsqueeze(1)
        )),
        "module occupancy requires an occupied agent slot"
    );
    auto generic = torch::logical_and(
        module_occupied, planes.at(ModuleTypeCode) == kGenericModule
    );
    auto operator_code = planes.at(OperatorCode);
    at::_assert_async(
        torch::all(torch::logical_or(
            torch::logical_not(generic),
            torch::logical_or(
                operator_code == kIntegratorOperator,
                operator_code == kGatedOperator
            )
        )),
        "generic modules require a supported operator code"
    );

    auto prerequisite_count = planes.at(DevelopmentPrerequisiteCount);
    at::_assert_async(
        torch::all(torch::logical_and(
            prerequisite_count >= 0, prerequisite_count <= 2
        )),
        "development prerequisite count is outside the fixed envelope"
    );
    auto prerequisite_positions = torch::arange(
        2,
        prerequisite_count.options().dtype(torch::kInt64)
    ).reshape({1, 1, 2});
    auto prerequisite_active = prerequisite_positions
        < prerequisite_count.unsqueeze(2).to(torch::kLong);
    auto prerequisite_slots = planes.at(DevelopmentPrerequisiteModuleSlot);
    at::_assert_async(
        torch::all(torch::logical_or(
            torch::logical_not(prerequisite_active),
            torch::logical_and(
                prerequisite_slots >= 0,
                prerequisite_slots < kModuleSlots
            )
        )),
        "active development prerequisite is outside the fixed module envelope"
    );
    validate_module_order(
        planes.at(DependencyTopologicalOrder),
        generic,
        "development order may contain generic modules only",
        "development order must contain every generic module exactly once"
    );
    validate_module_order(
        planes.at(ModuleExecutionOrder),
        generic,
        "module execution order may contain generic modules only",
        "module execution order must contain every generic module exactly once"
    );

    auto binding_positions = torch::arange(
        2,
        planes.at(InputBindingCount).options().dtype(torch::kInt64)
    ).reshape({1, 1, 2});
    auto input_count = planes.at(InputBindingCount).to(torch::kLong);
    auto input_mask = torch::logical_and(
        binding_positions < input_count.unsqueeze(2),
        planes.at(InputEnabled)
    );
    auto input_codes = planes.at(InputSemanticCode).to(torch::kLong);
    at::_assert_async(
        torch::all(torch::logical_and(input_count >= 0, input_count <= 2)),
        "input binding count is outside the fixed envelope"
    );
    at::_assert_async(
        torch::all(torch::logical_or(
            torch::logical_not(input_mask),
            torch::logical_and(input_codes >= 0, input_codes < 2)
        )),
        "active input binding semantic code is invalid"
    );

    auto output_count = planes.at(OutputBindingCount).to(torch::kLong);
    auto output_mask = torch::logical_and(
        binding_positions < output_count.unsqueeze(2),
        planes.at(OutputEnabled)
    );
    auto output_actions = planes.at(OutputActionIndex).to(torch::kLong);
    at::_assert_async(
        torch::all(torch::logical_and(output_count >= 0, output_count <= 2)),
        "output binding count is outside the fixed envelope"
    );
    at::_assert_async(
        torch::all(torch::logical_or(
            torch::logical_not(output_mask),
            torch::logical_and(output_actions >= 0, output_actions < kActionWidth)
        )),
        "active output binding action index is invalid"
    );

    auto connection_occupied = planes.at(ConnectionOccupied);
    auto connection_order = planes.at(ConnectionExecutionOrder).to(torch::kLong);
    auto connection_seen = torch::zeros(
        {capacity, kConnectionSlots},
        connection_order.options().dtype(torch::kInt32)
    );
    for (int64_t order = 0; order < kConnectionSlots; ++order) {
        auto selected = torch::logical_and(
            connection_occupied, connection_order == order
        );
        connection_seen = torch::add(
            connection_seen, selected.to(torch::kInt32)
        );
        at::_assert_async(
            torch::all(torch::sum(selected.to(torch::kLong), 1) <= 1),
            "connection execution order must be unique per agent"
        );
    }
    at::_assert_async(
        torch::all(connection_seen == connection_occupied.to(torch::kInt32)),
        "connection execution order must contain every occupied connection exactly once"
    );
    auto connection_endpoints_valid = torch::logical_and(
        torch::logical_and(
            planes.at(ConnectionSrcModuleSlot) >= 0,
            planes.at(ConnectionSrcModuleSlot) < kModuleSlots
        ),
        torch::logical_and(
            planes.at(ConnectionDstModuleSlot) >= 0,
            planes.at(ConnectionDstModuleSlot) < kModuleSlots
        )
    );
    at::_assert_async(
        torch::all(torch::logical_or(
            torch::logical_not(connection_occupied),
            connection_endpoints_valid
        )),
        "connection endpoint is outside the fixed module envelope"
    );
}

}  // namespace

namespace eiel {
namespace cuda_architecture_b {

void validate_modular_planes_v1(const std::vector<Tensor>& planes) {
    validate_plane_metadata(planes);
    validate_plane_contents(planes);
}

Tensor scatter_active_rows_by_slot_v1(
    Tensor rows,
    Tensor slots,
    Tensor active_mask,
    const int64_t capacity
) {
    TORCH_CHECK(capacity > 0, "slot scatter capacity must be positive");
    TORCH_CHECK(
        rows.dim() == 2
            && rows.size(0) == capacity
            && rows.is_contiguous()
            && !rows.requires_grad(),
        "slot scatter rows must be contiguous [capacity,width]"
    );
    TORCH_CHECK(
        slots.device() == rows.device()
            && slots.scalar_type() == torch::kInt64
            && slots.dim() == 1
            && slots.size(0) == capacity
            && slots.is_contiguous()
            && !slots.requires_grad(),
        "slot scatter slots must be contiguous int64[capacity]"
    );
    TORCH_CHECK(
        active_mask.device() == rows.device()
            && active_mask.scalar_type() == torch::kBool
            && active_mask.dim() == 1
            && active_mask.size(0) == capacity
            && active_mask.is_contiguous()
            && !active_mask.requires_grad(),
        "slot scatter active_mask must be contiguous bool[capacity]"
    );

    auto active_slots = torch::masked_select(slots, active_mask);
    auto bounded_slots = torch::clamp(active_slots, 0, capacity - 1);
    at::_assert_async(
        torch::all(active_slots == bounded_slots),
        "active slot is outside the fixed agent capacity"
    );
    auto slot_counts = torch::zeros(
        {capacity}, slots.options().dtype(torch::kInt32)
    );
    slot_counts.index_add_(
        0,
        bounded_slots,
        torch::ones(
            bounded_slots.sizes(), slots.options().dtype(torch::kInt32)
        )
    );
    at::_assert_async(
        torch::all(slot_counts <= 1),
        "active slots must be unique"
    );

    auto active_rows = torch::masked_select(
        rows,
        active_mask.unsqueeze(1)
            .expand({capacity, rows.size(1)})
    ).reshape({-1, rows.size(1)});
    auto slot_rows = torch::zeros_like(rows);
    slot_rows.index_copy_(0, bounded_slots, active_rows);
    return slot_rows.contiguous();
}

void clear_modular_rows_v1(
    const std::vector<Tensor>& planes,
    Tensor dead_slots,
    Tensor dead_count
) {
    validate_plane_metadata(planes);
    const int64_t capacity = planes.at(AgentOccupied).size(0);
    const auto device = planes.at(AgentOccupied).device();
    TORCH_CHECK(
        dead_slots.device() == device
            && dead_slots.scalar_type() == torch::kInt64
            && dead_slots.dim() == 1
            && dead_slots.size(0) == capacity
            && dead_slots.is_contiguous()
            && !dead_slots.requires_grad(),
        "dead_slots must be contiguous int64[capacity]"
    );
    TORCH_CHECK(
        dead_count.device() == device
            && dead_count.scalar_type() == torch::kInt64
            && dead_count.dim() == 1
            && dead_count.numel() == 1
            && dead_count.is_contiguous()
            && !dead_count.requires_grad(),
        "dead_count must be contiguous int64[1]"
    );
    at::_assert_async(
        torch::all(torch::logical_and(dead_count >= 0, dead_count <= capacity)),
        "dead_count is outside the fixed agent capacity"
    );
    auto positions = torch::arange(
        capacity, dead_slots.options().dtype(torch::kInt64)
    );
    auto valid = positions < torch::clamp(dead_count, 0, capacity);
    auto selected_slots = torch::masked_select(dead_slots, valid);
    auto bounded_slots = torch::clamp(selected_slots, 0, capacity - 1);
    at::_assert_async(
        torch::all(selected_slots == bounded_slots),
        "dead slot is outside the fixed agent capacity"
    );
    auto slot_counts = torch::zeros(
        {capacity}, dead_slots.options().dtype(torch::kInt32)
    );
    slot_counts.index_add_(
        0,
        bounded_slots,
        torch::ones(
            bounded_slots.sizes(), dead_slots.options().dtype(torch::kInt32)
        )
    );
    at::_assert_async(
        torch::all(slot_counts <= 1),
        "dead slots must be unique"
    );

    constexpr std::array<int64_t, kModularTickPlaneCount> kFillValues = {{
        0, -1, -1, 0, 0, 0, -1, -1, -1, -1,
        0, 0, 0, -1, -1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, -1, -1, 0, 0, 0, -1,
        -1, -1, 0, 0, 0, 0, 0, -1, -1, 0,
        0, 0, 0, 0, 0, -1, -1, -1, -1, -1,
        -1, -1, 0, -1, 0, 0, 0,
    }};
    for (int64_t index = 0; index < kModularTickPlaneCount; ++index) {
        planes.at(static_cast<std::size_t>(index)).index_fill_(
            0,
            bounded_slots,
            kFillValues.at(static_cast<std::size_t>(index))
        );
    }
}

std::vector<Tensor> modular_tick_v1(
    const std::vector<Tensor>& planes,
    Tensor observations,
    Tensor fatigue,
    Tensor stage_indices,
    const int64_t action_width
) {
    validate_modular_planes_v1(planes);
    validate_tick_inputs(
        planes, observations, fatigue, stage_indices, action_width
    );
    const int64_t capacity = planes.at(AgentOccupied).size(0);
    const auto float_options = observations.options().dtype(torch::kFloat32);
    const auto int64_options = observations.options().dtype(torch::kInt64);
    const auto int32_options = observations.options().dtype(torch::kInt32);

    auto module_occupied = planes.at(ModuleOccupied);
    at::_assert_async(
        torch::all(torch::logical_or(
            torch::logical_not(module_occupied),
            planes.at(AgentOccupied).unsqueeze(1)
        )),
        "module occupancy requires an occupied agent slot"
    );
    auto generic = torch::logical_and(
        module_occupied, planes.at(ModuleTypeCode) == kGenericModule
    );
    auto operator_code = planes.at(OperatorCode);
    at::_assert_async(
        torch::all(torch::logical_or(
            torch::logical_not(generic),
            torch::logical_or(
                operator_code == kIntegratorOperator,
                operator_code == kGatedOperator
            )
        )),
        "generic modules require a supported operator code"
    );

    auto old_latch_present = planes.at(DevelopmentLatchPresent).clone();
    auto old_latch_state = planes.at(DevelopmentLatchState).clone();
    auto effective_gain = torch::zeros({capacity, kModuleSlots}, float_options);
    auto dependency_ready = torch::zeros_like(generic);
    auto next_latch_present = old_latch_present.clone();
    auto next_latch_state = old_latch_state.clone();
    auto development_seen = torch::zeros({capacity, kModuleSlots}, int32_options);
    auto dependency_order = planes.at(DependencyTopologicalOrder);
    auto stage_gains = planes.at(DevelopmentStageGains);
    auto prerequisite_slots = planes.at(DevelopmentPrerequisiteModuleSlot);
    auto prerequisite_count = planes.at(DevelopmentPrerequisiteCount);
    auto latch_enabled = planes.at(DevelopmentLatchEnabled);
    auto binding_positions = torch::arange(2, int64_options).unsqueeze(0);
    at::_assert_async(
        torch::all(torch::logical_and(
            prerequisite_count >= 0, prerequisite_count <= 2
        )),
        "development prerequisite count is outside the fixed envelope"
    );

    for (int64_t order = 0; order < kModuleSlots; ++order) {
        auto slots = dependency_order.select(1, order).to(torch::kLong);
        auto valid = slots >= 0;
        auto selected_generic = gather_module_row(generic, slots);
        auto step_valid = torch::logical_and(valid, selected_generic);
        at::_assert_async(
            torch::all(torch::logical_or(torch::logical_not(valid), selected_generic)),
            "development order may contain generic modules only"
        );
        auto one_hot_bool = module_one_hot(slots, step_valid, torch::kBool);
        development_seen = torch::add(
            development_seen, one_hot_bool.to(torch::kInt32)
        );

        auto selected_gains = gather_module_row(stage_gains, slots);
        auto requested = selected_gains.gather(1, stage_indices.unsqueeze(1)).squeeze(1);
        auto selected_prerequisite_slots = gather_module_row(
            prerequisite_slots, slots
        ).to(torch::kLong);
        auto selected_prerequisite_count = gather_module_row(
            prerequisite_count, slots
        ).to(torch::kLong);
        auto prerequisite_mask = binding_positions
            < selected_prerequisite_count.unsqueeze(1);
        auto prerequisite_gain = effective_gain.gather(
            1, torch::clamp(selected_prerequisite_slots, 0, kModuleSlots - 1)
        );
        auto raw_ready = torch::all(
            torch::logical_or(torch::logical_not(prerequisite_mask), prerequisite_gain > 0.0),
            1
        );
        auto selected_latch_enabled = gather_module_row(latch_enabled, slots);
        auto previously_latched = torch::logical_and(
            gather_module_row(old_latch_present, slots),
            gather_module_row(old_latch_state, slots)
        );
        auto gate_ready = torch::logical_or(raw_ready, previously_latched);
        auto next_latched = torch::logical_and(
            selected_latch_enabled,
            torch::logical_or(previously_latched, raw_ready)
        );
        auto final_gain = torch::where(
            gate_ready, requested, torch::zeros_like(requested)
        );
        effective_gain = torch::where(
            one_hot_bool, final_gain.unsqueeze(1), effective_gain
        );
        dependency_ready = torch::where(
            one_hot_bool, gate_ready.unsqueeze(1), dependency_ready
        );
        next_latch_present = torch::where(
            one_hot_bool, next_latched.unsqueeze(1), next_latch_present
        );
        next_latch_state = torch::where(
            one_hot_bool, next_latched.unsqueeze(1), next_latch_state
        );
    }
    at::_assert_async(
        torch::all(development_seen == generic.to(torch::kInt32)),
        "development order must contain every generic module exactly once"
    );

    planes.at(DevelopmentLatchPresent).copy_(
        torch::where(generic, next_latch_present, old_latch_present)
    );
    planes.at(DevelopmentLatchState).copy_(
        torch::where(generic, next_latch_state, old_latch_state)
    );
    planes.at(DevelopmentEffectiveGainPresent).copy_(
        torch::where(
            generic,
            torch::ones_like(planes.at(DevelopmentEffectiveGainPresent)),
            planes.at(DevelopmentEffectiveGainPresent)
        )
    );
    planes.at(DevelopmentEffectiveGain).copy_(
        torch::where(generic, effective_gain, planes.at(DevelopmentEffectiveGain))
    );

    auto semantic_sources = torch::stack(
        {observations.select(1, kLocalObservationWidth),
         observations.select(1, kLocalObservationWidth + 1)},
        1
    );
    auto input_count = planes.at(InputBindingCount).to(torch::kLong);
    auto input_codes = planes.at(InputSemanticCode).to(torch::kLong);
    auto input_mask = torch::logical_and(
        binding_positions.reshape({1, 1, 2}) < input_count.unsqueeze(2),
        planes.at(InputEnabled)
    );
    at::_assert_async(
        torch::all(torch::logical_and(input_count >= 0, input_count <= 2)),
        "input binding count is outside the fixed envelope"
    );
    at::_assert_async(
        torch::all(torch::logical_or(
            torch::logical_not(input_mask),
            torch::logical_and(input_codes >= 0, input_codes < 2)
        )),
        "active input binding semantic code is invalid"
    );
    auto expanded_sources = semantic_sources.unsqueeze(1).expand(
        {capacity, kModuleSlots, 2}
    );
    auto routed_source = expanded_sources.gather(
        2, torch::clamp(input_codes, 0, 1)
    );
    auto routed_contribution = torch::mul(planes.at(InputGain), routed_source);
    routed_contribution = torch::where(
        input_mask, routed_contribution, torch::zeros_like(routed_contribution)
    );
    auto routed_observation = torch::zeros({capacity, kModuleSlots}, float_options);
    for (int64_t binding = 0; binding < 2; ++binding) {
        routed_observation = torch::add(
            routed_observation, routed_contribution.select(2, binding)
        );
    }
    routed_observation = torch::clamp(routed_observation, -1.0, 1.0);

    auto previous_output = planes.at(ModulePreviousOutput).select(2, 0).clone();
    auto stored_state = planes.at(ModuleState).select(2, 0).clone();
    auto old_state = torch::where(
        planes.at(ModuleStatePresent), stored_state, torch::zeros_like(stored_state)
    );
    auto dormant = planes.at(ModuleDormant);
    auto source_enabled = torch::logical_and(
        generic,
        torch::logical_and(torch::logical_not(dormant), effective_gain > 0.0)
    );
    auto connection_source_available = torch::logical_and(
        generic, torch::logical_not(dormant)
    );
    auto previous_for_connections = torch::where(
        effective_gain > 0.0, previous_output, torch::zeros_like(previous_output)
    );
    auto destination_enabled = source_enabled;
    auto routed_previous = torch::zeros_like(old_state);
    auto evaluated_connections = torch::zeros({capacity}, int64_options);
    auto connection_occupied = planes.at(ConnectionOccupied);
    auto connection_order = planes.at(ConnectionExecutionOrder);
    auto source_slots = planes.at(ConnectionSrcModuleSlot).to(torch::kLong);
    auto destination_slots = planes.at(ConnectionDstModuleSlot).to(torch::kLong);
    auto weights = planes.at(ConnectionWeight);
    auto connection_seen = torch::zeros(
        {capacity, kConnectionSlots}, int32_options
    );

    for (int64_t order = 0; order < kConnectionSlots; ++order) {
        auto selected = torch::logical_and(connection_occupied, connection_order == order);
        connection_seen = torch::add(connection_seen, selected.to(torch::kInt32));
        auto selected_count = torch::sum(selected.to(torch::kLong), 1);
        at::_assert_async(
            torch::all(selected_count <= 1),
            "connection execution order must be unique per agent"
        );
        auto present = selected_count == 1;
        auto source_slot = torch::sum(
            torch::where(selected, source_slots, torch::zeros_like(source_slots)), 1
        );
        auto destination_slot = torch::sum(
            torch::where(selected, destination_slots, torch::zeros_like(destination_slots)), 1
        );
        auto weight = torch::sum(
            torch::where(selected, weights, torch::zeros_like(weights)), 1
        );
        auto endpoint_range = torch::logical_and(
            torch::logical_and(source_slot >= 0, source_slot < kModuleSlots),
            torch::logical_and(
                destination_slot >= 0, destination_slot < kModuleSlots
            )
        );
        at::_assert_async(
            torch::all(torch::logical_or(
                torch::logical_not(present), endpoint_range
            )),
            "connection endpoint is outside the fixed module envelope"
        );
        auto source_value = gather_module_row(previous_for_connections, source_slot);
        auto source_active = gather_module_row(connection_source_available, source_slot);
        auto destination_active = gather_module_row(destination_enabled, destination_slot);
        auto evaluated = torch::logical_and(
            present, torch::logical_and(source_active, destination_active)
        );
        auto contribution = torch::where(
            evaluated, torch::mul(weight, source_value), torch::zeros_like(weight)
        );
        routed_previous = torch::add(
            routed_previous,
            module_one_hot(destination_slot, evaluated, torch::kFloat32)
                * contribution.unsqueeze(1)
        );
        evaluated_connections = torch::add(
            evaluated_connections, evaluated.to(torch::kLong)
        );
    }
    at::_assert_async(
        torch::all(connection_seen == connection_occupied.to(torch::kInt32)),
        "connection execution order must contain every occupied connection exactly once"
    );

    auto execution_order = planes.at(ModuleExecutionOrder);
    auto execution_seen = torch::zeros_like(development_seen);
    auto next_state = stored_state.clone();
    auto next_output = previous_output.clone();
    auto state_before_audit = old_state.clone();
    auto executing_integrators = torch::zeros({capacity}, int64_options);
    auto executing_gated = torch::zeros({capacity}, int64_options);
    auto executing_modules = torch::zeros({capacity}, int64_options);
    auto changing_state = torch::zeros({capacity}, int64_options);

    for (int64_t order = 0; order < kModuleSlots; ++order) {
        auto slots = execution_order.select(1, order).to(torch::kLong);
        auto valid = slots >= 0;
        auto selected_generic = gather_module_row(generic, slots);
        auto step_valid = torch::logical_and(valid, selected_generic);
        at::_assert_async(
            torch::all(torch::logical_or(torch::logical_not(valid), selected_generic)),
            "module execution order may contain generic modules only"
        );
        auto one_hot_bool = module_one_hot(slots, step_valid, torch::kBool);
        execution_seen = torch::add(execution_seen, one_hot_bool.to(torch::kInt32));
        auto selected_old_state = gather_module_row(old_state, slots);
        auto selected_routed_observation = gather_module_row(routed_observation, slots);
        auto selected_routed_previous = gather_module_row(routed_previous, slots);
        auto selected_gain = gather_module_row(effective_gain, slots);
        auto selected_dormant = gather_module_row(dormant, slots);
        auto selected_operator = gather_module_row(operator_code, slots);
        auto selected_integrator = gather_module_row(
            planes.at(IntegratorParameters), slots
        );
        auto selected_gated = gather_module_row(planes.at(GatedParameters), slots);
        auto combined_external = torch::where(
            selected_routed_observation == 0.0,
            fatigue,
            torch::add(fatigue, selected_routed_observation)
        );
        auto executing = torch::logical_and(
            step_valid,
            torch::logical_and(torch::logical_not(selected_dormant), selected_gain > 0.0)
        );
        auto integrator_mask = torch::logical_and(
            executing, selected_operator == kIntegratorOperator
        );
        auto gated_mask = torch::logical_and(
            executing, selected_operator == kGatedOperator
        );

        auto integrator_state = torch::mul(
            selected_integrator.select(1, 1), selected_old_state
        );
        integrator_state = torch::add(
            integrator_state,
            torch::mul(selected_integrator.select(1, 2), combined_external)
        );
        integrator_state = torch::clamp(
            torch::add(integrator_state, selected_routed_previous), -1.0, 1.0
        );
        auto integrator_output = torch::mul(
            selected_integrator.select(1, 0), selected_integrator.select(1, 3)
        );
        integrator_output = torch::clamp(
            torch::mul(integrator_output, integrator_state), -1.0, 1.0
        );
        integrator_output = torch::where(
            selected_gain == 1.0,
            integrator_output,
            torch::clamp(torch::mul(selected_gain, integrator_output), -1.0, 1.0)
        );
        auto gated = gated_step(
            combined_external,
            selected_routed_previous,
            selected_old_state,
            selected_integrator.select(1, 0),
            selected_gain,
            selected_gated
        );
        auto selected_next_state = torch::where(
            integrator_mask,
            integrator_state,
            torch::where(gated_mask, gated.first, selected_old_state)
        );
        auto selected_next_output = torch::where(
            integrator_mask,
            integrator_output,
            torch::where(gated_mask, gated.second, torch::zeros_like(gated.second))
        );
        next_state = torch::where(
            one_hot_bool, selected_next_state.unsqueeze(1), next_state
        );
        next_output = torch::where(
            one_hot_bool, selected_next_output.unsqueeze(1), next_output
        );
        executing_integrators = torch::add(
            executing_integrators, integrator_mask.to(torch::kLong)
        );
        executing_gated = torch::add(
            executing_gated, gated_mask.to(torch::kLong)
        );
        executing_modules = torch::add(
            executing_modules, executing.to(torch::kLong)
        );
        changing_state = torch::add(
            changing_state,
            torch::logical_and(step_valid, selected_next_state != selected_old_state)
                .to(torch::kLong)
        );
    }
    at::_assert_async(
        torch::all(execution_seen == generic.to(torch::kInt32)),
        "module execution order must contain every generic module exactly once"
    );

    planes.at(ModuleStatePresent).copy_(
        torch::where(
            generic,
            torch::ones_like(planes.at(ModuleStatePresent)),
            planes.at(ModuleStatePresent)
        )
    );
    planes.at(ModuleOutputPresent).copy_(
        torch::where(
            generic,
            torch::ones_like(planes.at(ModuleOutputPresent)),
            planes.at(ModuleOutputPresent)
        )
    );
    planes.at(ModuleState).select(2, 0).copy_(next_state);
    planes.at(ModulePreviousOutput).select(2, 0).copy_(next_output);

    auto legacy_rest = torch::zeros({capacity}, float_options);
    auto routed_action = torch::zeros({capacity, kActionWidth}, float_options);
    auto output_count = planes.at(OutputBindingCount).to(torch::kLong);
    auto output_actions = planes.at(OutputActionIndex).to(torch::kLong);
    auto output_enabled = planes.at(OutputEnabled);
    auto output_gain = planes.at(OutputGain);
    auto executable_output_bindings = torch::zeros({capacity}, int64_options);
    at::_assert_async(
        torch::all(torch::logical_and(output_count >= 0, output_count <= 2)),
        "output binding count is outside the fixed envelope"
    );

    for (int64_t order = 0; order < kModuleSlots; ++order) {
        auto slots = execution_order.select(1, order).to(torch::kLong);
        auto valid = slots >= 0;
        auto selected_generic = gather_module_row(generic, slots);
        auto step_valid = torch::logical_and(valid, selected_generic);
        auto selected_output = gather_module_row(next_output, slots);
        legacy_rest = torch::add(
            legacy_rest,
            torch::where(step_valid, selected_output, torch::zeros_like(selected_output))
        );
        auto selected_count = gather_module_row(output_count, slots);
        auto selected_actions = gather_module_row(output_actions, slots);
        auto selected_enabled = gather_module_row(output_enabled, slots);
        auto selected_output_gain = gather_module_row(output_gain, slots);
        auto selected_executing = torch::logical_and(
            step_valid,
            torch::logical_and(
                torch::logical_not(gather_module_row(dormant, slots)),
                gather_module_row(effective_gain, slots) > 0.0
            )
        );
        for (int64_t binding = 0; binding < 2; ++binding) {
            auto active_binding = torch::logical_and(
                step_valid,
                torch::logical_and(
                    selected_count > binding,
                    selected_enabled.select(1, binding)
                )
            );
            auto action_index = selected_actions.select(1, binding);
            at::_assert_async(
                torch::all(torch::logical_or(
                    torch::logical_not(active_binding),
                    torch::logical_and(action_index >= 0, action_index < kActionWidth)
                )),
                "active output binding action index is invalid"
            );
            auto contribution = torch::where(
                active_binding,
                torch::mul(selected_output_gain.select(1, binding), selected_output),
                torch::zeros_like(selected_output)
            );
            auto action_one_hot = one_hot(
                action_index, kActionWidth, torch::kFloat32
            );
            routed_action = torch::add(
                routed_action, action_one_hot * contribution.unsqueeze(1)
            );
            executable_output_bindings = torch::add(
                executable_output_bindings,
                torch::logical_and(active_binding, selected_executing).to(torch::kLong)
            );
        }
    }

    legacy_rest = torch::clamp(legacy_rest, -0.25, 0.25);
    auto action_residuals = torch::clamp(routed_action, -0.25, 0.25);
    auto routed_rest = routed_action.select(1, kRestAction);
    auto combined_rest = torch::where(
        routed_rest == 0.0,
        legacy_rest,
        torch::clamp(torch::add(legacy_rest, routed_rest), -0.25, 0.25)
    );
    auto rest_indices = torch::full({capacity}, kRestAction, int64_options);
    auto rest_selector = one_hot(rest_indices, kActionWidth, torch::kBool);
    action_residuals = torch::where(
        rest_selector, combined_rest.unsqueeze(1), action_residuals
    );

    auto allocated_modules = torch::sum(generic.to(torch::kLong), 1);
    auto occupied_agents = planes.at(AgentOccupied);
    auto structured_agents = torch::logical_and(occupied_agents, allocated_modules > 0);
    auto neutral_agents = torch::logical_and(occupied_agents, allocated_modules == 0);
    auto active_input_bindings = torch::sum(
        torch::logical_and(input_mask, source_enabled.unsqueeze(2)).to(torch::kLong),
        {1, 2}
    );
    auto dependency_edges = torch::sum(
        torch::where(
            source_enabled,
            prerequisite_count.to(torch::kLong),
            torch::zeros_like(prerequisite_count, torch::kLong)
        ),
        1
    );

    return {
        action_residuals,
        state_before_audit,
        next_state.clone(),
        next_output.clone(),
        effective_gain.clone(),
        dependency_ready.clone(),
        next_latch_state.clone(),
        routed_observation,
        routed_previous,
        legacy_rest,
        routed_action,
        action_residuals,
        allocated_modules,
        executing_modules,
        executing_integrators,
        executing_gated,
        torch::sum(torch::logical_and(generic, dormant).to(torch::kLong), 1),
        evaluated_connections,
        active_input_bindings,
        executable_output_bindings,
        dependency_edges,
        changing_state,
        torch::sum(structured_agents.to(torch::kLong)),
        torch::sum(neutral_agents.to(torch::kLong)),
        planes.at(StructuralCostCounts).clone(),
        planes.at(StructuralCostComponents).clone(),
        planes.at(StructuralCostTotal).clone(),
        torch::zeros({}, int64_options),
    };
}

}  // namespace cuda_architecture_b
}  // namespace eiel
