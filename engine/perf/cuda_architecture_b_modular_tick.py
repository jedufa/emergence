from __future__ import annotations

"""Device-capable passive tick executor for the Phase E modular overlay.

The executor owns no RNG and performs no policy choice, world update,
checkpoint, lifecycle, or campaign work.  CPU execution is the deterministic
oracle surface; the same tensor operations are CUDA-capable.
"""

from dataclasses import dataclass
from typing import Any, Final, Mapping, MutableMapping

import torch
import torch.nn.functional as functional

from env.gridworld_pkg.constants import ACT_REST, N_ACTIONS
from perf.cuda_architecture_b_modular_state_abi import (
    MODULAR_STORAGE_PROJECTION_PREFIX,
    build_modular_storage_layout,
)


MODULAR_TICK_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_modular_tick_v1"
)
MODULAR_TICK_LANE: Final[str] = "torch_device_resident_modular_tick_v1"
OBSERVATION_WIDTH: Final[int] = 27
LOCAL_OBSERVATION_WIDTH: Final[int] = 25
ACTION_WIDTH: Final[int] = 7
PHYSICAL_MODULE_SLOTS: Final[int] = 16
PHYSICAL_CONNECTION_SLOTS: Final[int] = 48
MAX_ROUTED_INPUT: Final[float] = 1.0
MAX_ACTION_RESIDUAL: Final[float] = 0.25
_GENERIC_MODULE_TYPE_CODE: Final[int] = 1
_INTEGRATOR_OPERATOR_CODE: Final[int] = 1
_GATED_OPERATOR_CODE: Final[int] = 2


class ModularTickError(RuntimeError):
    """The modular tick input or resident plane contract is invalid."""


@dataclass(frozen=True)
class ModularTickReceipt:
    schema_version: str
    lane: str
    field_namespace: str
    plane_count: int
    action_width: int
    rng_draws: int = 0
    device_to_host_synchronizations: int = 0
    hidden_fallback_count: int = 0
    simulation_executed: bool = False
    checkpoint_closure_claimed: bool = False
    lifecycle_claimed: bool = False
    campaign_claimed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class ModularTickResult:
    planes: MutableMapping[str, torch.Tensor]
    action_residuals: torch.Tensor
    audit_tensors: dict[str, torch.Tensor]
    receipt: ModularTickReceipt


def _assert_tensor(condition: torch.Tensor, message: str) -> None:
    try:
        torch._assert_async(condition, message)
    except RuntimeError as exc:
        raise ModularTickError(str(exc)) from exc


def _resolve_planes(
    planes: MutableMapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], str]:
    if not isinstance(planes, MutableMapping):
        raise ModularTickError("modular planes must be a mutable mapping")
    logical_names = tuple(
        field.name for field in build_modular_storage_layout(1).fields
    )
    logical_set = set(logical_names)
    prefixed_set = {
        f"{MODULAR_STORAGE_PROJECTION_PREFIX}{name}" for name in logical_names
    }
    keys = {str(name) for name in planes}
    logical_present = keys & logical_set
    prefixed_present = keys & prefixed_set
    other_modular = {
        name
        for name in keys
        if name.startswith("phase_e_modular_") and name not in prefixed_set
    }
    if other_modular or (logical_present and prefixed_present):
        raise ModularTickError("mixed modular plane namespaces are forbidden")
    if logical_present:
        if logical_present != logical_set or keys != logical_set:
            raise ModularTickError("unprefixed modular plane set must be exact and complete")
        return {name: planes[name] for name in logical_names}, "unprefixed"
    if prefixed_present:
        if prefixed_present != prefixed_set:
            raise ModularTickError("prefixed modular plane set is incomplete")
        return {
            name: planes[f"{MODULAR_STORAGE_PROJECTION_PREFIX}{name}"]
            for name in logical_names
        }, MODULAR_STORAGE_PROJECTION_PREFIX
    raise ModularTickError("complete modular plane set is required")


def _validate_inputs(
    resolved: Mapping[str, torch.Tensor],
    observations: torch.Tensor,
    fatigue: torch.Tensor,
    stage_indices: torch.Tensor,
    *,
    action_width: int,
) -> tuple[int, torch.device]:
    occupied = resolved["agent_occupied"]
    if not isinstance(occupied, torch.Tensor) or occupied.ndim != 1:
        raise ModularTickError("agent_occupied must be a rank-one tensor")
    capacity = int(occupied.shape[0])
    if capacity <= 0:
        raise ModularTickError("agent capacity must be positive")
    layout = build_modular_storage_layout(capacity)
    device = occupied.device
    for field in layout.fields:
        tensor = resolved[field.name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device != device
            or tensor.dtype != field.torch_dtype
            or tuple(tensor.shape) != field.shape
            or not tensor.is_contiguous()
            or tensor.requires_grad
        ):
            raise ModularTickError(f"invalid modular tick plane: {field.name}")
    if type(action_width) is not int or action_width != ACTION_WIDTH or action_width != int(N_ACTIONS):
        raise ModularTickError("modular tick action width must be exactly 7")
    expected_inputs = (
        ("observations", observations, torch.float32, (capacity, OBSERVATION_WIDTH)),
        ("fatigue", fatigue, torch.float32, (capacity,)),
        ("stage_indices", stage_indices, torch.int64, (capacity,)),
    )
    for name, tensor, dtype, shape in expected_inputs:
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device != device
            or tensor.dtype != dtype
            or tuple(tensor.shape) != shape
            or not tensor.is_contiguous()
            or tensor.requires_grad
        ):
            raise ModularTickError(f"invalid modular tick input: {name}")
    _assert_tensor(torch.all(torch.isfinite(observations)), "observations must be finite")
    _assert_tensor(torch.all(torch.isfinite(fatigue)), "fatigue must be finite")
    _assert_tensor(
        torch.all(torch.logical_and(stage_indices >= 0, stage_indices < 3)),
        "stage_indices must be in [0, 2]",
    )
    _assert_tensor(
        torch.all(torch.logical_and(fatigue >= 0.0, fatigue <= 1.0)),
        "fatigue must be in [0, 1]",
    )
    return capacity, device


def _gather_module_row(values: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    index = slots.clamp(0, PHYSICAL_MODULE_SLOTS - 1)
    if values.ndim == 2:
        return values.gather(1, index.unsqueeze(1)).squeeze(1)
    trailing = values.shape[2:]
    gather_index = index.reshape(index.shape[0], 1, *([1] * len(trailing))).expand(
        index.shape[0], 1, *trailing
    )
    return values.gather(1, gather_index).squeeze(1)


def _module_one_hot(slots: torch.Tensor, valid: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return functional.one_hot(
        slots.clamp(0, PHYSICAL_MODULE_SLOTS - 1),
        num_classes=PHYSICAL_MODULE_SLOTS,
    ).to(dtype=dtype) * valid.unsqueeze(1).to(dtype=dtype)


def _stable_sigmoid_f32(value: torch.Tensor) -> torch.Tensor:
    clipped = torch.clamp(value, -16.0, 16.0)
    positive_exp = torch.exp(-clipped)
    positive_denominator = torch.add(torch.ones_like(positive_exp), positive_exp)
    positive = torch.div(torch.ones_like(positive_denominator), positive_denominator)
    negative_exp = torch.exp(clipped)
    negative_denominator = torch.add(torch.ones_like(negative_exp), negative_exp)
    negative = torch.div(negative_exp, negative_denominator)
    return torch.where(clipped >= 0.0, positive, negative)


def _gated_step(
    external: torch.Tensor,
    routed_previous: torch.Tensor,
    previous_state: torch.Tensor,
    expression_scale: torch.Tensor,
    developmental_gain: torch.Tensor,
    parameters: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    aggregate_sum = torch.add(external, routed_previous)
    aggregate = torch.clamp(aggregate_sum, -1.0, 1.0)

    reset_input = torch.mul(parameters[:, 1], aggregate)
    reset_state = torch.mul(parameters[:, 2], previous_state)
    reset_sum_1 = torch.add(parameters[:, 0], reset_input)
    reset_pre = torch.clamp(torch.add(reset_sum_1, reset_state), -16.0, 16.0)
    reset = _stable_sigmoid_f32(reset_pre)

    update_input = torch.mul(parameters[:, 4], aggregate)
    update_state = torch.mul(parameters[:, 5], previous_state)
    update_sum_1 = torch.add(parameters[:, 3], update_input)
    update_pre = torch.clamp(torch.add(update_sum_1, update_state), -16.0, 16.0)
    update = _stable_sigmoid_f32(update_pre)

    reset_previous = torch.mul(reset, previous_state)
    candidate_input = torch.mul(parameters[:, 7], aggregate)
    candidate_state = torch.mul(parameters[:, 8], reset_previous)
    candidate_sum_1 = torch.add(parameters[:, 6], candidate_input)
    candidate_pre = torch.clamp(
        torch.add(candidate_sum_1, candidate_state), -16.0, 16.0
    )
    candidate = torch.tanh(candidate_pre)

    keep = torch.mul(torch.sub(torch.ones_like(update), update), previous_state)
    write = torch.mul(update, candidate)
    next_state = torch.clamp(torch.add(keep, write), -1.0, 1.0)
    base_output = torch.clamp(torch.mul(parameters[:, 9], next_state), -1.0, 1.0)
    expressed = torch.clamp(torch.mul(expression_scale, base_output), -1.0, 1.0)
    effective = torch.where(
        developmental_gain == 1.0,
        expressed,
        torch.clamp(torch.mul(developmental_gain, expressed), -1.0, 1.0),
    )
    return next_state, effective


def execute_modular_tick(
    planes: MutableMapping[str, torch.Tensor],
    observations: torch.Tensor,
    fatigue: torch.Tensor,
    stage_indices: torch.Tensor,
    *,
    action_width: int = ACTION_WIDTH,
) -> ModularTickResult:
    """Execute one deterministic modular tick in the supplied resident planes."""

    resolved, namespace = _resolve_planes(planes)
    capacity, device = _validate_inputs(
        resolved,
        observations,
        fatigue,
        stage_indices,
        action_width=action_width,
    )
    module_occupied = resolved["module_occupied"]
    _assert_tensor(
        torch.all(
            torch.logical_or(
                torch.logical_not(module_occupied),
                resolved["agent_occupied"].unsqueeze(1),
            )
        ),
        "module occupancy requires an occupied agent slot",
    )
    generic = torch.logical_and(
        module_occupied,
        resolved["module_type_code"] == _GENERIC_MODULE_TYPE_CODE,
    )
    operator = resolved["operator_code"]
    _assert_tensor(
        torch.all(
            torch.logical_or(
                torch.logical_not(generic),
                torch.logical_or(
                    operator == _INTEGRATOR_OPERATOR_CODE,
                    operator == _GATED_OPERATOR_CODE,
                ),
            )
        ),
        "generic modules require a supported operator code",
    )

    old_latch_present = resolved["development_latch_present"].clone()
    old_latch_state = resolved["development_latch_state"].clone()
    effective_gain = torch.zeros(
        (capacity, PHYSICAL_MODULE_SLOTS), dtype=torch.float32, device=device
    )
    dependency_ready = torch.zeros_like(generic)
    next_latch_present = old_latch_present.clone()
    next_latch_state = old_latch_state.clone()
    development_seen = torch.zeros(
        (capacity, PHYSICAL_MODULE_SLOTS), dtype=torch.int32, device=device
    )
    dependency_order = resolved["dependency_topological_order"]
    stage_gains = resolved["development_stage_gains"]
    prerequisite_slots = resolved["development_prerequisite_module_slot"]
    prerequisite_count = resolved["development_prerequisite_count"]
    latch_enabled = resolved["development_latch_enabled"]
    binding_positions = torch.arange(2, dtype=torch.int64, device=device).unsqueeze(0)
    _assert_tensor(
        torch.all(
            torch.logical_and(prerequisite_count >= 0, prerequisite_count <= 2)
        ),
        "development prerequisite count is outside the fixed envelope",
    )

    for order in range(PHYSICAL_MODULE_SLOTS):
        slots = dependency_order[:, order].to(dtype=torch.int64)
        valid = slots >= 0
        selected_generic = _gather_module_row(generic, slots)
        step_valid = torch.logical_and(valid, selected_generic)
        _assert_tensor(
            torch.all(torch.logical_or(torch.logical_not(valid), selected_generic)),
            "development order may contain generic modules only",
        )
        one_hot_bool = _module_one_hot(slots, step_valid, torch.bool)
        development_seen = torch.add(
            development_seen,
            one_hot_bool.to(dtype=torch.int32),
        )

        selected_gains = _gather_module_row(stage_gains, slots)
        requested = selected_gains.gather(1, stage_indices.unsqueeze(1)).squeeze(1)
        selected_prerequisite_slots = _gather_module_row(prerequisite_slots, slots).to(
            dtype=torch.int64
        )
        selected_prerequisite_count = _gather_module_row(
            prerequisite_count, slots
        ).to(dtype=torch.int64)
        prerequisite_mask = binding_positions < selected_prerequisite_count.unsqueeze(1)
        prerequisite_gain = effective_gain.gather(
            1,
            selected_prerequisite_slots.clamp(0, PHYSICAL_MODULE_SLOTS - 1),
        )
        raw_ready = torch.all(
            torch.logical_or(torch.logical_not(prerequisite_mask), prerequisite_gain > 0.0),
            dim=1,
        )
        selected_latch_enabled = _gather_module_row(latch_enabled, slots)
        previously_latched = torch.logical_and(
            _gather_module_row(old_latch_present, slots),
            _gather_module_row(old_latch_state, slots),
        )
        gate_ready = torch.logical_or(raw_ready, previously_latched)
        next_latched = torch.logical_and(
            selected_latch_enabled,
            torch.logical_or(previously_latched, raw_ready),
        )
        final_gain = torch.where(gate_ready, requested, torch.zeros_like(requested))
        effective_gain = torch.where(one_hot_bool, final_gain.unsqueeze(1), effective_gain)
        dependency_ready = torch.where(
            one_hot_bool,
            gate_ready.unsqueeze(1),
            dependency_ready,
        )
        next_latch_present = torch.where(
            one_hot_bool,
            next_latched.unsqueeze(1),
            next_latch_present,
        )
        next_latch_state = torch.where(
            one_hot_bool,
            next_latched.unsqueeze(1),
            next_latch_state,
        )

    _assert_tensor(
        torch.all(development_seen == generic.to(dtype=torch.int32)),
        "development order must contain every generic module exactly once",
    )
    resolved["development_latch_present"].copy_(
        torch.where(generic, next_latch_present, old_latch_present)
    )
    resolved["development_latch_state"].copy_(
        torch.where(generic, next_latch_state, old_latch_state)
    )
    resolved["development_effective_gain_present"].copy_(
        torch.where(
            generic,
            torch.ones_like(resolved["development_effective_gain_present"]),
            resolved["development_effective_gain_present"],
        )
    )
    resolved["development_effective_gain"].copy_(
        torch.where(
            generic,
            effective_gain,
            resolved["development_effective_gain"],
        )
    )

    semantic_sources = torch.stack(
        (
            observations[:, LOCAL_OBSERVATION_WIDTH],
            observations[:, LOCAL_OBSERVATION_WIDTH + 1],
        ),
        dim=1,
    )
    input_count = resolved["input_binding_count"].to(dtype=torch.int64)
    input_codes = resolved["input_semantic_code"].to(dtype=torch.int64)
    input_mask = torch.logical_and(
        binding_positions.reshape(1, 1, 2) < input_count.unsqueeze(2),
        resolved["input_enabled"],
    )
    _assert_tensor(
        torch.all(torch.logical_and(input_count >= 0, input_count <= 2)),
        "input binding count is outside the fixed envelope",
    )
    _assert_tensor(
        torch.all(
            torch.logical_or(
                torch.logical_not(input_mask),
                torch.logical_and(input_codes >= 0, input_codes < 2),
            )
        ),
        "active input binding semantic code is invalid",
    )
    expanded_sources = semantic_sources.unsqueeze(1).expand(
        capacity, PHYSICAL_MODULE_SLOTS, 2
    )
    routed_source = expanded_sources.gather(2, input_codes.clamp(0, 1))
    routed_contribution = torch.mul(resolved["input_gain"], routed_source)
    routed_contribution = torch.where(
        input_mask,
        routed_contribution,
        torch.zeros_like(routed_contribution),
    )
    routed_observation = torch.zeros(
        (capacity, PHYSICAL_MODULE_SLOTS), dtype=torch.float32, device=device
    )
    for binding in range(2):
        routed_observation = torch.add(
            routed_observation,
            routed_contribution[:, :, binding],
        )
    routed_observation = torch.clamp(
        routed_observation, -MAX_ROUTED_INPUT, MAX_ROUTED_INPUT
    )

    previous_output = resolved["module_previous_output"][..., 0].clone()
    stored_state = resolved["module_state"][..., 0].clone()
    old_state = torch.where(
        resolved["module_state_present"],
        stored_state,
        torch.zeros_like(stored_state),
    )
    dormant = resolved["module_dormant"]
    source_enabled = torch.logical_and(
        generic,
        torch.logical_and(torch.logical_not(dormant), effective_gain > 0.0),
    )
    connection_source_available = torch.logical_and(
        generic,
        torch.logical_not(dormant),
    )
    previous_for_connections = torch.where(
        effective_gain > 0.0,
        previous_output,
        torch.zeros_like(previous_output),
    )
    destination_enabled = source_enabled
    routed_previous = torch.zeros_like(old_state)
    evaluated_connections = torch.zeros((capacity,), dtype=torch.int64, device=device)
    connection_occupied = resolved["connection_occupied"]
    connection_order = resolved["connection_execution_order"]
    source_slots = resolved["connection_src_module_slot"].to(dtype=torch.int64)
    destination_slots = resolved["connection_dst_module_slot"].to(dtype=torch.int64)
    weights = resolved["connection_weight"]
    connection_seen = torch.zeros(
        (capacity, PHYSICAL_CONNECTION_SLOTS), dtype=torch.int32, device=device
    )

    for order in range(PHYSICAL_CONNECTION_SLOTS):
        selected = torch.logical_and(connection_occupied, connection_order == order)
        connection_seen = torch.add(connection_seen, selected.to(dtype=torch.int32))
        selected_count = torch.sum(selected.to(dtype=torch.int64), dim=1)
        _assert_tensor(
            torch.all(selected_count <= 1),
            "connection execution order must be unique per agent",
        )
        present = selected_count == 1
        source_slot = torch.sum(
            torch.where(selected, source_slots, torch.zeros_like(source_slots)), dim=1
        )
        destination_slot = torch.sum(
            torch.where(selected, destination_slots, torch.zeros_like(destination_slots)),
            dim=1,
        )
        weight = torch.sum(
            torch.where(selected, weights, torch.zeros_like(weights)), dim=1
        )
        endpoint_range = torch.logical_and(
            torch.logical_and(source_slot >= 0, source_slot < PHYSICAL_MODULE_SLOTS),
            torch.logical_and(
                destination_slot >= 0,
                destination_slot < PHYSICAL_MODULE_SLOTS,
            ),
        )
        _assert_tensor(
            torch.all(torch.logical_or(torch.logical_not(present), endpoint_range)),
            "connection endpoint is outside the fixed module envelope",
        )
        source_value = _gather_module_row(previous_for_connections, source_slot)
        source_active = _gather_module_row(connection_source_available, source_slot)
        destination_active = _gather_module_row(destination_enabled, destination_slot)
        evaluated = torch.logical_and(
            present,
            torch.logical_and(source_active, destination_active),
        )
        contribution = torch.where(
            evaluated,
            torch.mul(weight, source_value),
            torch.zeros_like(weight),
        )
        routed_previous = torch.add(
            routed_previous,
            _module_one_hot(destination_slot, evaluated, torch.float32)
            * contribution.unsqueeze(1),
        )
        evaluated_connections = torch.add(
            evaluated_connections,
            evaluated.to(dtype=torch.int64),
        )
    _assert_tensor(
        torch.all(connection_seen == connection_occupied.to(dtype=torch.int32)),
        "connection execution order must contain every occupied connection exactly once",
    )

    execution_order = resolved["module_execution_order"]
    execution_seen = torch.zeros_like(development_seen)
    next_state = stored_state.clone()
    next_output = previous_output.clone()
    state_before_audit = old_state.clone()
    executing_integrators = torch.zeros((capacity,), dtype=torch.int64, device=device)
    executing_gated = torch.zeros((capacity,), dtype=torch.int64, device=device)
    executing_modules = torch.zeros((capacity,), dtype=torch.int64, device=device)
    changing_state = torch.zeros((capacity,), dtype=torch.int64, device=device)

    for order in range(PHYSICAL_MODULE_SLOTS):
        slots = execution_order[:, order].to(dtype=torch.int64)
        valid = slots >= 0
        selected_generic = _gather_module_row(generic, slots)
        step_valid = torch.logical_and(valid, selected_generic)
        _assert_tensor(
            torch.all(torch.logical_or(torch.logical_not(valid), selected_generic)),
            "module execution order may contain generic modules only",
        )
        one_hot_bool = _module_one_hot(slots, step_valid, torch.bool)
        execution_seen = torch.add(execution_seen, one_hot_bool.to(dtype=torch.int32))
        selected_old_state = _gather_module_row(old_state, slots)
        selected_routed_observation = _gather_module_row(routed_observation, slots)
        selected_routed_previous = _gather_module_row(routed_previous, slots)
        selected_gain = _gather_module_row(effective_gain, slots)
        selected_dormant = _gather_module_row(dormant, slots)
        selected_operator = _gather_module_row(operator, slots)
        selected_integrator = _gather_module_row(
            resolved["integrator_parameters"], slots
        )
        selected_gated = _gather_module_row(resolved["gated_parameters"], slots)
        combined_external = torch.where(
            selected_routed_observation == 0.0,
            fatigue,
            torch.add(fatigue, selected_routed_observation),
        )
        executing = torch.logical_and(
            step_valid,
            torch.logical_and(torch.logical_not(selected_dormant), selected_gain > 0.0),
        )
        integrator_mask = torch.logical_and(
            executing, selected_operator == _INTEGRATOR_OPERATOR_CODE
        )
        gated_mask = torch.logical_and(
            executing, selected_operator == _GATED_OPERATOR_CODE
        )

        integrator_state = torch.mul(selected_integrator[:, 1], selected_old_state)
        integrator_state = torch.add(
            integrator_state,
            torch.mul(selected_integrator[:, 2], combined_external),
        )
        integrator_state = torch.clamp(
            torch.add(integrator_state, selected_routed_previous), -1.0, 1.0
        )
        integrator_output = torch.mul(
            selected_integrator[:, 0], selected_integrator[:, 3]
        )
        integrator_output = torch.clamp(
            torch.mul(integrator_output, integrator_state), -1.0, 1.0
        )
        integrator_output = torch.where(
            selected_gain == 1.0,
            integrator_output,
            torch.clamp(torch.mul(selected_gain, integrator_output), -1.0, 1.0),
        )
        gated_state, gated_output = _gated_step(
            combined_external,
            selected_routed_previous,
            selected_old_state,
            selected_integrator[:, 0],
            selected_gain,
            selected_gated,
        )
        selected_next_state = torch.where(
            integrator_mask,
            integrator_state,
            torch.where(gated_mask, gated_state, selected_old_state),
        )
        selected_next_output = torch.where(
            integrator_mask,
            integrator_output,
            torch.where(gated_mask, gated_output, torch.zeros_like(gated_output)),
        )
        next_state = torch.where(
            one_hot_bool,
            selected_next_state.unsqueeze(1),
            next_state,
        )
        next_output = torch.where(
            one_hot_bool,
            selected_next_output.unsqueeze(1),
            next_output,
        )
        executing_integrators = torch.add(
            executing_integrators, integrator_mask.to(dtype=torch.int64)
        )
        executing_gated = torch.add(
            executing_gated, gated_mask.to(dtype=torch.int64)
        )
        executing_modules = torch.add(
            executing_modules, executing.to(dtype=torch.int64)
        )
        changing_state = torch.add(
            changing_state,
            torch.logical_and(
                step_valid,
                selected_next_state != selected_old_state,
            ).to(dtype=torch.int64),
        )

    _assert_tensor(
        torch.all(execution_seen == generic.to(dtype=torch.int32)),
        "module execution order must contain every generic module exactly once",
    )
    resolved["module_state_present"].copy_(
        torch.where(generic, torch.ones_like(resolved["module_state_present"]), resolved["module_state_present"])
    )
    resolved["module_output_present"].copy_(
        torch.where(generic, torch.ones_like(resolved["module_output_present"]), resolved["module_output_present"])
    )
    resolved["module_state"][..., 0].copy_(next_state)
    resolved["module_previous_output"][..., 0].copy_(next_output)

    legacy_rest = torch.zeros((capacity,), dtype=torch.float32, device=device)
    routed_action = torch.zeros(
        (capacity, ACTION_WIDTH), dtype=torch.float32, device=device
    )
    output_count = resolved["output_binding_count"].to(dtype=torch.int64)
    output_actions = resolved["output_action_index"].to(dtype=torch.int64)
    output_enabled = resolved["output_enabled"]
    output_gain = resolved["output_gain"]
    executable_output_bindings = torch.zeros(
        (capacity,), dtype=torch.int64, device=device
    )
    _assert_tensor(
        torch.all(torch.logical_and(output_count >= 0, output_count <= 2)),
        "output binding count is outside the fixed envelope",
    )

    for order in range(PHYSICAL_MODULE_SLOTS):
        slots = execution_order[:, order].to(dtype=torch.int64)
        valid = slots >= 0
        selected_generic = _gather_module_row(generic, slots)
        step_valid = torch.logical_and(valid, selected_generic)
        selected_output = _gather_module_row(next_output, slots)
        legacy_rest = torch.add(
            legacy_rest,
            torch.where(step_valid, selected_output, torch.zeros_like(selected_output)),
        )
        selected_count = _gather_module_row(output_count, slots)
        selected_actions = _gather_module_row(output_actions, slots)
        selected_enabled = _gather_module_row(output_enabled, slots)
        selected_output_gain = _gather_module_row(output_gain, slots)
        selected_executing = torch.logical_and(
            step_valid,
            torch.logical_and(
                torch.logical_not(_gather_module_row(dormant, slots)),
                _gather_module_row(effective_gain, slots) > 0.0,
            ),
        )
        for binding in range(2):
            active_binding = torch.logical_and(
                step_valid,
                torch.logical_and(
                    selected_count > binding,
                    selected_enabled[:, binding],
                ),
            )
            action_index = selected_actions[:, binding]
            _assert_tensor(
                torch.all(
                    torch.logical_or(
                        torch.logical_not(active_binding),
                        torch.logical_and(action_index >= 0, action_index < ACTION_WIDTH),
                    )
                ),
                "active output binding action index is invalid",
            )
            contribution = torch.where(
                active_binding,
                torch.mul(selected_output_gain[:, binding], selected_output),
                torch.zeros_like(selected_output),
            )
            action_one_hot = functional.one_hot(
                action_index.clamp(0, ACTION_WIDTH - 1),
                num_classes=ACTION_WIDTH,
            ).to(dtype=torch.float32)
            routed_action = torch.add(
                routed_action,
                action_one_hot * contribution.unsqueeze(1),
            )
            executable_output_bindings = torch.add(
                executable_output_bindings,
                torch.logical_and(active_binding, selected_executing).to(
                    dtype=torch.int64
                ),
            )

    legacy_rest = torch.clamp(
        legacy_rest, -MAX_ACTION_RESIDUAL, MAX_ACTION_RESIDUAL
    )
    action_residuals = torch.clamp(
        routed_action, -MAX_ACTION_RESIDUAL, MAX_ACTION_RESIDUAL
    )
    routed_rest = routed_action[:, int(ACT_REST)]
    combined_rest = torch.where(
        routed_rest == 0.0,
        legacy_rest,
        torch.clamp(
            torch.add(legacy_rest, routed_rest),
            -MAX_ACTION_RESIDUAL,
            MAX_ACTION_RESIDUAL,
        ),
    )
    rest_selector = functional.one_hot(
        torch.full(
            (capacity,), int(ACT_REST), dtype=torch.int64, device=device
        ),
        num_classes=ACTION_WIDTH,
    ).to(dtype=torch.bool)
    action_residuals = torch.where(
        rest_selector,
        combined_rest.unsqueeze(1),
        action_residuals,
    )

    allocated_modules = torch.sum(generic.to(dtype=torch.int64), dim=1)
    occupied_agents = resolved["agent_occupied"]
    structured_agents = torch.logical_and(occupied_agents, allocated_modules > 0)
    neutral_agents = torch.logical_and(occupied_agents, allocated_modules == 0)
    active_input_bindings = torch.sum(
        torch.logical_and(
            input_mask,
            source_enabled.unsqueeze(2),
        ).to(dtype=torch.int64),
        dim=(1, 2),
    )
    dependency_edges = torch.sum(
        torch.where(
            source_enabled,
            prerequisite_count.to(dtype=torch.int64),
            torch.zeros_like(prerequisite_count, dtype=torch.int64),
        ),
        dim=1,
    )
    audit = {
        "state_before": state_before_audit,
        "state_after": next_state.clone(),
        "module_output": next_output.clone(),
        "development_effective_gain": effective_gain.clone(),
        "development_dependency_ready": dependency_ready.clone(),
        "development_latch_state": next_latch_state.clone(),
        "routed_observation": routed_observation,
        "previous_tick_connection_input": routed_previous,
        "legacy_rest_residual": legacy_rest,
        "routed_action_residual_before_rest_combine": routed_action,
        "action_residuals": action_residuals,
        "allocated_module_count": allocated_modules,
        "executing_module_count": executing_modules,
        "executing_integrator_count": executing_integrators,
        "executing_gated_count": executing_gated,
        "dormant_module_count": torch.sum(
            torch.logical_and(generic, dormant).to(dtype=torch.int64), dim=1
        ),
        "evaluated_connection_count": evaluated_connections,
        "active_input_binding_count": active_input_bindings,
        "active_output_binding_count": executable_output_bindings,
        "active_dependency_edge_count": dependency_edges,
        "changing_state_module_count": changing_state,
        "structured_agent_count": torch.sum(structured_agents.to(dtype=torch.int64)),
        "neutral_agent_count": torch.sum(neutral_agents.to(dtype=torch.int64)),
        "structural_cost_counts": resolved["structural_cost_counts"].clone(),
        "structural_cost_components": resolved["structural_cost_components"].clone(),
        "structural_cost_total": resolved["structural_cost_total"].clone(),
        "rng_draw_count": torch.zeros((), dtype=torch.int64, device=device),
    }
    return ModularTickResult(
        planes=planes,
        action_residuals=action_residuals,
        audit_tensors=audit,
        receipt=ModularTickReceipt(
            schema_version=MODULAR_TICK_SCHEMA_VERSION,
            lane=MODULAR_TICK_LANE,
            field_namespace=namespace,
            plane_count=len(resolved),
            action_width=ACTION_WIDTH,
        ),
    )


__all__ = [
    "ACTION_WIDTH",
    "LOCAL_OBSERVATION_WIDTH",
    "MODULAR_TICK_LANE",
    "MODULAR_TICK_SCHEMA_VERSION",
    "OBSERVATION_WIDTH",
    "ModularTickError",
    "ModularTickReceipt",
    "ModularTickResult",
    "execute_modular_tick",
]
