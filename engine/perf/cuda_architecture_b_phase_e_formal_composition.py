from __future__ import annotations

"""Independent CPU oracle for the bounded Phase E policy composition seam.

This module executes exactly one resident modular tick and one deterministic
fixed-MLP policy choice for one cell.  It deliberately does not import or call
the CUDA policy wrappers.  Every input and intermediate needed by a formal gate
is returned as raw data; no self-attested pass/fail booleans are produced.
"""

from copy import deepcopy
from dataclasses import dataclass
import random
from typing import Final, Mapping, MutableMapping

import numpy as np
import torch

from agents.batched_policy import forward_batched_mlp
from perf.cuda_architecture_b_modular_state_abi import (
    MODULAR_RUNTIME_MUTABLE_FIELD_NAMES,
    build_modular_storage_layout,
)
from perf.cuda_architecture_b_modular_tick import execute_modular_tick


PHASE_E_FORMAL_COMPOSITION_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_phase_e_cpu_formal_composition_v1"
)
PHASE_E_FORMAL_COMPOSITION_SCOPE: Final[str] = (
    "one_cell_one_tick_fixed_mlp_fixed_identity_no_lifecycle_v1"
)
PHASE_E_FORMAL_COMPOSITION_EXCLUDED_OPERATIONS: Final[tuple[str, ...]] = (
    "birth",
    "death",
    "cull",
    "refill",
    "turnover",
    "prune",
    "reuse",
    "migration",
)
ACTION_WIDTH: Final[int] = 7
MASKED_LOGIT_FLOOR: Final[float] = -1.0e9


class PhaseEFormalCompositionError(RuntimeError):
    """The bounded CPU composition inputs are malformed or out of scope."""


@dataclass(frozen=True)
class PhaseEFormalCompositionRaw:
    """Raw input, intermediate, output, plane, and RNG evidence."""

    schema_version: str
    scope: str
    plane_field_names: tuple[str, ...]
    runtime_mutable_plane_fields: tuple[str, ...]
    planes_before: dict[str, torch.Tensor]
    planes_after: dict[str, torch.Tensor]
    modular_observations: torch.Tensor
    fatigue: torch.Tensor
    stage_indices: torch.Tensor
    policy_forward_inputs: dict[str, torch.Tensor]
    base_logits: torch.Tensor
    modular_action_residuals: torch.Tensor
    composed_logits: torch.Tensor
    validity_mask_before: torch.Tensor
    validity_mask_after: torch.Tensor
    authority_mask_before: torch.Tensor
    authority_mask_after: torch.Tensor
    effective_action_mask: torch.Tensor
    masked_logits: torch.Tensor
    deterministic_actions: torch.Tensor
    modular_audit_tensors: dict[str, torch.Tensor]
    modular_receipt: dict[str, object]
    rng_stream_indices_before: dict[str, int]
    rng_stream_indices_after: dict[str, int]
    torch_rng_state_before: torch.Tensor
    torch_rng_state_after: torch.Tensor
    numpy_rng_state_before: dict[str, object]
    numpy_rng_state_after: dict[str, object]
    python_rng_state_before: tuple[object, ...]
    python_rng_state_after: tuple[object, ...]


def _clone_tensors(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        str(name): tensor.detach().clone().contiguous()
        for name, tensor in values.items()
    }


def _numpy_rng_state() -> dict[str, object]:
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "bit_generator": str(name),
        "keys": torch.from_numpy(keys.copy()),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _require_cpu_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise PhaseEFormalCompositionError(f"{name} must be a tensor")
    if tensor.device.type != "cpu":
        raise PhaseEFormalCompositionError(f"{name} must remain on CPU")
    if tensor.dtype != dtype or tuple(tensor.shape) != shape:
        raise PhaseEFormalCompositionError(
            f"{name} must have dtype={dtype} shape={shape}"
        )
    if not tensor.is_contiguous():
        raise PhaseEFormalCompositionError(f"{name} must be contiguous")
    if tensor.requires_grad:
        raise PhaseEFormalCompositionError(f"{name} must not require gradients")


def _validated_rng_indices(values: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(values, Mapping) or not values:
        raise PhaseEFormalCompositionError(
            "rng_stream_indices must be a nonempty mapping"
        )
    normalized: dict[str, int] = {}
    for name, value in sorted(values.items()):
        if type(name) is not str or not name:
            raise PhaseEFormalCompositionError("RNG stream names must be nonempty strings")
        if type(value) is not int or value < 0:
            raise PhaseEFormalCompositionError(
                f"RNG stream index must be a nonnegative integer: {name}"
            )
        normalized[name] = value
    return normalized


def execute_phase_e_cpu_formal_composition(
    *,
    planes: MutableMapping[str, torch.Tensor],
    modular_observations: torch.Tensor,
    fatigue: torch.Tensor,
    stage_indices: torch.Tensor,
    policy_observations: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    w3: torch.Tensor,
    b3: torch.Tensor,
    attack_bias: torch.Tensor,
    action_temperature: torch.Tensor,
    validity_mask: torch.Tensor,
    authority_mask: torch.Tensor,
    rng_stream_indices: Mapping[str, int],
) -> PhaseEFormalCompositionRaw:
    """Execute and expose one bounded CPU modular-to-policy composition."""

    layout = build_modular_storage_layout(1)
    field_names = tuple(field.name for field in layout.fields)
    if len(field_names) != 57:
        raise PhaseEFormalCompositionError("the Phase E resident plane count is not 57")
    if not isinstance(planes, MutableMapping) or tuple(sorted(planes)) != tuple(
        sorted(field_names)
    ):
        raise PhaseEFormalCompositionError(
            "planes must be the exact unprefixed one-cell 57-plane bank"
        )
    for field in layout.fields:
        _require_cpu_tensor(
            planes[field.name],
            name=f"planes.{field.name}",
            dtype=field.torch_dtype,
            shape=field.shape,
        )

    _require_cpu_tensor(
        modular_observations,
        name="modular_observations",
        dtype=torch.float32,
        shape=(1, 27),
    )
    _require_cpu_tensor(fatigue, name="fatigue", dtype=torch.float32, shape=(1,))
    _require_cpu_tensor(
        stage_indices,
        name="stage_indices",
        dtype=torch.int64,
        shape=(1,),
    )
    if not isinstance(policy_observations, torch.Tensor) or policy_observations.ndim != 2:
        raise PhaseEFormalCompositionError(
            "policy_observations must be a rank-two CPU float32 tensor"
        )
    observation_width = int(policy_observations.shape[1])
    if w1.ndim != 3:
        raise PhaseEFormalCompositionError("w1 must be rank three")
    hidden_size = int(w1.shape[1])
    policy_shapes = {
        "policy_observations": (policy_observations, (1, observation_width)),
        "w1": (w1, (1, hidden_size, observation_width)),
        "b1": (b1, (1, hidden_size)),
        "w2": (w2, (1, hidden_size, hidden_size)),
        "b2": (b2, (1, hidden_size)),
        "w3": (w3, (1, ACTION_WIDTH, hidden_size)),
        "b3": (b3, (1, ACTION_WIDTH)),
        "attack_bias": (attack_bias, (1,)),
        "action_temperature": (action_temperature, (1,)),
    }
    for name, (tensor, shape) in policy_shapes.items():
        _require_cpu_tensor(tensor, name=name, dtype=torch.float32, shape=shape)
    for name, mask in (
        ("validity_mask", validity_mask),
        ("authority_mask", authority_mask),
    ):
        _require_cpu_tensor(
            mask,
            name=name,
            dtype=torch.bool,
            shape=(1, ACTION_WIDTH),
        )
    effective_mask = torch.logical_and(validity_mask, authority_mask)
    if not bool(torch.all(torch.any(effective_mask, dim=1))):
        raise PhaseEFormalCompositionError(
            "validity and authority masks must leave an available action"
        )

    indices_before = _validated_rng_indices(rng_stream_indices)
    torch_before = torch.get_rng_state().clone()
    numpy_before = _numpy_rng_state()
    python_before = deepcopy(random.getstate())
    planes_before = _clone_tensors(planes)
    working_planes = _clone_tensors(planes)
    policy_inputs = _clone_tensors(
        {
            "policy_observations": policy_observations,
            "w1": w1,
            "b1": b1,
            "w2": w2,
            "b2": b2,
            "w3": w3,
            "b3": b3,
            "attack_bias": attack_bias,
            "action_temperature": action_temperature,
        }
    )

    tick = execute_modular_tick(
        working_planes,
        modular_observations,
        fatigue,
        stage_indices,
        action_width=ACTION_WIDTH,
    )
    base_logits = forward_batched_mlp(
        policy_observations,
        w1,
        b1,
        w2,
        b2,
        w3,
        b3,
        attack_bias=attack_bias,
        action_temperature=action_temperature,
    )
    modular_residuals = tick.action_residuals.detach().clone()
    composed_logits = torch.add(base_logits, modular_residuals)
    validity_before = validity_mask.detach().clone()
    authority_before = authority_mask.detach().clone()
    effective_action_mask = torch.logical_and(validity_mask, authority_mask)
    masked_logits = torch.where(
        effective_action_mask,
        composed_logits,
        torch.full_like(composed_logits, MASKED_LOGIT_FLOOR),
    )
    deterministic_actions = torch.argmax(masked_logits, dim=1).to(torch.int32)

    indices_after = _validated_rng_indices(rng_stream_indices)
    torch_after = torch.get_rng_state().clone()
    numpy_after = _numpy_rng_state()
    python_after = deepcopy(random.getstate())
    return PhaseEFormalCompositionRaw(
        schema_version=PHASE_E_FORMAL_COMPOSITION_SCHEMA_VERSION,
        scope=PHASE_E_FORMAL_COMPOSITION_SCOPE,
        plane_field_names=field_names,
        runtime_mutable_plane_fields=MODULAR_RUNTIME_MUTABLE_FIELD_NAMES,
        planes_before=planes_before,
        planes_after=_clone_tensors(working_planes),
        modular_observations=modular_observations.detach().clone(),
        fatigue=fatigue.detach().clone(),
        stage_indices=stage_indices.detach().clone(),
        policy_forward_inputs=policy_inputs,
        base_logits=base_logits.detach().clone(),
        modular_action_residuals=modular_residuals,
        composed_logits=composed_logits.detach().clone(),
        validity_mask_before=validity_before,
        validity_mask_after=validity_mask.detach().clone(),
        authority_mask_before=authority_before,
        authority_mask_after=authority_mask.detach().clone(),
        effective_action_mask=effective_action_mask.detach().clone(),
        masked_logits=masked_logits.detach().clone(),
        deterministic_actions=deterministic_actions.detach().clone(),
        modular_audit_tensors=_clone_tensors(tick.audit_tensors),
        modular_receipt=tick.receipt.to_dict(),
        rng_stream_indices_before=indices_before,
        rng_stream_indices_after=indices_after,
        torch_rng_state_before=torch_before,
        torch_rng_state_after=torch_after,
        numpy_rng_state_before=numpy_before,
        numpy_rng_state_after=numpy_after,
        python_rng_state_before=python_before,
        python_rng_state_after=python_after,
    )


def build_phase_e_formal_composition_fixture() -> PhaseEFormalCompositionRaw:
    """Build the canonical one-cell receipt fixture without importing tests."""

    layout = build_modular_storage_layout(1)
    planes = {
        field.name: torch.full(
            field.shape,
            field.fill_value,
            dtype=field.torch_dtype,
        ).contiguous()
        for field in layout.fields
    }
    planes["agent_occupied"][0] = True
    planes["agent_id"][0] = 41
    planes["agent_slot_token"][0] = 9001
    planes["module_count"][0] = 1
    planes["module_execution_order"][0, 0] = 0
    planes["dependency_topological_order"][0, 0] = 0
    planes["module_occupied"][0, 0] = True
    planes["module_local_id_index"][0, 0] = 100
    planes["module_instance_id_index"][0, 0] = 200
    planes["module_homology_id_index"][0, 0] = 300
    planes["module_type_index"][0, 0] = 400
    planes["module_type_code"][0, 0] = 1
    planes["identity_origin_code"][0, 0] = 2
    planes["operator_code"][0, 0] = 1
    planes["integrator_parameters"][0, 0] = torch.tensor(
        [0.75, 0.5, 0.75, 0.5], dtype=torch.float32
    )
    planes["module_state_present"][0, 0] = True
    planes["module_state"][0, 0, 0] = 0.2
    planes["module_output_present"][0, 0] = True
    planes["module_previous_output"][0, 0, 0] = 0.1
    planes["input_binding_count"][0, 0] = 1
    planes["input_semantic_code"][0, 0, 0] = 0
    planes["input_semantic_id_index"][0, 0, 0] = 500
    planes["input_gain"][0, 0, 0] = 1.0
    planes["input_enabled"][0, 0, 0] = True
    planes["output_binding_count"][0, 0] = 1
    planes["output_semantic_code"][0, 0, 0] = 0
    planes["output_semantic_id_index"][0, 0, 0] = 600
    planes["output_action_index"][0, 0, 0] = 5
    planes["output_gain"][0, 0, 0] = 1.0
    planes["output_enabled"][0, 0, 0] = True
    planes["development_explicit"][0, 0] = True
    planes["development_stage_gains"][0, 0] = 1.0

    modular_observations = torch.zeros((1, 27), dtype=torch.float32)
    modular_observations[0, 25] = 0.8
    modular_observations[0, 26] = 0.6
    hidden_size = 3
    return execute_phase_e_cpu_formal_composition(
        planes=planes,
        modular_observations=modular_observations,
        fatigue=torch.tensor([0.1], dtype=torch.float32),
        stage_indices=torch.tensor([0], dtype=torch.int64),
        policy_observations=torch.linspace(
            -0.5, 0.5, 27, dtype=torch.float32
        ).reshape(1, 27),
        w1=torch.zeros((1, hidden_size, 27), dtype=torch.float32),
        b1=torch.zeros((1, hidden_size), dtype=torch.float32),
        w2=torch.zeros((1, hidden_size, hidden_size), dtype=torch.float32),
        b2=torch.zeros((1, hidden_size), dtype=torch.float32),
        w3=torch.zeros((1, ACTION_WIDTH, hidden_size), dtype=torch.float32),
        b3=torch.tensor(
            [[0.2, 0.05, 0.04, 0.03, 0.02, 0.0, 0.5]],
            dtype=torch.float32,
        ),
        attack_bias=torch.zeros((1,), dtype=torch.float32),
        action_temperature=torch.ones((1,), dtype=torch.float32),
        validity_mask=torch.tensor(
            [[True, True, True, True, True, True, False]], dtype=torch.bool
        ),
        authority_mask=torch.tensor(
            [[False, True, True, True, True, True, True]], dtype=torch.bool
        ),
        rng_stream_indices={"environment": 23, "policy": 17},
    )


__all__ = [
    "ACTION_WIDTH",
    "MASKED_LOGIT_FLOOR",
    "PHASE_E_FORMAL_COMPOSITION_EXCLUDED_OPERATIONS",
    "PHASE_E_FORMAL_COMPOSITION_SCHEMA_VERSION",
    "PHASE_E_FORMAL_COMPOSITION_SCOPE",
    "PhaseEFormalCompositionError",
    "PhaseEFormalCompositionRaw",
    "build_phase_e_formal_composition_fixture",
    "execute_phase_e_cpu_formal_composition",
]
