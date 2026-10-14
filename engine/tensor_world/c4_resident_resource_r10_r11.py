"""Resource and residency preparation for the executable C4 R9/R10 route.

This module does not claim either the B48 R10 rung or the composed R11 rung.
It freezes the exact caller-owned R9 staging allocation at the bounded
B48/C32 five-tick R10 envelope and provides metadata-only measurement of
an allocated staging object.  CUDA allocator counters and timing remain the
responsibility of an explicitly admitted probe.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from math import prod
from typing import Final

import torch

from .c4_resident_population_r9 import (
    TensorWorldC4ResidentPopulationR9StagingV1,
)
from .contracts import TensorWorldContractError


C4_RESIDENT_RESOURCE_R10_R11_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_resource_r10_r11_preparation_v1"
)
C4_RESIDENT_RESOURCE_R10_R11_DECISION_V1: Final[str] = "NO_GATE_DECISION"


@dataclass(frozen=True, slots=True)
class C4ResidentStagingBudgetRowV1:
    """One physical tensor in the nested caller-owned R9 staging object."""

    path: str
    shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def bytes(self) -> int:
        return prod(self.shape) * torch.empty((), dtype=self.dtype).element_size()


def _birth_rows(prefix: str, batch: int, capacity: int) -> list[C4ResidentStagingBudgetRowV1]:
    slots = (batch, capacity)
    return [
        C4ResidentStagingBudgetRowV1(f"{prefix}.spawn_uniforms", (*slots, 3), torch.float64),
        C4ResidentStagingBudgetRowV1(f"{prefix}.parent_slots", slots, torch.int32),
        C4ResidentStagingBudgetRowV1(f"{prefix}.child_slots", slots, torch.int32),
        C4ResidentStagingBudgetRowV1(f"{prefix}.child_pos_y", slots, torch.int32),
        C4ResidentStagingBudgetRowV1(f"{prefix}.child_pos_x", slots, torch.int32),
        C4ResidentStagingBudgetRowV1(f"{prefix}.child_energy", slots, torch.float32),
        C4ResidentStagingBudgetRowV1(f"{prefix}.planned_count", (batch,), torch.int64),
        C4ResidentStagingBudgetRowV1(f"{prefix}.decision_parent_slots", slots, torch.int64),
        C4ResidentStagingBudgetRowV1(f"{prefix}.decision_adjacent_counts", slots, torch.int32),
        C4ResidentStagingBudgetRowV1(f"{prefix}.decision_reason_codes", slots, torch.int32),
        C4ResidentStagingBudgetRowV1(f"{prefix}.decision_count", (batch,), torch.int64),
        C4ResidentStagingBudgetRowV1(f"{prefix}.planned_repro_cooldown", slots, torch.int32),
        C4ResidentStagingBudgetRowV1(f"{prefix}.spawn_rng_draws_added", (batch,), torch.int64),
        C4ResidentStagingBudgetRowV1(
            f"{prefix}.spawn_rng_semantic_draws_added", (batch,), torch.int64
        ),
        C4ResidentStagingBudgetRowV1(f"{prefix}.child_agent_ids", slots, torch.int64),
        C4ResidentStagingBudgetRowV1(f"{prefix}.queue_cursor", (batch,), torch.int64),
        C4ResidentStagingBudgetRowV1(f"{prefix}.event_sequence", (batch,), torch.int64),
        C4ResidentStagingBudgetRowV1(f"{prefix}.status", (batch,), torch.int32),
    ]


def c4_resident_r9_staging_budget_rows_v1(
    *, batch: int = 48, capacity: int = 32, grid_size: int = 8, streams: int = 2
) -> tuple[C4ResidentStagingBudgetRowV1, ...]:
    """Return the exact nested R9 staging ABI without allocating CUDA memory."""

    if batch != 48 or capacity <= 0 or grid_size <= 0 or streams < 2:
        raise TensorWorldContractError(
            "C4 R10/R11 preparation requires the positive B48 R9 envelope and >=2 streams"
        )
    slots = (batch, capacity)
    grid = (batch, grid_size, grid_size)
    rows = _birth_rows("turnover.birth", batch, capacity)
    rows.extend(
        [
            C4ResidentStagingBudgetRowV1("turnover.death_agent_ids", slots, torch.int64),
            C4ResidentStagingBudgetRowV1("turnover.death_slots", slots, torch.int64),
            C4ResidentStagingBudgetRowV1("turnover.death_reason_codes", slots, torch.int32),
            C4ResidentStagingBudgetRowV1("turnover.death_count", (batch,), torch.int64),
            C4ResidentStagingBudgetRowV1("turnover.virtual_free_queue", slots, torch.int64),
            C4ResidentStagingBudgetRowV1("turnover.virtual_free_count", (batch,), torch.int64),
            C4ResidentStagingBudgetRowV1("turnover.death_status", (batch,), torch.int32),
            C4ResidentStagingBudgetRowV1("turnover.shadow_alive", slots, torch.bool),
            C4ResidentStagingBudgetRowV1("turnover.shadow_agent_id", slots, torch.int64),
            C4ResidentStagingBudgetRowV1("turnover.shadow_pos_y", slots, torch.int64),
            C4ResidentStagingBudgetRowV1("turnover.shadow_pos_x", slots, torch.int64),
            C4ResidentStagingBudgetRowV1("turnover.shadow_energy", slots, torch.float32),
            C4ResidentStagingBudgetRowV1("turnover.shadow_health", slots, torch.float32),
            C4ResidentStagingBudgetRowV1("turnover.shadow_occupancy_agent", grid, torch.int64),
            C4ResidentStagingBudgetRowV1("turnover.shadow_occupancy_slot", grid, torch.int64),
            C4ResidentStagingBudgetRowV1("turnover.shadow_patch_agent", grid, torch.int32),
            C4ResidentStagingBudgetRowV1("turnover.shadow_agent_id_to_slot", slots, torch.int32),
            C4ResidentStagingBudgetRowV1("turnover.shadow_repro_cooldown", slots, torch.int32),
            C4ResidentStagingBudgetRowV1(
                "turnover.shadow_policy_energy_observation", slots, torch.float64
            ),
            C4ResidentStagingBudgetRowV1(
                "turnover.shadow_event_active_count", (batch,), torch.int64
            ),
            C4ResidentStagingBudgetRowV1(
                "turnover.shadow_event_status", (batch,), torch.int32
            ),
        ]
    )
    rows.extend(_birth_rows("mutation.birth", batch, capacity))
    rows.extend(
        [
            C4ResidentStagingBudgetRowV1("mutation.mate_slots", slots, torch.int32),
            C4ResidentStagingBudgetRowV1("mutation.decision_mate_slots", slots, torch.int64),
            C4ResidentStagingBudgetRowV1("mutation.runner_rng_draws_added", (batch,), torch.int64),
            C4ResidentStagingBudgetRowV1(
                "mutation.runner_rng_semantic_draws_added", (batch,), torch.int64
            ),
            C4ResidentStagingBudgetRowV1(
                "mutation.pcg64_post_state_words", (batch, streams, 4), torch.int64
            ),
            C4ResidentStagingBudgetRowV1(
                "mutation.pcg64_post_buffer_words", (batch, streams, 2), torch.int64
            ),
            C4ResidentStagingBudgetRowV1(
                "mutation.pcg64_post_cursors", (batch, streams), torch.int64
            ),
            C4ResidentStagingBudgetRowV1(
                "mutation.pcg64_post_semantic_draws", (batch, streams), torch.int64
            ),
            C4ResidentStagingBudgetRowV1("mutation.python_post_words", (batch, 624), torch.uint32),
            C4ResidentStagingBudgetRowV1("mutation.python_post_index", (batch,), torch.int64),
            C4ResidentStagingBudgetRowV1("mutation.python_post_has_gauss", (batch,), torch.bool),
            C4ResidentStagingBudgetRowV1(
                "mutation.python_post_gauss_next", (batch,), torch.float64
            ),
            C4ResidentStagingBudgetRowV1("mutation.python_post_raw_draws", (batch,), torch.int64),
            C4ResidentStagingBudgetRowV1(
                "mutation.python_post_semantic_draws", (batch,), torch.int64
            ),
            C4ResidentStagingBudgetRowV1(
                "mutation.child_scalar_values", (batch, capacity, 16), torch.float32
            ),
            C4ResidentStagingBudgetRowV1("mutation.mutation_occurred", slots, torch.bool),
            C4ResidentStagingBudgetRowV1("mutation.mutation_delta_total", slots, torch.float32),
            C4ResidentStagingBudgetRowV1("mutation.shadow_pos_y_i32", slots, torch.int32),
            C4ResidentStagingBudgetRowV1("mutation.shadow_pos_x_i32", slots, torch.int32),
            C4ResidentStagingBudgetRowV1("mutation.shadow_energy_f64", slots, torch.float64),
            C4ResidentStagingBudgetRowV1("mutation.planned_energy_f64", slots, torch.float64),
            C4ResidentStagingBudgetRowV1(
                "mutation.planned_parent_energy_after", slots, torch.float64
            ),
            C4ResidentStagingBudgetRowV1(
                "mutation.planned_mate_energy_after", slots, torch.float64
            ),
            C4ResidentStagingBudgetRowV1(
                "mutation.spawn_positions_i32", (batch, capacity, 2), torch.int32
            ),
            C4ResidentStagingBudgetRowV1(
                "mutation.decision_spawn_candidate_counts", slots, torch.int32
            ),
            C4ResidentStagingBudgetRowV1("mutation.reason_counts", (batch, 12), torch.int64),
            C4ResidentStagingBudgetRowV1("mutation.free_queue_head", (batch,), torch.int64),
            C4ResidentStagingBudgetRowV1(
                "mutation.planned_parent_slots_i64", slots, torch.int64
            ),
            C4ResidentStagingBudgetRowV1(
                "mutation.planned_mate_slots_i64", slots, torch.int64
            ),
            C4ResidentStagingBudgetRowV1(
                "mutation.planned_child_slots_i64", slots, torch.int64
            ),
            C4ResidentStagingBudgetRowV1(
                "mutation.planned_child_energy_f64", slots, torch.float64
            ),
            C4ResidentStagingBudgetRowV1("shadow_event_sequence", (batch,), torch.int64),
            C4ResidentStagingBudgetRowV1("preflight_status", (batch,), torch.int32),
        ]
    )
    return tuple(rows)


def _staging_tensor_rows(
    staging: TensorWorldC4ResidentPopulationR9StagingV1,
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}

    def visit(owner: object, prefix: str = "") -> None:
        if not is_dataclass(owner):
            return
        for field in fields(owner):
            value = getattr(owner, field.name)
            path = f"{prefix}.{field.name}" if prefix else field.name
            if isinstance(value, torch.Tensor):
                tensors[path] = value
            elif is_dataclass(value):
                visit(value, path)

    visit(staging)
    return tensors


def measure_c4_resident_r9_staging_v1(
    staging: TensorWorldC4ResidentPopulationR9StagingV1,
    *,
    batch: int = 48,
    capacity: int = 32,
    grid_size: int = 8,
    streams: int = 2,
) -> dict[str, object]:
    """Validate and measure one allocated staging object without synchronization."""

    if not isinstance(staging, TensorWorldC4ResidentPopulationR9StagingV1):
        raise TensorWorldContractError("C4 R10/R11 measurement requires R9 staging")
    staging.validate(
        batch=batch, capacity=capacity, grid_size=grid_size, streams=streams
    )
    expected = c4_resident_r9_staging_budget_rows_v1(
        batch=batch, capacity=capacity, grid_size=grid_size, streams=streams
    )
    actual = _staging_tensor_rows(staging)
    if set(actual) != {row.path for row in expected}:
        raise TensorWorldContractError("C4 R9 staging inventory paths drifted")
    for row in expected:
        tensor = actual[row.path]
        if tuple(tensor.shape) != row.shape or tensor.dtype != row.dtype:
            raise TensorWorldContractError(f"C4 R9 staging tensor drifted: {row.path}")
    storage_bytes = sum(
        tensor.untyped_storage().nbytes()
        for tensor in {tensor.untyped_storage().data_ptr(): tensor for tensor in actual.values()}.values()
    )
    payload_bytes = sum(tensor.numel() * tensor.element_size() for tensor in actual.values())
    return {
        "tensor_count": len(actual),
        "payload_bytes": payload_bytes,
        "unique_storage_bytes": storage_bytes,
        "device": str(staging.device),
        "synchronized": False,
        "allocator_counters_included": False,
    }


def c4_resident_resource_r10_r11_contract_v1() -> dict[str, object]:
    """Freeze bounded resource/residency preparation without a rung claim."""

    rows = c4_resident_r9_staging_budget_rows_v1()
    return {
        "schema_version": C4_RESIDENT_RESOURCE_R10_R11_SCHEMA_VERSION_V1,
        "route": "R9_B48_C32_R10_RESOURCE_RESIDENCY_PREPARATION_ONLY",
        "decision": C4_RESIDENT_RESOURCE_R10_R11_DECISION_V1,
        "batch": 48,
        "capacity": 32,
        "grid_size": 8,
        "streams": 2,
        "staging_tensor_count": len(rows),
        "staging_payload_bytes": sum(row.bytes for row in rows),
        "staging_scope": "caller_owned_R3_plus_R4_plus_R6_status_nested_staging_only",
        "unused_nested_turnover_birth_payload_bytes": sum(
            row.bytes for row in rows if row.path.startswith("turnover.birth.")
        ),
        "required_steady_state_h2d_bytes": 0,
        "required_steady_state_d2h_bytes": 0,
        "required_steady_state_host_callbacks": 0,
        "required_steady_state_synchronizations": 0,
        "required_steady_state_materializations": 0,
        "required_steady_state_device_allocator_calls": 0,
        "required_steady_state_per_world_python_loops": 0,
        "required_steady_state_per_agent_python_loops": 0,
        "measured_steady_state_counters": None,
        "source_observed_execute_h2d_or_d2h_tokens": 0,
        "source_observed_execute_sync_or_materialization_tokens": 0,
        "source_observed_execute_device_allocator_tokens": 0,
        "source_observed_execute_per_world_python_loops": 0,
        "source_observed_execute_per_agent_python_loops": 0,
        "source_observed_host_validation_loops": 1,
        "source_observed_host_extension_load_calls": 1,
        "source_observed_host_native_invocations": 1,
        "preallocated_staging_lifetime_bound_to_production_owner": False,
        "native_kernel_launches_per_transaction": 105,
        "component_execution_classification": (
            "device_local_multi_kernel_host_invocation_graph_binding_unproven"
        ),
        "cuda_graph_capture_claim": False,
        "measurement_boundary": "CUDA_allocator_counters_and_events_outside_tick_only",
        "r10_blocker": None,
        "r10_execution_scope": "five_host_launched_R9_transactions_not_graph_bound",
        "r11_blocker": "no_composed_C1_C2_C3_C4_resident_tick_seam",
        "resident_total_vram_claim": False,
        "timing_claim": False,
        "gate_claim": False,
    }


__all__ = [
    "C4_RESIDENT_RESOURCE_R10_R11_DECISION_V1",
    "C4_RESIDENT_RESOURCE_R10_R11_SCHEMA_VERSION_V1",
    "C4ResidentStagingBudgetRowV1",
    "c4_resident_r9_staging_budget_rows_v1",
    "c4_resident_resource_r10_r11_contract_v1",
    "measure_c4_resident_r9_staging_v1",
]
