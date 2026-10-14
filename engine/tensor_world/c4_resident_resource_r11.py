"""Current-source resource inventory for the composed R11 graph owner.

This is a bounded accounting contract, not a residency or performance gate.
It deliberately inventories the R11 wrapper separately from the historical
R9/R10 staging receipt so typed role streams and device run-control state
cannot disappear from the resource ledger.
"""

from __future__ import annotations

from typing import Final

import torch

from .c4_resident_resource_r10_r11 import (
    C4ResidentStagingBudgetRowV1,
    c4_resident_r9_staging_budget_rows_v1,
)
from .c2_state import C2_POLICY_PACKED_WIDTH_V1
from .contracts import TensorWorldContractError
from .state import PATCH_POLICY_PACKED_WIDTH_V1


C4_RESIDENT_RESOURCE_R11_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_resource_r11_graph_v1"
)
C4_RESIDENT_RESOURCE_R11_DECISION_V1: Final[str] = "NO_GATE_DECISION"
C4_RESIDENT_RESOURCE_R11_BATCH_V1: Final[int] = 48
C4_RESIDENT_RESOURCE_R11_CAPACITY_V1: Final[int] = 512
C4_RESIDENT_RESOURCE_R11_STREAMS_V1: Final[int] = 5
C4_RESIDENT_RESOURCE_R11_WINDOW_TICKS_V1: Final[int] = 5
_RUN_CONTROL_FIELDS: Final[tuple[tuple[str, torch.dtype], ...]] = (
    ("c3_status", torch.int32),
    ("c4_status", torch.int32),
    ("commit_status", torch.int32),
    ("generation", torch.int64),
    ("tick", torch.int64),
    ("shadow_event_sequence", torch.int64),
    ("requested_ticks", torch.int64),
    ("completed_ticks", torch.int64),
    ("current_tick", torch.int64),
    ("global_status", torch.int32),
    ("first_error_stage", torch.int32),
    ("first_error_world", torch.int32),
    ("first_error_code", torch.int32),
    ("first_error_tick", torch.int64),
    ("stop_requested", torch.int32),
    ("receipt_write_index", torch.int64),
)
_IDENTITY_STAGING_FIELDS: Final[tuple[tuple[str, tuple[str, ...], torch.dtype], ...]] = (
    ("death_slot_tokens", ("slots",), torch.int64),
    ("parent_slot_tokens", ("slots",), torch.int64),
    ("mate_slot_tokens", ("slots",), torch.int64),
    ("child_slot_tokens", ("slots",), torch.int64),
    ("next_agent_id_before", ("batch",), torch.int64),
    ("status", ("batch",), torch.int32),
)


def c4_resident_composed_r11_budget_rows_v1(
    *,
    batch: int = C4_RESIDENT_RESOURCE_R11_BATCH_V1,
    capacity: int = C4_RESIDENT_RESOURCE_R11_CAPACITY_V1,
    grid_size: int = 32,
    streams: int = C4_RESIDENT_RESOURCE_R11_STREAMS_V1,
) -> tuple[C4ResidentStagingBudgetRowV1, ...]:
    """Return the fixed R11 transaction plus device run-control inventory."""

    if (batch, capacity, streams) != (
        C4_RESIDENT_RESOURCE_R11_BATCH_V1,
        C4_RESIDENT_RESOURCE_R11_CAPACITY_V1,
        C4_RESIDENT_RESOURCE_R11_STREAMS_V1,
    ) or grid_size <= 0:
        raise TensorWorldContractError(
            "R11 production resource inventory is bounded to B48/W32/C512/S5"
        )
    rows = list(
        c4_resident_r9_staging_budget_rows_v1(
            batch=batch, capacity=capacity, grid_size=grid_size, streams=streams
        )
    )
    for name, shape_kind, dtype in _IDENTITY_STAGING_FIELDS:
        shape = (batch, capacity) if shape_kind == ("slots",) else (batch,)
        rows.append(C4ResidentStagingBudgetRowV1(f"identity.{name}", shape, dtype))
    for name, dtype in _RUN_CONTROL_FIELDS:
        rows.append(C4ResidentStagingBudgetRowV1(f"run_control.{name}", (batch,), dtype))
    return tuple(rows)


def c4_resident_composed_r11_resource_contract_v1() -> dict[str, object]:
    rows = c4_resident_composed_r11_budget_rows_v1()
    payload = sum(row.bytes for row in rows)
    r9_rows = c4_resident_r9_staging_budget_rows_v1(
        capacity=C4_RESIDENT_RESOURCE_R11_CAPACITY_V1,
        grid_size=32,
        streams=5,
    )
    identity_payload = sum(
        row.bytes for row in rows if row.path.startswith("identity.")
    )
    guarded_policy_bytes = (
        C4_RESIDENT_RESOURCE_R11_BATCH_V1
        * C4_RESIDENT_RESOURCE_R11_CAPACITY_V1
        * (PATCH_POLICY_PACKED_WIDTH_V1 + C2_POLICY_PACKED_WIDTH_V1)
        * torch.tensor([], dtype=torch.float32).element_size()
    )
    return {
        "schema_version": C4_RESIDENT_RESOURCE_R11_SCHEMA_VERSION_V1,
        "decision": C4_RESIDENT_RESOURCE_R11_DECISION_V1,
        "route": "R11_B48_W32_P256_C512_FIXED_TOPOLOGY_GRAPH_OWNER",
        "batch": C4_RESIDENT_RESOURCE_R11_BATCH_V1,
        "capacity": C4_RESIDENT_RESOURCE_R11_CAPACITY_V1,
        "streams": C4_RESIDENT_RESOURCE_R11_STREAMS_V1,
        "graph_window_ticks": C4_RESIDENT_RESOURCE_R11_WINDOW_TICKS_V1,
        "staging_tensor_count": len(rows),
        "staging_payload_bytes": payload,
        "r9_typed_staging_payload_bytes": sum(row.bytes for row in r9_rows),
        "run_control_tensor_count": len(_RUN_CONTROL_FIELDS),
        "identity_staging_tensor_count": len(_IDENTITY_STAGING_FIELDS),
        "identity_staging_payload_bytes": identity_payload,
        "run_control_payload_bytes": (
            payload - sum(row.bytes for row in r9_rows) - identity_payload
        ),
        "required_steady_state_h2d_bytes": 0,
        "required_steady_state_d2h_bytes": 0,
        "required_steady_state_host_callbacks": 0,
        "required_steady_state_synchronizations": 0,
        "required_steady_state_materializations": 0,
        "required_steady_state_device_allocator_calls": 0,
        "required_steady_state_per_world_python_loops": 0,
        "required_steady_state_per_agent_python_loops": 0,
        "full_policy_shadow_payload_bytes": 0,
        "guarded_shared_policy_payload_bytes": guarded_policy_bytes,
        "guarded_shared_policy_paths": [
            "base.base.core.policy.packed_weights",
            "base.base.policy.packed_weights",
        ],
        "graph_node_count": None,
        "allocator_counters": None,
        "target_gpu_trace": None,
        "component_execution_classification": (
            "fixed_topology_graph_window_device_time_unmeasured"
        ),
        "resident_total_vram_claim": False,
        "timing_claim": False,
        "gate_claim": False,
    }


__all__ = [
    "C4_RESIDENT_RESOURCE_R11_BATCH_V1",
    "C4_RESIDENT_RESOURCE_R11_CAPACITY_V1",
    "C4_RESIDENT_RESOURCE_R11_DECISION_V1",
    "C4_RESIDENT_RESOURCE_R11_SCHEMA_VERSION_V1",
    "C4_RESIDENT_RESOURCE_R11_STREAMS_V1",
    "C4_RESIDENT_RESOURCE_R11_WINDOW_TICKS_V1",
    "c4_resident_composed_r11_budget_rows_v1",
    "c4_resident_composed_r11_resource_contract_v1",
]
