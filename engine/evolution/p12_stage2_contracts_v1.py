"""P1.2 Stage 2 substrate contracts.

These contracts freeze the next implementation target without promoting native
or CUDA structural/recurrent mutation. They are intentionally explicit about
surfaces that remain fail-closed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Final, Mapping

from engine.authoritative_state.schema import content_sha256


PUBLIC_IDENTITY_CONTRACT_ID: Final[str] = "p12-public-identity-slot-handle-v1"
CUDA_SHADOW_DISPOSITION_ID: Final[str] = "p12-cuda-agent-id-to-slot-quarantine-v1"
WDL_CAPACITY_CONTRACT_ID: Final[str] = "p12-wdl-compact-reporting-capacity-v1"
EED_CURSOR_CONTRACT_ID: Final[str] = "p12-eed-slot-token-cursor-v1"
RESPEC_CURSOR_CONTRACT_ID: Final[str] = "p12-respec-slot-token-cursor-v1"
STRUCTURAL_ORACLE_CONTRACT_ID: Final[str] = "p12-structural-ordinary-birth-reference-oracle-v1"
RECURRENT_ORACLE_CONTRACT_ID: Final[str] = "p12-recurrent-ordinary-birth-reference-oracle-v1"

DEFAULT_RESIDENT_CAPACITY: Final[int] = 256
DEFAULT_WDL_EVENT_FAMILIES_PER_SLOT: Final[int] = 4
INT64_MAX: Final[int] = (1 << 63) - 1
PUBLIC_GRID_AGENT_ID_MAX: Final[int] = (1 << 31) - 1


class P12Stage2ContractError(ValueError):
    """A P1.2 Stage 2 contract input is invalid."""


def _canonical(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["contract_sha256"] = content_sha256(payload)
    return payload


def canonical_contract_bytes_v1(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def public_identity_contract_v1() -> dict[str, Any]:
    """Return the selected public identity ABI target."""

    return _canonical(
        {
            "contract_id": PUBLIC_IDENTITY_CONTRACT_ID,
            "architecture": "resident_slot_handle_public_grid",
            "grid_storage": "int32 resident slot handle",
            "empty_value": -1,
            "occupied_handle_range": "[0, resident_slot_capacity)",
            "stable_biological_id_role": "history/reporting identity only",
            "stable_id_resolution": "slot_to_agent_id:int64 plus slot_tokens:int64",
            "mutable_state_authority": "resident slot plus slot token",
            "reporting_identity": "stable biological id emitted from sparse identity table",
            "forbidden": (
                "public mutable-state path may not index by stable biological id"
            ),
            "current_python_status": (
                "implemented in Python reference: GridWorld.agent_grid stores "
                "resident slot handles; unsupported above-int32 biological IDs "
                "still fail closed at current public API boundaries"
            ),
        }
    )


def cuda_shadow_disposition_v1() -> dict[str, Any]:
    """Return the selected disposition for legacy CUDA ``agent_id_to_slot``."""

    return _canonical(
        {
            "contract_id": CUDA_SHADOW_DISPOSITION_ID,
            "field": "agent_id_to_slot",
            "disposition": "quarantine_non_authoritative_compatibility_projection",
            "authoritative": False,
            "shape_rule": "[batch, resident_slot_capacity]",
            "value_rule": "slot-capacity compatibility shadow only",
            "high_id_lookup_allowed": False,
            "production_authority_allowed": False,
            "removal_milestone": "after public identity slot-handle ABI is route-qualified",
            "current_route_status": (
                "current flattened and WDL CUDA routes materialize guarded "
                "low-ID mirror writes; legacy/deprecated CUDA kernels still "
                "contain dense lookup reads and remain compatibility debt"
            ),
        }
    )


def cuda_shadow_route_audit_v1(repo_root: str | Path) -> dict[str, Any]:
    """Audit current and legacy CUDA ``agent_id_to_slot`` compatibility use."""

    root = Path(repo_root)
    current_routes = (
        "engine/perf/cuda_architecture_b_flattened_controller_v1_kernels.cu",
        "engine/perf/cuda_architecture_b_flattened_wdl_controller_v2_kernels.cu",
    )
    current_bindings = (
        "engine/perf/cuda_architecture_b_flattened_controller_v1_binding.cpp",
    )
    legacy_routes = (
        "engine/perf/cuda_architecture_b_phased_controller_v1_kernels.cu",
        "engine/perf/cuda_architecture_b_stackless_birth_generation_kernel.cu",
        "engine/perf/cuda_architecture_b_persistent_generation_kernel.cu",
        "engine/perf/native/cuda_architecture_b_lifecycle.cu",
    )

    direct_pattern = re.compile(r"agent_id_to_slot\s*\[[^\]\n]*\]")

    def scan(path: str) -> dict[str, Any]:
        text = (root / path).read_text(encoding="utf-8")
        direct: list[str] = []
        dense_reads: list[str] = []
        for line in text.splitlines():
            for match in direct_pattern.finditer(line):
                access = match.group(0)
                direct.append(access)
                after = line[match.end() :]
                is_write = after.lstrip().startswith("=")
                is_dense_lookup = (
                    "cell_id_map_offset + agent_id" in access
                    or access == "agent_id_to_slot[agent_id]"
                )
                if is_dense_lookup and not is_write:
                    dense_reads.append(access)
        return {
            "path": path,
            "direct_access_count": len(direct),
            "direct_accesses": tuple(direct),
            "dense_lookup_reads": tuple(dense_reads),
        }

    binding_exports = []
    for path in current_bindings:
        text = (root / path).read_text(encoding="utf-8")
        binding_exports.append(
            {
                "path": path,
                "exports_shadow_pointer": "agent_id_to_slot.data_ptr<int32_t>()"
                in text,
                "requires_capacity_shape": "slot-capacity compatibility shadow"
                in text,
            }
        )

    current = tuple(scan(path) for path in current_routes)
    legacy = tuple(scan(path) for path in legacy_routes)
    return _canonical(
        {
            "contract_id": CUDA_SHADOW_DISPOSITION_ID,
            "disposition": "quarantine_non_authoritative_compatibility_projection",
            "current_routes": current,
            "current_bindings": tuple(binding_exports),
            "legacy_routes": legacy,
            "production_shadow_exported": any(
                item["exports_shadow_pointer"] for item in binding_exports
            ),
            "current_route_dense_lookup_reads": tuple(
                item
                for item in current
                if item["dense_lookup_reads"]
            ),
            "legacy_dense_lookup_debt_files": tuple(
                item["path"] for item in legacy if item["dense_lookup_reads"]
            ),
            "high_id_lookup_allowed": False,
            "absence_changes_current_route_abi": True,
        }
    )


def wdl_capacity_projection_v1(
    *,
    cells: int,
    resident_capacity: int = DEFAULT_RESIDENT_CAPACITY,
    event_families_per_slot: int = DEFAULT_WDL_EVENT_FAMILIES_PER_SLOT,
) -> dict[str, Any]:
    if cells <= 0:
        raise P12Stage2ContractError("cells must be positive")
    if resident_capacity <= 0:
        raise P12Stage2ContractError("resident_capacity must be positive")
    if event_families_per_slot <= 0:
        raise P12Stage2ContractError("event_families_per_slot must be positive")
    unique_agent_rows = int(cells) * int(resident_capacity)
    event_rows = unique_agent_rows * int(event_families_per_slot)
    bytes_per_agent_row = 8 + 4 + 8 + (12 * 8)
    bytes_per_event_row = 8 + 8 + 8 + 4 + 4
    return {
        "cells": int(cells),
        "resident_capacity": int(resident_capacity),
        "unique_agent_rows_max": unique_agent_rows,
        "unique_lineage_rows_max": unique_agent_rows,
        "cursor_rows_max": unique_agent_rows,
        "event_rows_max_per_generation": event_rows,
        "agent_and_lineage_key_dtype": "int64",
        "cell_and_generation_key_dtype": "int64",
        "duplicate_aggregation": "stable sort by cell,generation,slot,lineage,event_family",
        "overflow_behavior": "fail_closed_not_evaluable",
        "dropped_count_semantics": "nonzero dropped count makes qualification/science not_evaluable",
        "canonical_byte_formula": (
            "cells*resident_capacity*(agent_row_bytes+lineage_row_bytes+cursor_row_bytes)"
            "+cells*resident_capacity*event_families_per_slot*event_row_bytes"
        ),
        "projected_bytes_per_generation": (
            unique_agent_rows * (bytes_per_agent_row * 3) + event_rows * bytes_per_event_row
        ),
    }


def wdl_capacity_contract_v1() -> dict[str, Any]:
    projections = {
        f"b{cells}": wdl_capacity_projection_v1(cells=cells)
        for cells in (1, 2, 12, 48, 72)
    }
    return _canonical(
        {
            "contract_id": WDL_CAPACITY_CONTRACT_ID,
            "row_families": (
                "agent_terminal",
                "lineage_terminal",
                "slot_cursor",
                "event_budget",
            ),
            "capacity_basis": "resident slot capacity, never max stable biological id",
            "checkpoint_resume": "terminal-prefix merge with canonical serialization",
            "overflow_behavior": "fail_closed_not_evaluable",
            "projections": projections,
        }
    )


def cursor_contract_v1(contract_id: str) -> dict[str, Any]:
    if contract_id not in {EED_CURSOR_CONTRACT_ID, RESPEC_CURSOR_CONTRACT_ID}:
        raise P12Stage2ContractError(f"unknown cursor contract: {contract_id}")
    return _canonical(
        {
            "contract_id": contract_id,
            "owner_key": "resident_slot_plus_slot_token",
            "creation": "initialize cursor only for current slot token",
            "reset": "clear cursor on death, rollback rejection, and turnover reset",
            "slot_reuse": "new child must use a strictly newer slot token",
            "checkpoint": "persist cursor rows by slot and validate token on restore",
            "multicell_remap": "cell-local slots; no cross-cell stable-id cursor lookup",
            "stale_token": "reject",
            "zero_initialized_memory": "not an ownership contract",
            "python_reference_fixture_status": "implemented",
            "reference_fixture_module": "engine/evolution/p12_stage2_cursor_fixtures_v1.py",
        }
    )


def dense_identity_route_envelope_v1(
    *,
    cells: int,
    active_counts: tuple[int, ...],
    max_stable_biological_id: int,
    resident_capacity: int = DEFAULT_RESIDENT_CAPACITY,
) -> dict[str, Any]:
    """Return a capacity-bounded dense-ID route envelope projection."""

    cells = int(cells)
    resident_capacity = int(resident_capacity)
    max_stable_biological_id = int(max_stable_biological_id)
    if cells <= 0:
        raise P12Stage2ContractError("cells must be positive")
    if len(active_counts) != cells:
        raise P12Stage2ContractError("active_counts length must match cells")
    if resident_capacity <= 0:
        raise P12Stage2ContractError("resident_capacity must be positive")
    if max_stable_biological_id < 0:
        raise P12Stage2ContractError("max_stable_biological_id must be nonnegative")
    if max_stable_biological_id == INT64_MAX:
        raise P12Stage2ContractError("INT64_MAX is reserved as an invalid identity sentinel")
    if max_stable_biological_id > PUBLIC_GRID_AGENT_ID_MAX:
        raise P12Stage2ContractError("stable biological id exceeds public-grid route envelope")

    active = tuple(int(count) for count in active_counts)
    for count in active:
        if count < 0 or count > resident_capacity:
            raise P12Stage2ContractError("active count exceeds resident capacity")
    total_active = sum(active)
    return _canonical(
        {
            "contract_id": "p12-dense-id-route-envelope-v1",
            "cells": cells,
            "resident_capacity": resident_capacity,
            "active_counts": active,
            "ragged_multicell": len(set(active)) > 1,
            "max_stable_biological_id": max_stable_biological_id,
            "public_grid_agent_id_max": PUBLIC_GRID_AGENT_ID_MAX,
            "slot_to_agent_id_rows": cells * resident_capacity,
            "active_identity_pair_rows": total_active,
            "agent_id_to_slot_shadow_rows": cells * resident_capacity,
            "wdl_unique_agent_rows_max": (
                wdl_capacity_projection_v1(
                    cells=cells,
                    resident_capacity=resident_capacity,
                )["unique_agent_rows_max"]
            ),
            "storage_scales_with": "cells*resident_capacity plus live active rows",
            "does_not_scale_with": "max(stable_biological_id)+1",
            "int64_max_behavior": "fail_closed_reserved_sentinel",
        }
    )


def validate_cursor_token_v1(*, current_token: int, supplied_token: int) -> None:
    if int(current_token) <= 0:
        raise P12Stage2ContractError("current_token must be positive")
    if int(supplied_token) != int(current_token):
        raise P12Stage2ContractError("stale slot-token cursor write rejected")


def mutation_oracle_contracts_v1() -> dict[str, Any]:
    return _canonical(
        {
            "structural_contract_id": STRUCTURAL_ORACLE_CONTRACT_ID,
            "recurrent_contract_id": RECURRENT_ORACLE_CONTRACT_ID,
            "lifecycle": "Genome V2 ordinary live birth only",
            "structural_status": "python reference oracle implemented; native/cuda open",
            "recurrent_status": "python reference oracle implemented; native/cuda open",
            "reference_oracle_module": "engine/evolution/p12_stage2_mutation_oracles_v1.py",
            "rng_budget_required": True,
            "rollback_required": True,
            "checkpoint_required": True,
            "native_cuda_promotion_authorized": False,
            "hidden_width_resize_allowed": False,
        }
    )
