from __future__ import annotations

"""CXXBACKBONE-1 resident state v4 and all-surface authority ledger.

This module is intentionally stricter than the older CXXWAVE1G global gate. It
tracks every required native authority surface, whether that surface is active
for a requested envelope, and whether it is actually native-authoritative.
"""

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from perf.cxxwave1h_capability_live_backbone import (
    capability_graph_payload as cxxwave1h_capability_graph_payload,
    resolve_capabilities_v2,
)


SCHEMA_VERSION = "cxxbackbone1_native_authority_v1"
RESIDENT_STATE_SCHEMA_VERSION = "resident_runtime_state_v4"
CAPABILITY_GRAPH_SCHEMA_VERSION = "cpp_backbone_capability_graph_v3"
SURFACE_LEDGER_SCHEMA_VERSION = "cxxbackbone1_all_19_surface_ledger_v1"


SURFACE_NAMES: tuple[tuple[int, str], ...] = (
    (1, "action_cache"),
    (2, "capability_support_mask"),
    (3, "execution_cell_metadata"),
    (4, "hmem"),
    (5, "jungle_state"),
    (6, "lifecycle_reproduction_mutation_state"),
    (7, "lineage_founder_state"),
    (8, "mlp_gru_policy_banks"),
    (9, "observations_staging"),
    (10, "patch_world_state"),
    (11, "policy_row_mappings"),
    (12, "population_slots_stable_ids"),
    (13, "rng_state_indices"),
    (14, "wdl_state"),
    (15, "public_output_compatibility"),
    (16, "full_resident_checkpoint"),
    (17, "live_lifecycle_execution"),
    (18, "mutation_rng_draw_schedule"),
    (19, "multi_cell_batching"),
)


@dataclass(frozen=True)
class NativeAuthorityFieldV4:
    group: str
    field: str
    authority: str
    dtype: str
    device: str
    shape: tuple[int, ...]
    ordering: str
    mutation_owner: str
    checkpoint_member: bool
    rebuild_behavior: str
    parity_comparison: str
    capability_dependencies: tuple[str, ...]


@dataclass(frozen=True)
class NativeSurfaceStatusV1:
    surface_id: int
    surface_name: str
    active: bool
    native_authoritative: bool
    implemented_contract: str
    integrated_live: bool
    unit_parity: str
    generation_parity: str
    checkpoint_parity: str
    public_output_parity: str
    blocker: str
    evidence: str


@dataclass(frozen=True)
class ResidentRuntimeStateV4:
    schema_version: str
    generation: int
    tick: int
    cell_id: str
    execution: Mapping[str, Any]
    capabilities: Mapping[str, Any]
    population: Mapping[str, np.ndarray]
    policy: Mapping[str, np.ndarray]
    hmem: Mapping[str, np.ndarray]
    action_cache: Mapping[str, np.ndarray]
    observations: Mapping[str, np.ndarray]
    scalar_state: Mapping[str, np.ndarray]
    chemistry_maintenance: Mapping[str, np.ndarray]
    patch: Mapping[str, np.ndarray]
    jungle: Mapping[str, np.ndarray]
    wdl: Mapping[str, np.ndarray]
    lifecycle: Mapping[str, np.ndarray]
    mutation_recombination: Mapping[str, np.ndarray]
    lineage_founder: Mapping[str, np.ndarray]
    events: Mapping[str, np.ndarray]
    checkpoint: Mapping[str, Any]
    rng: Mapping[str, Any]
    batch_metadata: Mapping[str, Any]
    errors_status: Mapping[str, Any]
    field_authority: tuple[NativeAuthorityFieldV4, ...]
    surface_status: tuple[NativeSurfaceStatusV1, ...]
    state_digest: str


def cxxbackbone1_surface_requirements() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "surface_id": sid,
            "surface_name": name,
            "required_for_full_single_cell": sid != 19,
            "required_for_full_mission": True,
            "native_authority_required": True,
        }
        for sid, name in SURFACE_NAMES
    )


def build_surface_ledger(
    *,
    active_features: Sequence[str],
    implemented_surfaces: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[NativeSurfaceStatusV1, ...]:
    active = set(str(x) for x in active_features)
    implemented = {str(k): dict(v) for k, v in (implemented_surfaces or {}).items()}
    rows: list[NativeSurfaceStatusV1] = []
    for sid, name in SURFACE_NAMES:
        impl = implemented.get(name, {})
        is_active = _surface_active(name, active)
        native = bool(impl.get("native_authoritative", False))
        blocker = "" if native or not is_active else str(impl.get("blocker") or f"{name}:native_authority_missing")
        rows.append(
            NativeSurfaceStatusV1(
                surface_id=sid,
                surface_name=name,
                active=is_active,
                native_authoritative=native,
                implemented_contract=str(impl.get("implemented_contract", "")),
                integrated_live=bool(impl.get("integrated_live", False)),
                unit_parity=str(impl.get("unit_parity", "not_run")),
                generation_parity=str(impl.get("generation_parity", "not_run")),
                checkpoint_parity=str(impl.get("checkpoint_parity", "not_applicable" if sid not in {1, 4, 6, 7, 8, 10, 13, 14, 15, 16, 18, 19} else "not_run")),
                public_output_parity=str(impl.get("public_output_parity", "not_applicable" if sid != 15 else "not_run")),
                blocker=blocker,
                evidence=str(impl.get("evidence", "")),
            )
        )
    return tuple(rows)


def cxx_backbone_capability_graph_v3_payload(
    *,
    active_features: Sequence[str],
    implemented_surfaces: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    ledger = build_surface_ledger(active_features=active_features, implemented_surfaces=implemented_surfaces)
    flat_t0 = resolve_capabilities_v2(
        requested_tier="T0",
        active_features=("flat_world", "mlp_only", "deterministic_actions", "no_hmem", "no_checkpoint", "no_public_report"),
        native_state=(
            "execution_cell_metadata",
            "population_slots_stable_ids",
            "mlp_policy_banks",
            "policy_row_mappings",
            "flat_observations_staging",
            "action_selection",
            "action_cache_cursor",
            "flat_environment_state",
            "chemistry_scalar_state",
            "minimal_summary",
        ),
        native_kernels=("cxxwave1h_flat_mlp_generation_v1",),
    )
    return {
        "schema_version": CAPABILITY_GRAPH_SCHEMA_VERSION,
        "surface_requirements": list(cxxbackbone1_surface_requirements()),
        "surface_ledger": [row.__dict__ for row in ledger],
        "all_active_surfaces_native": all(row.native_authoritative for row in ledger if row.active),
        "native_authoritative_count": sum(1 for row in ledger if row.native_authoritative),
        "active_surface_count": sum(1 for row in ledger if row.active),
        "blocking_surface_count": sum(1 for row in ledger if row.active and row.blocker),
        "cxxwave1h_t0_capability_graph": cxxwave1h_capability_graph_payload(flat_t0),
    }


def resident_runtime_state_v4(
    *,
    generation: int,
    tick: int,
    cell_id: str,
    active_features: Sequence[str],
    arrays: Mapping[str, Any] | None = None,
    implemented_surfaces: Mapping[str, Mapping[str, Any]] | None = None,
) -> ResidentRuntimeStateV4:
    arr = {str(k): np.ascontiguousarray(v) for k, v in (arrays or {}).items()}
    ledger = build_surface_ledger(active_features=active_features, implemented_surfaces=implemented_surfaces)
    capabilities = cxx_backbone_capability_graph_v3_payload(
        active_features=active_features,
        implemented_surfaces=implemented_surfaces,
    )
    groups = _group_arrays(arr)
    authority = tuple(_field_authority_rows(groups=groups, ledger=ledger))
    payload = {
        "schema_version": RESIDENT_STATE_SCHEMA_VERSION,
        "generation": int(generation),
        "tick": int(tick),
        "cell_id": str(cell_id),
        "arrays": {name: _array_digest(value) for name, value in sorted(arr.items())},
        "capabilities": _json_digest(capabilities),
        "surfaces": [row.__dict__ for row in ledger],
    }
    return ResidentRuntimeStateV4(
        schema_version=RESIDENT_STATE_SCHEMA_VERSION,
        generation=int(generation),
        tick=int(tick),
        cell_id=str(cell_id),
        execution={"generation": int(generation), "tick": int(tick), "cell_id": str(cell_id)},
        capabilities=capabilities,
        population=groups["population"],
        policy=groups["policy"],
        hmem=groups["hmem"],
        action_cache=groups["action_cache"],
        observations=groups["observations"],
        scalar_state=groups["scalar_state"],
        chemistry_maintenance=groups["chemistry_maintenance"],
        patch=groups["patch"],
        jungle=groups["jungle"],
        wdl=groups["wdl"],
        lifecycle=groups["lifecycle"],
        mutation_recombination=groups["mutation_recombination"],
        lineage_founder=groups["lineage_founder"],
        events=groups["events"],
        checkpoint={"checkpoint_ready": False, "reason": "full resident checkpoint surface not native-authoritative"},
        rng={"rng_native_authority": False, "reason": "rng_state_indices surface not native-authoritative"},
        batch_metadata={"multi_cell_native_authority": False},
        errors_status={"hidden_fallback": False},
        field_authority=authority,
        surface_status=ledger,
        state_digest=_json_digest(payload),
    )


def resident_runtime_state_v4_payload(state: ResidentRuntimeStateV4) -> dict[str, Any]:
    return {
        "schema_version": state.schema_version,
        "generation": int(state.generation),
        "tick": int(state.tick),
        "cell_id": state.cell_id,
        "surface_status": [row.__dict__ for row in state.surface_status],
        "field_authority": [row.__dict__ for row in state.field_authority],
        "capabilities": state.capabilities,
        "state_digest": state.state_digest,
    }


def _surface_active(surface_name: str, active: set[str]) -> bool:
    optional_map = {
        "jungle_state": "jungle",
        "patch_world_state": "patch_world",
        "wdl_state": "wdl",
        "public_output_compatibility": "public_report",
        "multi_cell_batching": "multi_cell",
    }
    feature = optional_map.get(surface_name)
    if feature is None:
        return True
    return feature in active or "full_scientific" in active


def _group_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    groups = {
        "population": {},
        "policy": {},
        "hmem": {},
        "action_cache": {},
        "observations": {},
        "scalar_state": {},
        "chemistry_maintenance": {},
        "patch": {},
        "jungle": {},
        "wdl": {},
        "lifecycle": {},
        "mutation_recombination": {},
        "lineage_founder": {},
        "events": {},
    }
    prefixes = {
        "agent_": "population",
        "slot_": "population",
        "policy_": "policy",
        "mlp_": "policy",
        "gru_": "policy",
        "hmem": "hmem",
        "action_cache": "action_cache",
        "obs": "observations",
        "scalar_": "scalar_state",
        "chemistry_": "chemistry_maintenance",
        "maintenance_": "chemistry_maintenance",
        "patch_": "patch",
        "jungle_": "jungle",
        "wdl_": "wdl",
        "lifecycle_": "lifecycle",
        "mutation_": "mutation_recombination",
        "recombination_": "mutation_recombination",
        "lineage_": "lineage_founder",
        "founder_": "lineage_founder",
        "event_": "events",
    }
    for name, value in arrays.items():
        group = "scalar_state"
        for prefix, candidate in prefixes.items():
            if name.startswith(prefix):
                group = candidate
                break
        groups[group][name] = value
    return groups


def _field_authority_rows(
    *,
    groups: Mapping[str, Mapping[str, np.ndarray]],
    ledger: Sequence[NativeSurfaceStatusV1],
) -> list[NativeAuthorityFieldV4]:
    surface_by_group = {
        "population": "population_slots_stable_ids",
        "policy": "mlp_gru_policy_banks",
        "hmem": "hmem",
        "action_cache": "action_cache",
        "observations": "observations_staging",
        "patch": "patch_world_state",
        "jungle": "jungle_state",
        "wdl": "wdl_state",
        "lifecycle": "lifecycle_reproduction_mutation_state",
        "mutation_recombination": "mutation_rng_draw_schedule",
        "lineage_founder": "lineage_founder_state",
        "events": "public_output_compatibility",
    }
    native_by_surface = {row.surface_name: bool(row.native_authoritative) for row in ledger}
    rows: list[NativeAuthorityFieldV4] = []
    for group, values in groups.items():
        surface = surface_by_group.get(group, "execution_cell_metadata")
        native = native_by_surface.get(surface, False)
        if not values:
            rows.append(
                NativeAuthorityFieldV4(
                    group=group,
                    field="__empty__",
                    authority="cpp_backbone" if native else "python_live_runner",
                    dtype="none",
                    device="cpu",
                    shape=(),
                    ordering="schema_order",
                    mutation_owner="cpp_backbone" if native else "python_live_runner",
                    checkpoint_member=surface in {
                        "action_cache",
                        "hmem",
                        "patch_world_state",
                        "jungle_state",
                        "wdl_state",
                        "lineage_founder_state",
                        "full_resident_checkpoint",
                    },
                    rebuild_behavior="native_rebuild" if native else "python_object_rebuild_required",
                    parity_comparison="exact_digest",
                    capability_dependencies=(surface,),
                )
            )
            continue
        for field, value in sorted(values.items()):
            rows.append(
                NativeAuthorityFieldV4(
                    group=group,
                    field=field,
                    authority="cpp_backbone" if native else "python_live_runner",
                    dtype=str(value.dtype),
                    device="cpu",
                    shape=tuple(int(x) for x in value.shape),
                    ordering="stable_slot_order" if group in {"population", "policy", "hmem", "action_cache"} else "schema_order",
                    mutation_owner="cpp_backbone" if native else "python_live_runner",
                    checkpoint_member=True,
                    rebuild_behavior="native_rebuild" if native else "python_object_rebuild_required",
                    parity_comparison="exact_digest",
                    capability_dependencies=(surface,),
                )
            )
    return rows


def _array_digest(value: Any) -> str:
    arr = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(json.dumps(list(arr.shape), separators=(",", ":")).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _json_digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


__all__ = [
    "CAPABILITY_GRAPH_SCHEMA_VERSION",
    "RESIDENT_STATE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SURFACE_LEDGER_SCHEMA_VERSION",
    "SURFACE_NAMES",
    "NativeAuthorityFieldV4",
    "NativeSurfaceStatusV1",
    "ResidentRuntimeStateV4",
    "build_surface_ledger",
    "cxx_backbone_capability_graph_v3_payload",
    "cxxbackbone1_surface_requirements",
    "resident_runtime_state_v4",
    "resident_runtime_state_v4_payload",
]
