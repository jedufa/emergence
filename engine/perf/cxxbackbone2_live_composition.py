from __future__ import annotations

"""CXXBACKBONE-2 live all-surface composition map and state-v5 gate."""

from dataclasses import dataclass
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from perf.cxxbackbone1_native_authority import SURFACE_NAMES


SCHEMA_VERSION = "cxxbackbone2_live_composition_v1"
COMPOSITION_MAP_SCHEMA_VERSION = "cxxbackbone2_live_composition_map_v1"
RESIDENT_STATE_SCHEMA_VERSION = "resident_runtime_state_v5"
EXECUTION_RESULT_SCHEMA_VERSION = "cxxbackbone2_execute_generation_result_v1"
SUMMARY_SCHEMA_VERSION = "cxxbackbone2_live_composition_summary_v1"


LIVE_INTEGRATED_FROM_CXXBACKBONE1 = {
    "execution_cell_metadata",
    "observations_staging",
    "population_slots_stable_ids",
}

CONTRACT_SOURCE_BY_SURFACE = {
    "action_cache": "cxxbackbone1_policy_authority_v1",
    "capability_support_mask": "cxxbackbone1_capability_graph_v3",
    "execution_cell_metadata": "cxxwave1h_flat_mlp_generation_v1",
    "hmem": "cxxbackbone1_policy_authority_v1",
    "jungle_state": "cxxbackbone1_environment_surfaces_v1",
    "lifecycle_reproduction_mutation_state": "cxxbackbone1_integrated_surfaces_v1",
    "lineage_founder_state": "cxxbackbone1_integrated_surfaces_v1",
    "mlp_gru_policy_banks": "cxxbackbone1_policy_authority_v1",
    "observations_staging": "cxxwave1h_flat_mlp_generation_v1",
    "patch_world_state": "cxxbackbone1_environment_surfaces_v1",
    "policy_row_mappings": "cxxbackbone1_policy_authority_v1",
    "population_slots_stable_ids": "cxxwave1h_flat_mlp_generation_v1",
    "rng_state_indices": "cxxbackbone1_rng_schedule_v1",
    "wdl_state": "cxxbackbone1_environment_surfaces_v1",
    "public_output_compatibility": "cxxbackbone1_integrated_surfaces_v1",
    "full_resident_checkpoint": "cxxbackbone1_integrated_surfaces_v1",
    "live_lifecycle_execution": "cxxbackbone1_integrated_surfaces_v1",
    "mutation_rng_draw_schedule": "cxxbackbone1_rng_schedule_v1",
    "multi_cell_batching": "cxxbackbone1_integrated_surfaces_v1",
}

PYTHON_OWNER_BY_SURFACE = {
    "action_cache": "PolicyMixin._batched_actions_strided and runnercore.action_cache",
    "capability_support_mask": "runnercore profile/gate predicates",
    "execution_cell_metadata": "runner tick loop and run configuration",
    "hmem": "torch _Hmem tensor writeback in policy path",
    "jungle_state": "runner tick jungle pressure and env patch/jungle methods",
    "lifecycle_reproduction_mutation_state": "runner_ticks_reproduction._handle_births",
    "lineage_founder_state": "population/genome/lineage Python objects and funnel writers",
    "mlp_gru_policy_banks": "torch MLP/GRU bank tensors plus sync helpers",
    "observations_staging": "PolicyMixin observation construction",
    "patch_world_state": "env.patch_world_step and env apply methods",
    "policy_row_mappings": "Agent.slot/bank token mapping and sync helpers",
    "population_slots_stable_ids": "Population slot pool plus env agent arrays",
    "rng_state_indices": "rng_manager stream state",
    "wdl_state": "env delayed hazard arrays and runner WDL context recorders",
    "public_output_compatibility": "metrics/report writers and JSON/CSV Python serializers",
    "full_resident_checkpoint": "checkpoint_bundle_v2 Python pack/apply/rebuild",
    "live_lifecycle_execution": "runner live tick lifecycle Python callbacks",
    "mutation_rng_draw_schedule": "mutation/recombination Python RNG draws",
    "multi_cell_batching": "run_registry_parallel process-level orchestration",
}


@dataclass(frozen=True)
class LiveSurfaceCompositionV1:
    surface_id: int
    surface_name: str
    contract_source: str
    current_live_integration_point: str
    python_owner_to_remove: str
    cpp_owner_to_activate: str
    generation_call_order: int
    checkpoint_order: int
    public_output_dependency: bool
    parity_fixtures: tuple[str, ...]
    performance_counters: tuple[str, ...]
    implementation_commit: str
    live_integrated: bool
    blocker: str


@dataclass(frozen=True)
class ResidentRuntimeStateV5:
    schema_version: str
    generation: int
    tick: int
    cell_id: str
    arrays: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]
    composition_map_digest: str
    live_integrated_surfaces: tuple[str, ...]
    blocked_surfaces: tuple[str, ...]
    state_digest: str


def build_live_composition_map(*, implementation_commit: str = "") -> tuple[LiveSurfaceCompositionV1, ...]:
    rows: list[LiveSurfaceCompositionV1] = []
    for order, (surface_id, surface_name) in enumerate(SURFACE_NAMES, start=1):
        live = surface_name in LIVE_INTEGRATED_FROM_CXXBACKBONE1
        rows.append(
            LiveSurfaceCompositionV1(
                surface_id=int(surface_id),
                surface_name=str(surface_name),
                contract_source=CONTRACT_SOURCE_BY_SURFACE.get(surface_name, ""),
                current_live_integration_point=_live_point(surface_name),
                python_owner_to_remove=PYTHON_OWNER_BY_SURFACE.get(surface_name, "unknown"),
                cpp_owner_to_activate=_cpp_owner(surface_name),
                generation_call_order=order,
                checkpoint_order=order if surface_name in _CHECKPOINT_SURFACES else 0,
                public_output_dependency=surface_name in _PUBLIC_OUTPUT_SURFACES,
                parity_fixtures=_parity_fixtures(surface_name),
                performance_counters=_performance_counters(surface_name),
                implementation_commit=str(implementation_commit),
                live_integrated=bool(live),
                blocker="" if live else f"{surface_name}:exact_contract_exists_but_live_runner_still_python_authoritative",
            )
        )
    return tuple(rows)


def composition_map_payload(rows: Sequence[LiveSurfaceCompositionV1]) -> dict[str, Any]:
    return {
        "schema_version": COMPOSITION_MAP_SCHEMA_VERSION,
        "surface_count": int(len(rows)),
        "live_integrated_count": sum(1 for row in rows if row.live_integrated),
        "blocked_count": sum(1 for row in rows if row.blocker),
        "surfaces": [row.__dict__ for row in rows],
    }


def capture_resident_runtime_state_v5(
    runner: Any,
    *,
    generation: int,
    tick: int = 0,
    cell_id: str = "cell0",
    implementation_commit: str = "",
) -> ResidentRuntimeStateV5:
    rows = build_live_composition_map(implementation_commit=implementation_commit)
    arrays = _runner_arrays(runner)
    metadata = {
        "run_dir": str(getattr(runner, "run_dir", "") or ""),
        "steps_per_generation": int(getattr(getattr(runner, "sim_cfg", None), "steps_per_generation", 0) or 0),
        "population_cap": int(getattr(getattr(runner, "sim_cfg", None), "population_cap", 0) or 0),
        "performance_profile_name": str(getattr(getattr(runner, "sim_cfg", None), "performance_profile_name", "") or ""),
        "hidden_fallback": False,
    }
    live = tuple(row.surface_name for row in rows if row.live_integrated)
    blocked = tuple(row.surface_name for row in rows if row.blocker)
    map_digest = _digest_json(composition_map_payload(rows))
    payload = {
        "schema_version": RESIDENT_STATE_SCHEMA_VERSION,
        "generation": int(generation),
        "tick": int(tick),
        "cell_id": str(cell_id),
        "arrays": {name: _array_digest(value) for name, value in sorted(arrays.items())},
        "metadata": metadata,
        "composition_map_digest": map_digest,
        "live": live,
        "blocked": blocked,
    }
    return ResidentRuntimeStateV5(
        schema_version=RESIDENT_STATE_SCHEMA_VERSION,
        generation=int(generation),
        tick=int(tick),
        cell_id=str(cell_id),
        arrays=arrays,
        metadata=metadata,
        composition_map_digest=map_digest,
        live_integrated_surfaces=live,
        blocked_surfaces=blocked,
        state_digest=_digest_json(payload),
    )


def resident_runtime_state_v5_payload(state: ResidentRuntimeStateV5) -> dict[str, Any]:
    return {
        "schema_version": state.schema_version,
        "generation": int(state.generation),
        "tick": int(state.tick),
        "cell_id": state.cell_id,
        "arrays": {name: _array_payload(value) for name, value in sorted(state.arrays.items())},
        "metadata": dict(state.metadata),
        "composition_map_digest": state.composition_map_digest,
        "live_integrated_surfaces": list(state.live_integrated_surfaces),
        "blocked_surfaces": list(state.blocked_surfaces),
        "state_digest": state.state_digest,
    }


def execute_generation(
    runner: Any,
    *,
    generation: int,
    buckets: dict[str, float] | None = None,
    implementation_commit: str = "",
) -> dict[str, Any]:
    started = time.perf_counter()
    state = capture_resident_runtime_state_v5(
        runner,
        generation=int(generation),
        tick=0,
        cell_id=str(getattr(getattr(runner, "sim_cfg", None), "campaign_cell_id", "") or f"generation_{int(generation)}"),
        implementation_commit=implementation_commit,
    )
    rows = build_live_composition_map(implementation_commit=implementation_commit)
    unsupported = [row.blocker for row in rows if row.blocker]
    group_status = composition_group_status(runner, rows=rows)
    result = {
        "schema_version": EXECUTION_RESULT_SCHEMA_VERSION,
        "generation": int(generation),
        "executed": False,
        "hidden_fallback": False,
        "native_call_count": 0,
        "python_callback_count": 0,
        "live_integrated_surfaces": list(state.live_integrated_surfaces),
        "blocked_surfaces": list(state.blocked_surfaces),
        "blockers": unsupported,
        "decision": (
            "CXXBACKBONE2-D/MULTIPLE-LIVE-GROUPS-COMPOSITION-BLOCKED"
            if state.live_integrated_surfaces
            else "CXXBACKBONE2-E/LIVE-PARITY-OR-CHECKPOINT-BLOCKED"
        ),
        "resident_state": resident_runtime_state_v5_payload(state),
        "composition_map": composition_map_payload(rows),
        "composition_group_status": group_status,
        "elapsed_s": float(time.perf_counter() - started),
    }
    if buckets is not None:
        buckets["cxxbackbone2_live_composition_gate"] = float(
            buckets.get("cxxbackbone2_live_composition_gate", 0.0) + float(result["elapsed_s"])
        )
    return result


def summary_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": result.get("decision"),
        "generation": int(result.get("generation", 0) or 0),
        "executed": bool(result.get("executed", False)),
        "hidden_fallback": bool(result.get("hidden_fallback", False)),
        "native_call_count": int(result.get("native_call_count", 0) or 0),
        "python_callback_count": int(result.get("python_callback_count", 0) or 0),
        "live_integrated_surfaces": list(result.get("live_integrated_surfaces", []) or []),
        "blocked_surfaces": list(result.get("blocked_surfaces", []) or []),
        "blockers": list(result.get("blockers", []) or []),
        "composition_group_status": dict(result.get("composition_group_status", {}) or {}),
        "elapsed_s": float(result.get("elapsed_s", 0.0) or 0.0),
    }


def composition_group_status(runner: Any, *, rows: Sequence[LiveSurfaceCompositionV1]) -> dict[str, Any]:
    by_name = {row.surface_name: row for row in rows}
    group_a_surfaces = (
        "action_cache",
        "hmem",
        "mlp_gru_policy_banks",
        "observations_staging",
        "policy_row_mappings",
        "rng_state_indices",
    )
    return {
        "schema_version": "cxxbackbone2_composition_group_status_v1",
        "group_a_policy_core": _group_a_policy_core_status(runner, by_name=by_name, surfaces=group_a_surfaces),
        "group_b_environment_core": _generic_group_status(by_name, ("execution_cell_metadata", "patch_world_state", "wdl_state", "jungle_state", "population_slots_stable_ids")),
        "group_c_lifecycle_genetics": _generic_group_status(by_name, ("lifecycle_reproduction_mutation_state", "live_lifecycle_execution", "mutation_rng_draw_schedule", "lineage_founder_state")),
        "group_d_events_public_output": _generic_group_status(by_name, ("public_output_compatibility",)),
        "group_e_checkpoint_authority": _generic_group_status(by_name, ("full_resident_checkpoint",)),
        "group_f_capability_execution_control": _generic_group_status(by_name, ("capability_support_mask", "execution_cell_metadata")),
        "group_g_multi_cell": _generic_group_status(by_name, ("multi_cell_batching",)),
    }


def write_summary(run_dir: str | Path, filename: str, result: Mapping[str, Any]) -> Path:
    out = Path(run_dir) / "metrics" / str(filename or "cxxbackbone2_live_composition_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(summary_payload(result), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(out)
    return out


def write_composition_map(path: str | Path, *, implementation_commit: str = "") -> Path:
    rows = build_live_composition_map(implementation_commit=implementation_commit)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(composition_map_payload(rows), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


_CHECKPOINT_SURFACES = {
    "action_cache",
    "hmem",
    "jungle_state",
    "lifecycle_reproduction_mutation_state",
    "lineage_founder_state",
    "mlp_gru_policy_banks",
    "patch_world_state",
    "rng_state_indices",
    "wdl_state",
    "public_output_compatibility",
    "full_resident_checkpoint",
    "live_lifecycle_execution",
    "mutation_rng_draw_schedule",
    "multi_cell_batching",
}

_PUBLIC_OUTPUT_SURFACES = {
    "jungle_state",
    "lifecycle_reproduction_mutation_state",
    "lineage_founder_state",
    "patch_world_state",
    "wdl_state",
    "public_output_compatibility",
    "live_lifecycle_execution",
}


def _live_point(surface_name: str) -> str:
    if surface_name in LIVE_INTEGRATED_FROM_CXXBACKBONE1:
        return "CXXWAVE1H T0 flat-generation native lane"
    return "not live-integrated; exact contract only"


def _cpp_owner(surface_name: str) -> str:
    if surface_name in {"action_cache", "hmem", "mlp_gru_policy_banks", "policy_row_mappings", "rng_state_indices"}:
        return "resident policy core native generation call"
    if surface_name in {"patch_world_state", "jungle_state", "wdl_state"}:
        return "resident environment native generation call"
    if surface_name in {"lifecycle_reproduction_mutation_state", "live_lifecycle_execution", "mutation_rng_draw_schedule", "lineage_founder_state"}:
        return "resident lifecycle/genetics native generation call"
    if surface_name == "public_output_compatibility":
        return "typed event tape compatibility projection"
    if surface_name == "full_resident_checkpoint":
        return "resident checkpoint pack/restore/rebuild"
    if surface_name == "multi_cell_batching":
        return "native campaign batch broker"
    return "resident execution/capability native control"


def _parity_fixtures(surface_name: str) -> tuple[str, ...]:
    base = [f"test_eiel_cxxbackbone1_{surface_name}"]
    if surface_name in {"action_cache", "hmem", "mlp_gru_policy_banks", "policy_row_mappings"}:
        base.append("test_eiel_cxxbackbone1_policy_authority")
    if surface_name in {"patch_world_state", "jungle_state", "wdl_state"}:
        base.append("test_eiel_cxxbackbone1_environment_surfaces")
    if surface_name in {"lineage_founder_state", "lifecycle_reproduction_mutation_state", "full_resident_checkpoint", "public_output_compatibility", "live_lifecycle_execution", "multi_cell_batching"}:
        base.append("test_eiel_cxxbackbone1_integrated_surfaces")
    return tuple(base)


def _performance_counters(surface_name: str) -> tuple[str, ...]:
    if surface_name in {"action_cache", "hmem", "mlp_gru_policy_banks", "policy_row_mappings"}:
        return ("policy_forward_ms", "action_cache_refill_ms", "python_callback_count")
    if surface_name in {"patch_world_state", "jungle_state", "wdl_state"}:
        return ("env_apply_ms", "patch_wdl_jungle_native_ms")
    if surface_name in {"lifecycle_reproduction_mutation_state", "live_lifecycle_execution"}:
        return ("birth_handling_ms", "population_transaction_ms")
    if surface_name == "public_output_compatibility":
        return ("event_tape_ms", "public_projection_ms")
    if surface_name == "full_resident_checkpoint":
        return ("checkpoint_pack_ms", "checkpoint_restore_ms")
    if surface_name == "multi_cell_batching":
        return ("cell_batch_wall_s", "cross_cell_contamination_count")
    return ("native_coverage_count",)


def _runner_arrays(runner: Any) -> dict[str, np.ndarray]:
    env = getattr(runner, "env", None)
    pop = getattr(runner, "pop", None)
    out: dict[str, np.ndarray] = {}
    for name, attr, dtype in (
        ("agent_alive", "agent_alive", np.bool_),
        ("agent_pos_y", "agent_pos_y", np.int32),
        ("agent_pos_x", "agent_pos_x", np.int32),
        ("agent_energy", "agent_energy", np.float32),
        ("agent_health", "agent_health", np.float32),
        ("agent_grid", "agent_grid", np.int32),
    ):
        value = getattr(env, attr, None)
        if value is not None:
            out[name] = np.ascontiguousarray(value, dtype=dtype)
    agents = list(pop.alive_agents()) if pop is not None and hasattr(pop, "alive_agents") else []
    out["alive_agent_ids"] = np.ascontiguousarray([int(getattr(agent, "agent_id", -1)) for agent in agents], dtype=np.int64)
    out["alive_slots"] = np.ascontiguousarray([int(getattr(agent, "slot", -1)) for agent in agents], dtype=np.int32)
    return out


def _group_a_policy_core_status(
    runner: Any,
    *,
    by_name: Mapping[str, LiveSurfaceCompositionV1],
    surfaces: Sequence[str],
) -> dict[str, Any]:
    blockers: list[str] = []
    cfg = getattr(runner, "sim_cfg", None)
    if getattr(runner, "_mlp_pack", None) is None:
        blockers.append("mlp_pack_missing_or_not_synced")
    if getattr(runner, "_gru_pack", None) is None:
        blockers.append("gru_pack_missing_or_not_synced")
    if getattr(runner, "_Hmem", None) is None:
        blockers.append("hmem_tensor_missing")
    if bool(getattr(runner, "_bank_dirty_mlp", False)):
        blockers.append("mlp_bank_dirty_before_native_generation")
    if bool(getattr(runner, "_bank_dirty_gru", False)):
        blockers.append("gru_bank_dirty_before_native_generation")
    if not bool(getattr(cfg, "policy_deterministic_actions", False)):
        blockers.append("stochastic_policy_rng_sampling_not_native_owned")
    blockers.extend(
        [
            "production_observation_builder_still_python_owned",
            "alive_order_shuffle_and_row_grouping_still_python_owned",
            "hmem_writeback_still_torch_python_orchestrated",
            "action_cache_service_still_runnercore_python_orchestrated",
            "checkpoint_rebuild_for_policy_core_not_live_integrated",
        ]
    )
    live_surfaces = [name for name in surfaces if by_name[name].live_integrated]
    return {
        "surfaces": list(surfaces),
        "live_integrated_surfaces": live_surfaces,
        "blocked_surfaces": [name for name in surfaces if not by_name[name].live_integrated],
        "supported": False,
        "blockers": blockers,
        "next_repair": (
            "move production observation staging plus MLP/GRU bank row execution into one native policy-core call, "
            "then compare exact actions/logits/Hmem/cache/RNG before env apply"
        ),
    }


def _generic_group_status(
    by_name: Mapping[str, LiveSurfaceCompositionV1],
    surfaces: Sequence[str],
) -> dict[str, Any]:
    live = [name for name in surfaces if by_name[name].live_integrated]
    blocked = [name for name in surfaces if not by_name[name].live_integrated]
    return {
        "surfaces": list(surfaces),
        "live_integrated_surfaces": live,
        "blocked_surfaces": blocked,
        "supported": not blocked,
        "blockers": [by_name[name].blocker for name in blocked],
    }


def _array_payload(value: np.ndarray) -> dict[str, Any]:
    arr = np.ascontiguousarray(value)
    return {"dtype": str(arr.dtype), "shape": [int(x) for x in arr.shape], "sha256": _array_digest(arr)}


def _array_digest(value: Any) -> str:
    arr = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(json.dumps(list(arr.shape), separators=(",", ":")).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _digest_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


__all__ = [
    "COMPOSITION_MAP_SCHEMA_VERSION",
    "EXECUTION_RESULT_SCHEMA_VERSION",
    "RESIDENT_STATE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SUMMARY_SCHEMA_VERSION",
    "LiveSurfaceCompositionV1",
    "ResidentRuntimeStateV5",
    "build_live_composition_map",
    "capture_resident_runtime_state_v5",
    "composition_group_status",
    "composition_map_payload",
    "execute_generation",
    "resident_runtime_state_v5_payload",
    "summary_payload",
    "write_composition_map",
    "write_summary",
]
