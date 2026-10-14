from __future__ import annotations

"""CXXWAVE1H capability-aware live native backbone.

Default-off prototype for the first concrete CXXWAVE1H tier. It replaces the
global "everything must be native" idea with tier-specific capability
resolution, then exercises a single-call flat MLP generation kernel.
"""

from dataclasses import dataclass
import ctypes
import hashlib
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "cxxwave1h_capability_live_backbone_v1"
CAPABILITY_GRAPH_SCHEMA_VERSION = "cpp_backbone_capability_graph_v2"
FLAT_MLP_INPUT_SCHEMA_VERSION = "cxxwave1h_flat_mlp_generation_input_v1"
FLAT_MLP_RESULT_SCHEMA_VERSION = "cxxwave1h_flat_mlp_generation_result_v1"
SUMMARY_NAMES = ("energy_sum", "energy_min", "energy_max", "died_count", "pos_checksum", "grid_sum", "cost_total")


@dataclass(frozen=True)
class CapabilityNodeV2:
    capability_id: str
    tier: str
    feature_predicates: tuple[str, ...]
    required_native_state: tuple[str, ...]
    required_kernels: tuple[str, ...]
    required_checkpoint_state: tuple[str, ...]
    required_event_state: tuple[str, ...]
    required_rng_state: tuple[str, ...]
    dependencies: tuple[str, ...]
    incompatibilities: tuple[str, ...]
    support_tests: tuple[str, ...]
    rejection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class CapabilityResolutionV2:
    requested_tier: str
    active_capabilities: tuple[str, ...]
    inactive_capabilities: tuple[str, ...]
    missing_state: tuple[str, ...]
    missing_kernels: tuple[str, ...]
    blocked_capabilities: tuple[str, ...]
    blockers: tuple[str, ...]
    supported: bool


@dataclass(frozen=True)
class FlatMlpGenerationInputV1:
    agent_ids: np.ndarray
    initial_energy: np.ndarray
    initial_health: np.ndarray
    initial_pos_y_by_id: np.ndarray
    initial_pos_x_by_id: np.ndarray
    initial_agent_grid: np.ndarray
    initial_grid: np.ndarray
    mlp_pack: np.ndarray
    drives: np.ndarray
    gain: np.ndarray
    aggression: np.ndarray
    temperature: np.ndarray
    move_cost_mults: np.ndarray
    juvenile_harvest_mults: np.ndarray
    scalar_costs: np.ndarray
    action_cache_cursor: np.ndarray
    cache_stride: int
    ticks: int
    height: int
    width: int
    view_radius: int
    hidden: int
    actions: int
    move_cost: float
    gather_cost: float
    rest_cost: float
    attack_cost: float
    action_cost_mult: float
    rest_cost_mult: float
    food_energy: float
    max_energy: float
    w_hunger_gather: float
    w_pain_rest: float
    w_fatigue_rest: float
    w_pain_attack: float
    w_fear_attack: float


@dataclass(frozen=True)
class FlatMlpGenerationResultV1:
    output: Mapping[str, np.ndarray]
    actions_by_tick: np.ndarray
    logits_by_tick: np.ndarray
    action_cache_cursor: np.ndarray
    final_summary: Mapping[str, float]
    timing: Mapping[str, float]
    executed_native: bool
    hidden_fallback: bool


def cpp_backbone_capability_graph_v2() -> tuple[CapabilityNodeV2, ...]:
    return (
        CapabilityNodeV2(
            capability_id="flat_mlp_generation_native",
            tier="T0",
            feature_predicates=("flat_world", "mlp_only", "deterministic_actions", "no_hmem", "no_checkpoint", "no_public_report"),
            required_native_state=(
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
            required_kernels=("cxxwave1h_flat_mlp_generation_v1",),
            required_checkpoint_state=(),
            required_event_state=(),
            required_rng_state=(),
            dependencies=(),
            incompatibilities=("patch_world", "jungle", "wdl", "gru", "hmem", "births", "mutation", "public_report", "checkpoint"),
            support_tests=("test_cxxwave1h_t0_native_flat_mlp_generation_parity",),
        ),
        CapabilityNodeV2(
            capability_id="flat_gru_hmem_generation_native",
            tier="T1",
            feature_predicates=("flat_world", "gru_or_hmem_active"),
            required_native_state=("gru_policy_banks", "hmem", "action_cache_cursor"),
            required_kernels=("future_cxxwave1h_flat_gru_generation",),
            required_checkpoint_state=("hmem", "action_cache_cursor"),
            required_event_state=(),
            required_rng_state=(),
            dependencies=("flat_mlp_generation_native",),
            incompatibilities=("patch_world", "jungle", "wdl", "births", "mutation", "public_report"),
            rejection_reasons=("production GRU/Hmem native execution not implemented in CXXWAVE1H",),
            support_tests=(),
        ),
        CapabilityNodeV2(
            capability_id="patch_low_churn_checkpoint_native",
            tier="T2",
            feature_predicates=("patch_world", "no_churn", "checkpoint_active"),
            required_native_state=("patch_world_state", "checkpoint_cursors", "rng_state_indices"),
            required_kernels=("future_cxxwave1h_patch_checkpoint_generation",),
            required_checkpoint_state=("full_resident_checkpoint",),
            required_event_state=(),
            required_rng_state=("env_rng", "policy_rng"),
            dependencies=("flat_mlp_generation_native",),
            incompatibilities=("jungle", "wdl", "births", "mutation", "public_report"),
            rejection_reasons=("Patch checkpoint native generation is not implemented in CXXWAVE1H",),
            support_tests=(),
        ),
        CapabilityNodeV2(
            capability_id="patch_high_churn_lifecycle_native",
            tier="T3",
            feature_predicates=("patch_world", "high_churn"),
            required_native_state=("population_transaction_tape", "mutation_rng", "policy_row_mappings", "hmem"),
            required_kernels=("future_cxxwave1h_lifecycle_generation",),
            required_checkpoint_state=("population_transaction_tape",),
            required_event_state=("birth_death_event_tape",),
            required_rng_state=("mutation_rng",),
            dependencies=("patch_low_churn_checkpoint_native",),
            incompatibilities=("public_report", "wdl", "jungle"),
            rejection_reasons=("Lifecycle reproduction/mutation remains Python-owned",),
            support_tests=(),
        ),
        CapabilityNodeV2(
            capability_id="public_lineage_report_native",
            tier="T4",
            feature_predicates=("public_report", "lineage_reporting"),
            required_native_state=("typed_event_tape", "lineage_identity"),
            required_kernels=("future_cxxwave1h_public_event_export",),
            required_checkpoint_state=("event_cursor",),
            required_event_state=("typed_event_tape", "compatibility_export_cursor"),
            required_rng_state=(),
            dependencies=("patch_high_churn_lifecycle_native",),
            incompatibilities=(),
            rejection_reasons=("public compatibility output bytes remain Python-owned",),
            support_tests=(),
        ),
        CapabilityNodeV2(
            capability_id="wdl_gru_science_native",
            tier="T5",
            feature_predicates=("wdl", "gru_or_hmem_active"),
            required_native_state=("wdl_state", "gru_policy_banks", "hmem", "rng_state_indices"),
            required_kernels=("future_cxxwave1h_wdl_gru_generation",),
            required_checkpoint_state=("wdl_state", "hmem", "rng_state_indices"),
            required_event_state=("typed_event_tape",),
            required_rng_state=("env_rng", "policy_rng"),
            dependencies=("public_lineage_report_native",),
            incompatibilities=("jungle",),
            rejection_reasons=("WDL delayed hazard and GRU/Hmem native coupling not implemented",),
            support_tests=(),
        ),
        CapabilityNodeV2(
            capability_id="patch_jungle_native",
            tier="T6",
            feature_predicates=("patch_world", "jungle"),
            required_native_state=("patch_world_state", "jungle_state"),
            required_kernels=("future_cxxwave1h_patch_jungle_generation",),
            required_checkpoint_state=("patch_world_state", "jungle_state"),
            required_event_state=("typed_event_tape",),
            required_rng_state=("env_rng",),
            dependencies=("wdl_gru_science_native",),
            incompatibilities=(),
            rejection_reasons=("Jungle state remains outside the native T0/T1 envelope",),
            support_tests=(),
        ),
        CapabilityNodeV2(
            capability_id="multi_cell_runtime_native",
            tier="T7",
            feature_predicates=("multi_cell_campaign",),
            required_native_state=("cell_isolation", "batch_scatter", "per_cell_rng_state"),
            required_kernels=("future_cxxwave1h_multicell_generation",),
            required_checkpoint_state=("per_cell_checkpoint_cursors",),
            required_event_state=("per_cell_event_cursors",),
            required_rng_state=("per_cell_rng_state",),
            dependencies=("patch_jungle_native",),
            incompatibilities=(),
            rejection_reasons=("multi-cell native runtime broker not implemented",),
            support_tests=(),
        ),
    )


def resolve_capabilities_v2(
    *,
    requested_tier: str,
    active_features: Sequence[str],
    native_state: Sequence[str],
    native_kernels: Sequence[str],
) -> CapabilityResolutionV2:
    tier_order = ["T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7"]
    tier_limit = tier_order.index(str(requested_tier)) if str(requested_tier) in tier_order else 0
    active = set(str(x) for x in active_features)
    state = set(str(x) for x in native_state)
    kernels = set(str(x) for x in native_kernels)
    active_caps: list[str] = []
    inactive_caps: list[str] = []
    missing_state: list[str] = []
    missing_kernels: list[str] = []
    blocked_caps: list[str] = []
    blockers: list[str] = []
    for node in cpp_backbone_capability_graph_v2():
        if tier_order.index(node.tier) > tier_limit:
            inactive_caps.append(node.capability_id)
            continue
        predicates = set(node.feature_predicates)
        incompatible = set(node.incompatibilities)
        active_for_cell = predicates.issubset(active) and not (incompatible & active)
        if not active_for_cell:
            inactive_caps.append(node.capability_id)
            continue
        active_caps.append(node.capability_id)
        node_missing_state = [name for name in node.required_native_state if name not in state]
        node_missing_kernels = [name for name in node.required_kernels if name not in kernels]
        node_blockers = [f"{node.capability_id}:missing_state:{name}" for name in node_missing_state]
        node_blockers.extend(f"{node.capability_id}:missing_kernel:{name}" for name in node_missing_kernels)
        node_blockers.extend(f"{node.capability_id}:rejected:{reason}" for reason in node.rejection_reasons)
        if node_blockers:
            blocked_caps.append(node.capability_id)
            blockers.extend(node_blockers)
            missing_state.extend(node_missing_state)
            missing_kernels.extend(node_missing_kernels)
    return CapabilityResolutionV2(
        requested_tier=str(requested_tier),
        active_capabilities=tuple(active_caps),
        inactive_capabilities=tuple(inactive_caps),
        missing_state=tuple(sorted(set(missing_state))),
        missing_kernels=tuple(sorted(set(missing_kernels))),
        blocked_capabilities=tuple(blocked_caps),
        blockers=tuple(blockers),
        supported=not blockers,
    )


def capability_graph_payload(resolution: CapabilityResolutionV2 | None = None) -> dict[str, Any]:
    return {
        "schema_version": CAPABILITY_GRAPH_SCHEMA_VERSION,
        "nodes": [node.__dict__ for node in cpp_backbone_capability_graph_v2()],
        "resolution": resolution.__dict__ if resolution is not None else None,
    }


def make_flat_mlp_fixture(
    *,
    population: int = 16,
    grid: int = 8,
    ticks: int = 8,
    seed: int = 824,
    hidden: int = 8,
    view_radius: int = 1,
) -> FlatMlpGenerationInputV1:
    rng = np.random.default_rng(int(seed))
    n = int(population)
    height = width = int(grid)
    agent_ids = np.arange(n, dtype=np.int32)
    pos_y = np.zeros((n,), dtype=np.int32)
    pos_x = np.zeros((n,), dtype=np.int32)
    occ = np.full((height, width), -1, dtype=np.int32)
    for aid in range(n):
        y = 1 + (aid // max(1, width - 2)) % max(1, height - 2)
        x = 1 + (aid % max(1, width - 2))
        while occ[y, x] != -1:
            x += 1
            if x >= width - 1:
                x = 1
                y = 1 + ((y + 1) % max(1, height - 2))
        pos_y[aid] = y
        pos_x[aid] = x
        occ[y, x] = aid
    food = np.zeros((height, width), dtype=np.int8)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if occ[y, x] == -1 and (x * 7 + y * 5 + int(seed)) % 4 == 0:
                food[y, x] = 1
    obs_cols = (2 * int(view_radius) + 1) ** 2
    actions = 7
    pack_stride = hidden * obs_cols + hidden + hidden * hidden + hidden + actions * hidden + actions
    mlp_pack = rng.normal(0.0, 0.08, size=(n, pack_stride)).astype(np.float32)
    # Make the fixture movement-rich and deterministic without hard-coding actions:
    # action biases vary by row so the native MLP path must still execute.
    b3_off = hidden * obs_cols + hidden + hidden * hidden + hidden + actions * hidden
    for row in range(n):
        mlp_pack[row, b3_off + (row % 5)] += np.float32(0.45)
        mlp_pack[row, b3_off + 4] += np.float32(0.10 if row % 3 == 0 else -0.05)
    return FlatMlpGenerationInputV1(
        agent_ids=agent_ids,
        initial_energy=np.full((n,), 12.0, dtype=np.float32),
        initial_health=np.full((n,), 10.0, dtype=np.float32),
        initial_pos_y_by_id=pos_y,
        initial_pos_x_by_id=pos_x,
        initial_agent_grid=occ,
        initial_grid=food,
        mlp_pack=np.ascontiguousarray(mlp_pack, dtype=np.float32),
        drives=rng.random((n, 4), dtype=np.float32) * np.float32(0.25),
        gain=np.full((n,), 1.0, dtype=np.float32),
        aggression=np.zeros((n,), dtype=np.float32),
        temperature=np.ones((n,), dtype=np.float32),
        move_cost_mults=np.ones((n,), dtype=np.float32),
        juvenile_harvest_mults=np.ones((n,), dtype=np.float32),
        scalar_costs=np.full((n,), 0.015, dtype=np.float32),
        action_cache_cursor=np.zeros((n,), dtype=np.int32),
        cache_stride=16,
        ticks=int(ticks),
        height=height,
        width=width,
        view_radius=int(view_radius),
        hidden=int(hidden),
        actions=actions,
        move_cost=0.05,
        gather_cost=0.02,
        rest_cost=0.005,
        attack_cost=0.08,
        action_cost_mult=1.0,
        rest_cost_mult=1.0,
        food_energy=0.7,
        max_energy=20.0,
        w_hunger_gather=0.35,
        w_pain_rest=0.25,
        w_fatigue_rest=0.15,
        w_pain_attack=0.20,
        w_fear_attack=0.10,
    )


def run_flat_mlp_python_reference(inputs: FlatMlpGenerationInputV1) -> FlatMlpGenerationResultV1:
    t0 = time.perf_counter()
    ids = np.ascontiguousarray(inputs.agent_ids, dtype=np.int32)
    n = int(ids.shape[0])
    energy = np.asarray(inputs.initial_energy, dtype=np.float32).copy()
    health = np.asarray(inputs.initial_health, dtype=np.float32).copy()
    pos_y = np.asarray(inputs.initial_pos_y_by_id, dtype=np.int32).copy()
    pos_x = np.asarray(inputs.initial_pos_x_by_id, dtype=np.int32).copy()
    occ = np.asarray(inputs.initial_agent_grid, dtype=np.int32).copy()
    grid = np.asarray(inputs.initial_grid, dtype=np.int8).copy()
    cost_total = np.zeros((n,), dtype=np.float32)
    food_total = np.zeros((n,), dtype=np.float32)
    moved = np.zeros((n,), dtype=bool)
    died = np.zeros((n,), dtype=bool)
    cursor = np.asarray(inputs.action_cache_cursor, dtype=np.int32).copy()
    actions_by_tick = np.zeros((int(inputs.ticks), n), dtype=np.int32)
    logits_by_tick = np.zeros((int(inputs.ticks), n, int(inputs.actions)), dtype=np.float32)
    order = np.argsort(ids, kind="mergesort")

    for tick in range(int(inputs.ticks)):
        tick_actions = np.zeros((n,), dtype=np.int32)
        costs = np.zeros((n,), dtype=np.float32)
        for src in order.tolist():
            obs = _flat_observation(occ, grid, int(pos_y[int(ids[src])]), int(pos_x[int(ids[src])]), int(inputs.view_radius))
            logits = _mlp_logits(inputs, src, obs)
            logits_by_tick[tick, src, :] = logits
            action = int(np.argmax(logits))
            tick_actions[src] = action
            actions_by_tick[tick, src] = action
            cursor[src] = np.int32((int(cursor[src]) + 1) % int(inputs.cache_stride)) if int(inputs.cache_stride) > 0 else np.int32(int(cursor[src]) + 1)
            cost = float(inputs.move_cost) * float(inputs.move_cost_mults[src])
            if action == 4:
                cost = float(inputs.gather_cost)
            elif action == 5:
                cost = float(inputs.rest_cost)
            elif action == 6:
                cost = float(inputs.attack_cost)
            if float(inputs.action_cost_mult) != 1.0:
                cost *= float(inputs.action_cost_mult)
            if float(inputs.rest_cost_mult) != 1.0 and action == 5:
                cost *= float(inputs.rest_cost_mult)
            costs[src] = np.float32(cost)
        food_gained = np.zeros((n,), dtype=np.float32)
        for src in order.tolist():
            action = int(tick_actions[src])
            aid = int(ids[src])
            y = int(pos_y[aid])
            x = int(pos_x[aid])
            if 0 <= action <= 3:
                ny = y + (-1 if action == 0 else 1 if action == 1 else 0)
                nx = x + (1 if action == 2 else -1 if action == 3 else 0)
                if 0 <= ny < int(inputs.height) and 0 <= nx < int(inputs.width) and int(occ[ny, nx]) == -1:
                    if int(occ[y, x]) == aid:
                        occ[y, x] = -1
                    occ[ny, nx] = aid
                    pos_y[aid] = np.int32(ny)
                    pos_x[aid] = np.int32(nx)
                    moved[src] = True
            elif action == 4 and int(grid[y, x]) == 1:
                grid[y, x] = np.int8(0)
                food_gained[src] = np.float32(float(inputs.food_energy) * float(inputs.juvenile_harvest_mults[src]))
        cost_total = np.ascontiguousarray(cost_total + inputs.scalar_costs.astype(np.float32, copy=False))
        food_total = np.ascontiguousarray(food_total + food_gained)
        energy = energy - costs + food_gained - inputs.scalar_costs.astype(np.float32, copy=False)
        energy = np.maximum(np.minimum(energy, np.float32(float(inputs.max_energy))), np.float32(0.0)).astype(np.float32, copy=False)
        died = ((energy <= 0.0) | (health <= 0.0)).astype(bool, copy=False)

    output = {
        "new_energy": np.ascontiguousarray(energy),
        "new_health": np.ascontiguousarray(health),
        "new_pos_y": np.ascontiguousarray(pos_y[ids].astype(np.int32, copy=False)),
        "new_pos_x": np.ascontiguousarray(pos_x[ids].astype(np.int32, copy=False)),
        "grid": np.ascontiguousarray(grid),
        "agent_grid": np.ascontiguousarray(occ),
        "cost_total_by_agent": np.ascontiguousarray(cost_total),
        "food_gained": np.ascontiguousarray(food_total),
        "moved_mask": np.ascontiguousarray(moved),
        "died_mask": np.ascontiguousarray(died),
        "views_dirty": np.asarray([bool(np.any(moved) or np.any(food_total > 0.0))], dtype=bool),
    }
    return FlatMlpGenerationResultV1(
        output=output,
        actions_by_tick=np.ascontiguousarray(actions_by_tick),
        logits_by_tick=np.ascontiguousarray(logits_by_tick),
        action_cache_cursor=np.ascontiguousarray(cursor),
        final_summary=_summary_dict(inputs.agent_ids, energy, died, pos_y, pos_x, grid, cost_total),
        timing={"wall_s": float(time.perf_counter() - t0), "backend_compute_s": float(time.perf_counter() - t0)},
        executed_native=False,
        hidden_fallback=False,
    )


def compile_flat_mlp_kernel(build_dir: Path | str, *, cxx: str | None = None) -> dict[str, Any]:
    compiler = cxx or shutil.which("g++")
    build = Path(build_dir)
    source_path = ROOT / "engine" / "perf" / "native" / "cxxwave1h_flat_mlp_generation_kernel.cpp"
    library_path = build / "libcxxwave1h_flat_mlp_generation.so"
    if not compiler:
        return {"ok": False, "reason": "g++ not found", "library_path": "", "source_path": str(source_path), "command": []}
    build.mkdir(parents=True, exist_ok=True)
    cmd = [str(compiler), "-O3", "-std=c++17", "-shared", "-fPIC", str(source_path), "-o", str(library_path)]
    cp = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return {
        "ok": cp.returncode == 0,
        "reason": "" if cp.returncode == 0 else "compile failed",
        "command": cmd,
        "returncode": int(cp.returncode),
        "stdout_tail": cp.stdout[-4000:],
        "stderr_tail": cp.stderr[-4000:],
        "source_path": str(source_path),
        "library_path": str(library_path) if cp.returncode == 0 else "",
    }


def load_flat_mlp_native_backend(library_path: Path | str):
    lib = ctypes.CDLL(str(library_path))
    fn = lib.cxxwave1h_flat_mlp_generation_v1
    f32p = ctypes.POINTER(ctypes.c_float)
    f64p = ctypes.POINTER(ctypes.c_double)
    i32p = ctypes.POINTER(ctypes.c_int32)
    i8p = ctypes.POINTER(ctypes.c_int8)
    u8p = ctypes.POINTER(ctypes.c_uint8)
    fn.argtypes = [
        i32p, f32p, f32p, i32p, i32p, ctypes.c_int64, i32p, i8p, f32p, ctypes.c_int64,
        f32p, f32p, f32p, f32p, f32p, f32p, f32p, i32p, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        f32p, f32p, i32p, i32p, i8p, i32p, f32p, f32p, u8p, u8p, u8p, i32p, f32p, i32p, f64p,
    ]
    fn.restype = ctypes.c_int

    def _call(inputs: FlatMlpGenerationInputV1) -> FlatMlpGenerationResultV1:
        prep_t0 = time.perf_counter()
        ids = np.ascontiguousarray(inputs.agent_ids, dtype=np.int32)
        n = int(ids.shape[0])
        height = int(inputs.height)
        width = int(inputs.width)
        ticks = int(inputs.ticks)
        actions = int(inputs.actions)
        out_energy = np.zeros((n,), dtype=np.float32)
        out_health = np.zeros((n,), dtype=np.float32)
        out_pos_y = np.zeros((n,), dtype=np.int32)
        out_pos_x = np.zeros((n,), dtype=np.int32)
        out_grid = np.zeros((height * width,), dtype=np.int8)
        out_agent_grid = np.zeros((height * width,), dtype=np.int32)
        out_cost = np.zeros((n,), dtype=np.float32)
        out_food = np.zeros((n,), dtype=np.float32)
        out_moved = np.zeros((n,), dtype=np.uint8)
        out_died = np.zeros((n,), dtype=np.uint8)
        out_views_dirty = np.zeros((1,), dtype=np.uint8)
        out_actions = np.zeros((ticks, n), dtype=np.int32)
        out_logits = np.zeros((ticks, n, actions), dtype=np.float32)
        out_cursor = np.zeros((n,), dtype=np.int32)
        out_summary = np.zeros((len(SUMMARY_NAMES),), dtype=np.float64)
        prep_s = time.perf_counter() - prep_t0
        compute_t0 = time.perf_counter()
        status = fn(
            ids.ctypes.data_as(i32p),
            np.ascontiguousarray(inputs.initial_energy, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(inputs.initial_health, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(inputs.initial_pos_y_by_id, dtype=np.int32).ctypes.data_as(i32p),
            np.ascontiguousarray(inputs.initial_pos_x_by_id, dtype=np.int32).ctypes.data_as(i32p),
            ctypes.c_int64(int(inputs.initial_pos_y_by_id.shape[0])),
            np.ascontiguousarray(inputs.initial_agent_grid, dtype=np.int32).ctypes.data_as(i32p),
            np.ascontiguousarray(inputs.initial_grid, dtype=np.int8).ctypes.data_as(i8p),
            np.ascontiguousarray(inputs.mlp_pack, dtype=np.float32).ctypes.data_as(f32p),
            ctypes.c_int64(int(inputs.mlp_pack.shape[1])),
            np.ascontiguousarray(inputs.drives, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(inputs.gain, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(inputs.aggression, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(inputs.temperature, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(inputs.move_cost_mults, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(inputs.juvenile_harvest_mults, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(inputs.scalar_costs, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(inputs.action_cache_cursor, dtype=np.int32).ctypes.data_as(i32p),
            ctypes.c_int64(int(inputs.cache_stride)),
            ctypes.c_int64(ticks),
            ctypes.c_int64(n),
            ctypes.c_int64(height),
            ctypes.c_int64(width),
            ctypes.c_int64(int(inputs.view_radius)),
            ctypes.c_int64(int(inputs.hidden)),
            ctypes.c_int64((2 * int(inputs.view_radius) + 1) ** 2),
            ctypes.c_int64(actions),
            ctypes.c_float(float(inputs.move_cost)),
            ctypes.c_float(float(inputs.gather_cost)),
            ctypes.c_float(float(inputs.rest_cost)),
            ctypes.c_float(float(inputs.attack_cost)),
            ctypes.c_float(float(inputs.action_cost_mult)),
            ctypes.c_float(float(inputs.rest_cost_mult)),
            ctypes.c_float(float(inputs.food_energy)),
            ctypes.c_float(float(inputs.max_energy)),
            ctypes.c_float(float(inputs.w_hunger_gather)),
            ctypes.c_float(float(inputs.w_pain_rest)),
            ctypes.c_float(float(inputs.w_fatigue_rest)),
            ctypes.c_float(float(inputs.w_pain_attack)),
            ctypes.c_float(float(inputs.w_fear_attack)),
            out_energy.ctypes.data_as(f32p),
            out_health.ctypes.data_as(f32p),
            out_pos_y.ctypes.data_as(i32p),
            out_pos_x.ctypes.data_as(i32p),
            out_grid.ctypes.data_as(i8p),
            out_agent_grid.ctypes.data_as(i32p),
            out_cost.ctypes.data_as(f32p),
            out_food.ctypes.data_as(f32p),
            out_moved.ctypes.data_as(u8p),
            out_died.ctypes.data_as(u8p),
            out_views_dirty.ctypes.data_as(u8p),
            out_actions.ctypes.data_as(i32p),
            out_logits.ctypes.data_as(f32p),
            out_cursor.ctypes.data_as(i32p),
            out_summary.ctypes.data_as(f64p),
        )
        compute_s = time.perf_counter() - compute_t0
        if int(status) != 0:
            raise RuntimeError(f"CXXWAVE1H native flat MLP generation failed with status {int(status)}")
        output = {
            "new_energy": out_energy,
            "new_health": out_health,
            "new_pos_y": out_pos_y,
            "new_pos_x": out_pos_x,
            "grid": out_grid.reshape((height, width)),
            "agent_grid": out_agent_grid.reshape((height, width)),
            "cost_total_by_agent": out_cost,
            "food_gained": out_food,
            "moved_mask": out_moved.astype(bool),
            "died_mask": out_died.astype(bool),
            "views_dirty": out_views_dirty.astype(bool),
        }
        return FlatMlpGenerationResultV1(
            output=output,
            actions_by_tick=out_actions,
            logits_by_tick=out_logits,
            action_cache_cursor=out_cursor,
            final_summary={name: float(out_summary[i]) for i, name in enumerate(SUMMARY_NAMES)},
            timing={"setup_s": float(prep_s), "backend_compute_s": float(compute_s), "wall_s": float(prep_s + compute_s)},
            executed_native=True,
            hidden_fallback=False,
        )

    return _call


def compare_flat_mlp_results(
    expected: FlatMlpGenerationResultV1,
    observed: FlatMlpGenerationResultV1,
    *,
    atol: float = 1.0e-5,
) -> dict[str, Any]:
    checks: list[tuple[str, Any, Any]] = []
    for name in sorted(expected.output):
        checks.append((name, expected.output[name], observed.output[name]))
    checks.extend(
        [
            ("actions_by_tick", expected.actions_by_tick, observed.actions_by_tick),
            ("logits_by_tick", expected.logits_by_tick, observed.logits_by_tick),
            ("action_cache_cursor", expected.action_cache_cursor, observed.action_cache_cursor),
        ]
    )
    for name, exp, got in checks:
        a = np.asarray(exp)
        b = np.asarray(got)
        if a.shape != b.shape:
            return {"ok": False, "first_divergence": {"field": name, "expected_shape": list(a.shape), "observed_shape": list(b.shape)}}
        if a.dtype.kind in {"f", "c"} or b.dtype.kind in {"f", "c"}:
            diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
            if bool(np.any(diff > float(atol))):
                idx = tuple(int(x) for x in np.argwhere(diff > float(atol))[0])
                return {
                    "ok": False,
                    "first_divergence": {
                        "field": name,
                        "index": list(idx),
                        "expected": float(a[idx]),
                        "observed": float(b[idx]),
                        "abs_delta": float(diff[idx]),
                    },
                }
        elif not bool(np.array_equal(a, b)):
            idx = tuple(int(x) for x in np.argwhere(a != b)[0])
            return {"ok": False, "first_divergence": {"field": name, "index": list(idx), "expected": str(a[idx]), "observed": str(b[idx])}}
    for name in SUMMARY_NAMES:
        delta = abs(float(expected.final_summary[name]) - float(observed.final_summary[name]))
        if delta > float(atol):
            return {
                "ok": False,
                "first_divergence": {
                    "field": f"final_summary.{name}",
                    "expected": float(expected.final_summary[name]),
                    "observed": float(observed.final_summary[name]),
                    "abs_delta": float(delta),
                },
            }
    return {"ok": True, "first_divergence": None, "state_digest": result_digest(observed)}


def execute_t0_flat_mlp_generation(
    inputs: FlatMlpGenerationInputV1,
    *,
    native_backend: Any,
) -> dict[str, Any]:
    resolution = resolve_capabilities_v2(
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
    if not resolution.supported:
        return {
            "schema_version": SCHEMA_VERSION,
            "decision": "CXXWAVE1H-E/NO-ACTUAL-LIVE-NATIVE-EXECUTION",
            "executed_native": False,
            "hidden_fallback": False,
            "resolution": resolution.__dict__,
            "parity": {"ok": False, "first_divergence": {"field": "capability_resolution", "blockers": list(resolution.blockers)}},
        }
    reference = run_flat_mlp_python_reference(inputs)
    native = native_backend(inputs)
    parity = compare_flat_mlp_results(reference, native)
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": "CXXWAVE1H-T0-A/FLAT-MLP-LIVE-NATIVE-PARITY" if parity["ok"] else "CXXWAVE1H-T0-C/FLAT-MLP-LIVE-PARITY-BLOCKED",
        "executed_native": bool(native.executed_native),
        "hidden_fallback": False,
        "resolution": resolution.__dict__,
        "parity": parity,
        "python_timing": dict(reference.timing),
        "native_timing": dict(native.timing),
        "python_digest": result_digest(reference),
        "native_digest": result_digest(native),
        "final_summary": dict(native.final_summary),
        "actions_digest": _array_digest(native.actions_by_tick),
        "logits_digest": _array_digest(native.logits_by_tick),
    }


def result_digest(result: FlatMlpGenerationResultV1) -> str:
    payload = {
        "output": {name: _array_digest(value) for name, value in sorted(result.output.items())},
        "actions_by_tick": _array_digest(result.actions_by_tick),
        "logits_by_tick": _array_digest(result.logits_by_tick),
        "action_cache_cursor": _array_digest(result.action_cache_cursor),
        "final_summary": {name: float(result.final_summary[name]) for name in SUMMARY_NAMES},
    }
    return _json_digest(payload)


def input_payload(inputs: FlatMlpGenerationInputV1) -> dict[str, Any]:
    return {
        "schema_version": FLAT_MLP_INPUT_SCHEMA_VERSION,
        "population": int(inputs.agent_ids.shape[0]),
        "ticks": int(inputs.ticks),
        "height": int(inputs.height),
        "width": int(inputs.width),
        "view_radius": int(inputs.view_radius),
        "hidden": int(inputs.hidden),
        "actions": int(inputs.actions),
        "input_digest": _json_digest(
            {
                "agent_ids": _array_digest(inputs.agent_ids),
                "initial_energy": _array_digest(inputs.initial_energy),
                "initial_health": _array_digest(inputs.initial_health),
                "initial_pos_y_by_id": _array_digest(inputs.initial_pos_y_by_id),
                "initial_pos_x_by_id": _array_digest(inputs.initial_pos_x_by_id),
                "initial_agent_grid": _array_digest(inputs.initial_agent_grid),
                "initial_grid": _array_digest(inputs.initial_grid),
                "mlp_pack": _array_digest(inputs.mlp_pack),
                "drives": _array_digest(inputs.drives),
            }
        ),
    }


def _flat_observation(occ: np.ndarray, grid: np.ndarray, y: int, x: int, radius: int) -> np.ndarray:
    values: list[float] = []
    for dy in range(-int(radius), int(radius) + 1):
        for dx in range(-int(radius), int(radius) + 1):
            yy = int(y) + dy
            xx = int(x) + dx
            if 0 <= yy < int(grid.shape[0]) and 0 <= xx < int(grid.shape[1]):
                values.append(-1.0 if int(occ[yy, xx]) != -1 else float(grid[yy, xx]))
            else:
                values.append(0.0)
    return np.asarray(values, dtype=np.float32)


def _mlp_logits(inputs: FlatMlpGenerationInputV1, row: int, obs: np.ndarray) -> np.ndarray:
    h = int(inputs.hidden)
    o = int(obs.shape[0])
    a = int(inputs.actions)
    pack = np.asarray(inputs.mlp_pack[row], dtype=np.float32)
    off = 0
    w1 = pack[off : off + h * o].reshape(h, o)
    off += h * o
    b1 = pack[off : off + h]
    off += h
    w2 = pack[off : off + h * h].reshape(h, h)
    off += h * h
    b2 = pack[off : off + h]
    off += h
    w3 = pack[off : off + a * h].reshape(a, h)
    off += a * h
    b3 = pack[off : off + a]
    h1 = np.zeros((h,), dtype=np.float32)
    h2 = np.zeros((h,), dtype=np.float32)
    logits = np.zeros((a,), dtype=np.float32)
    for yy in range(h):
        acc = np.float32(b1[yy])
        for xx in range(o):
            acc = np.float32(acc + np.float32(w1[yy, xx] * obs[xx]))
        h1[yy] = np.float32(math.tanh(float(acc)))
    for yy in range(h):
        acc = np.float32(b2[yy])
        for xx in range(h):
            acc = np.float32(acc + np.float32(w2[yy, xx] * h1[xx]))
        h2[yy] = np.float32(math.tanh(float(acc)))
    for yy in range(a):
        acc = np.float32(b3[yy])
        for xx in range(h):
            acc = np.float32(acc + np.float32(w3[yy, xx] * h2[xx]))
        logits[yy] = acc
    logits[6] = np.float32(logits[6] + np.float32(inputs.aggression[row]))
    logits = (logits / np.maximum(np.float32(inputs.temperature[row]), np.float32(0.05))).astype(np.float32, copy=False)
    drive = np.asarray(inputs.drives[row], dtype=np.float32)
    gain = np.float32(inputs.gain[row])
    logits[4] = np.float32(logits[4] + gain * (np.float32(inputs.w_hunger_gather) * drive[0]))
    logits[5] = np.float32(logits[5] + gain * (np.float32(inputs.w_pain_rest) * drive[1] + np.float32(inputs.w_fatigue_rest) * drive[2]))
    logits[6] = np.float32(logits[6] - gain * (np.float32(inputs.w_pain_attack) * drive[1] + np.float32(inputs.w_fear_attack) * drive[3]))
    return np.ascontiguousarray(logits)


def _summary_dict(
    agent_ids: np.ndarray,
    energy: np.ndarray,
    died: np.ndarray,
    pos_y: np.ndarray,
    pos_x: np.ndarray,
    grid: np.ndarray,
    cost_total: np.ndarray,
) -> dict[str, float]:
    ids = np.asarray(agent_ids, dtype=np.int32)
    y = pos_y[ids].astype(np.float64, copy=False)
    x = pos_x[ids].astype(np.float64, copy=False)
    e = energy.astype(np.float64, copy=False)
    return {
        "energy_sum": float(np.sum(e, dtype=np.float64)),
        "energy_min": float(np.min(e)) if e.size else 0.0,
        "energy_max": float(np.max(e)) if e.size else 0.0,
        "died_count": float(np.sum(died.astype(np.float64), dtype=np.float64)),
        "pos_checksum": float(np.sum(y * 1009.0 + x * 917.0, dtype=np.float64)),
        "grid_sum": float(np.sum(grid.astype(np.float64), dtype=np.float64)),
        "cost_total": float(np.sum(cost_total.astype(np.float64), dtype=np.float64)),
    }


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
    "FLAT_MLP_INPUT_SCHEMA_VERSION",
    "FLAT_MLP_RESULT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "CapabilityNodeV2",
    "CapabilityResolutionV2",
    "FlatMlpGenerationInputV1",
    "FlatMlpGenerationResultV1",
    "capability_graph_payload",
    "compare_flat_mlp_results",
    "compile_flat_mlp_kernel",
    "cpp_backbone_capability_graph_v2",
    "execute_t0_flat_mlp_generation",
    "input_payload",
    "load_flat_mlp_native_backend",
    "make_flat_mlp_fixture",
    "resolve_capabilities_v2",
    "result_digest",
    "run_flat_mlp_python_reference",
]
