from __future__ import annotations

"""CXXBACKBONE-5 live policy-envelope boundary map and fail-closed ledger."""

from engine.runtime_roots import public_runtime_path

from dataclasses import asdict, dataclass
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


BOUNDARY_SCHEMA_VERSION = "cxxbackbone5_live_policy_boundary_map_v1"
LEDGER_SCHEMA_VERSION = "cxxbackbone5_master_surface_ledger_v1"
PHASE_SCHEMA_VERSION = "cxxbackbone5_phase_attempt_ledger_v1"
COUNTER_SCHEMA_VERSION = "cxxbackbone5_policy_counter_snapshot_v1"

RAW_ROOT_DEFAULT = Path(public_runtime_path('evidence_raw/eiel_cxxbackbone5_20260712'))

POLICY_BOUNDARY_STEPS: tuple[dict[str, Any], ...] = (
    {
        "step": "cache_hit_refill_decision",
        "python_function": "simulation.runnercore.action_cache.runnercore1_actions_strided",
        "torch_aten_function": "",
        "state_owner": "RunnerCore1TickState NumPy arrays, orchestrated by Python",
        "dtype_device": "int64/int32 CPU NumPy",
        "allocations": "hit/miss index arrays; refill id/order arrays",
        "h2d": False,
        "d2h": False,
        "synchronization": "none explicit",
        "python_call_count": 1,
        "launch_count": 0,
        "checkpoint_member": "policy_action_cache_v1 via checkpoint_bundle_v2",
        "intended_native_replacement": "cxxbackbone5_policy_service_v1 cache service over resident slot/cache arrays",
        "current_blocker": "Python branches decide hits/refills and write cache rows.",
    },
    {
        "step": "alive_row_extraction",
        "python_function": "state.slot_for_agent_id loop in _compute_actions_k",
        "torch_aten_function": "",
        "state_owner": "RunnerCore1TickState agent-id maps",
        "dtype_device": "int32/int64 CPU NumPy",
        "allocations": "slot row array",
        "h2d": False,
        "d2h": False,
        "synchronization": "none explicit",
        "python_call_count": "one Python lookup per requested agent id",
        "launch_count": 0,
        "checkpoint_member": "slot_to_agent_id/slot_tokens",
        "intended_native_replacement": "resident row-map gather in cxxbackbone5_policy_service_v1",
        "current_blocker": "Per-agent Python lookup still defines live row order.",
    },
    {
        "step": "observation_construction",
        "python_function": "env.get_local_observations plus NumPy frame assembly",
        "torch_aten_function": "torch.from_numpy(...).to(device)",
        "state_owner": "GridWorld/Patch observation Python env arrays",
        "dtype_device": "float32 CPU NumPy then torch CPU/CUDA",
        "allocations": "local observation, obs_np, energy/health views",
        "h2d": "yes on CUDA lanes for obs/drives/scalars",
        "d2h": False,
        "synchronization": "implicit if CUDA tensors are later copied to CPU",
        "python_call_count": 1,
        "launch_count": "ATen copy/conversion dependent on device",
        "checkpoint_member": "world/env arrays plus scalar state",
        "intended_native_replacement": "native observation gather from resident world/scalar buffers",
        "current_blocker": "Production observation semantics are still Python/env authoritative.",
    },
    {
        "step": "policy_family_grouping",
        "python_function": "np.nonzero(state.use_gru[slots])",
        "torch_aten_function": "",
        "state_owner": "RunnerCore1TickState use_gru array",
        "dtype_device": "bool CPU NumPy",
        "allocations": "mlp_idx/gru_idx arrays",
        "h2d": False,
        "d2h": False,
        "synchronization": "none explicit",
        "python_call_count": 1,
        "launch_count": 0,
        "checkpoint_member": "policy family rows",
        "intended_native_replacement": "resident policy-family partition in native service",
        "current_blocker": "Python creates the production grouping and row order.",
    },
    {
        "step": "row_map_creation",
        "python_function": "_torch_index and NumPy bank slot gathers",
        "torch_aten_function": "index_select",
        "state_owner": "Python-created row maps",
        "dtype_device": "int64 CPU/CUDA torch",
        "allocations": "row and bank-slot tensors",
        "h2d": "yes for CUDA row tensors",
        "d2h": False,
        "synchronization": "none explicit",
        "python_call_count": "one call per active policy family",
        "launch_count": "index_select per bank and input family",
        "checkpoint_member": "policy row mappings",
        "intended_native_replacement": "persistent resident row maps updated on membership change",
        "current_blocker": "Per-refill Python row maps select production policy rows.",
    },
    {
        "step": "mlp_dispatch",
        "python_function": "_compute_actions_k MLP branch",
        "torch_aten_function": "cxxbackbone4_mlp_policy_aten_v1 when enabled",
        "state_owner": "C++/ATen math under default-off flag",
        "dtype_device": "float32 torch CPU/CUDA",
        "allocations": "selected pack/input tensors and output logits",
        "h2d": "inherited from input frame on CUDA",
        "d2h": False,
        "synchronization": "none explicit until action copy",
        "python_call_count": 1,
        "launch_count": "ATen graph dependent",
        "checkpoint_member": "MLP policy bank",
        "intended_native_replacement": "same C++/ATen graph behind unified policy service",
        "current_blocker": "Math is native; surrounding dispatch and residency are Python-owned.",
    },
    {
        "step": "gru_hmem_dispatch",
        "python_function": "_compute_actions_k GRU branch",
        "torch_aten_function": "cxxbackbone3_gru_policy_aten_v1 when enabled",
        "state_owner": "C++/ATen math and Hmem update under default-off flag",
        "dtype_device": "float32 torch CPU/CUDA",
        "allocations": "selected pack/input/Hmem tensors and output logits/Hmem",
        "h2d": "inherited from input frame on CUDA",
        "d2h": False,
        "synchronization": "none explicit until action copy",
        "python_call_count": 1,
        "launch_count": "ATen graph dependent",
        "checkpoint_member": "GRU policy bank and Hmem",
        "intended_native_replacement": "same C++/ATen graph behind unified policy service",
        "current_blocker": "Math is native; Hmem row selection and service orchestration are Python-owned.",
    },
    {
        "step": "policy_rng",
        "python_function": "_compute_actions_k deterministic-only guard",
        "torch_aten_function": "",
        "state_owner": "unsupported for stochastic live authority",
        "dtype_device": "not active in current deterministic lane",
        "allocations": "not applicable",
        "h2d": False,
        "d2h": False,
        "synchronization": "not applicable",
        "python_call_count": 0,
        "launch_count": 0,
        "checkpoint_member": "RNG state/indices",
        "intended_native_replacement": "native policy RNG stream with exact draw-order ledger",
        "current_blocker": "Stochastic action parity is not implemented in the live policy envelope.",
    },
    {
        "step": "action_selection",
        "python_function": "_compute_actions_k",
        "torch_aten_function": "torch.argmax",
        "state_owner": "torch operation scheduled by Python",
        "dtype_device": "int64 torch CPU/CUDA",
        "allocations": "actions_all tensor",
        "h2d": False,
        "d2h": False,
        "synchronization": "none explicit until final CPU copy",
        "python_call_count": "one call per active policy family",
        "launch_count": "argmax/index_copy dependent on active families",
        "checkpoint_member": "policy action stream",
        "intended_native_replacement": "native deterministic/stochastic action selection and scatter",
        "current_blocker": "Output tensor and scatter are Python-scheduled ATen operations.",
    },
    {
        "step": "output_scatter",
        "python_function": "_compute_actions_k actions_all.index_copy_",
        "torch_aten_function": "index_copy_",
        "state_owner": "torch output tensor created by Python",
        "dtype_device": "int64 torch CPU/CUDA",
        "allocations": "actions_all",
        "h2d": False,
        "d2h": "yes at final actions_all.to('cpu').numpy() on CUDA",
        "synchronization": "CUDA-to-CPU action copy synchronizes",
        "python_call_count": "one call per active policy family",
        "launch_count": "index_copy per active family",
        "checkpoint_member": "not persisted directly; cache rows persist selected actions",
        "intended_native_replacement": "resident int32 action buffer consumed by environment apply",
        "current_blocker": "Actions return to CPU NumPy before environment application.",
    },
    {
        "step": "cache_write",
        "python_function": "runnercore1_actions_strided refill writeback",
        "torch_aten_function": "",
        "state_owner": "RunnerCore1TickState cache arrays, written by Python",
        "dtype_device": "int32/int64 CPU NumPy",
        "allocations": "none beyond refill arrays",
        "h2d": False,
        "d2h": False,
        "synchronization": "after action copy on CUDA lanes",
        "python_call_count": "loop or vectorized assignment after refill",
        "launch_count": 0,
        "checkpoint_member": "policy_action_cache_v1",
        "intended_native_replacement": "native cache cursor/action row update",
        "current_blocker": "Python remains authoritative for cache cursor and refill rows.",
    },
    {
        "step": "action_application",
        "python_function": "runner tick-loop environment/action handlers after runnercore1_actions_strided",
        "torch_aten_function": "",
        "state_owner": "Python runner/env/lifecycle code",
        "dtype_device": "CPU Python objects and NumPy env arrays",
        "allocations": "workload dependent",
        "h2d": False,
        "d2h": False,
        "synchronization": "policy CUDA lane already synchronized by CPU action copy",
        "python_call_count": "per tick/per action handler path",
        "launch_count": 0,
        "checkpoint_member": "world/scalar/lifecycle/event state",
        "intended_native_replacement": "native environment/lifecycle/action application over resident state",
        "current_blocker": "Environment, lifecycle, public output, and checkpoint authority remain separate phases.",
    },
)


SURFACE_NAMES: tuple[str, ...] = (
    "action_cache",
    "capability_support_mask",
    "execution_cell_metadata",
    "hmem",
    "jungle_state",
    "lifecycle_reproduction_mutation_state",
    "lineage_founder_state",
    "mlp_gru_policy_banks",
    "observations_staging",
    "patch_world_state",
    "policy_row_mappings",
    "population_slots_stable_ids",
    "rng_state_indices",
    "wdl_state",
    "public_output_compatibility",
    "full_resident_checkpoint",
    "live_lifecycle_execution",
    "mutation_rng_draw_schedule",
    "multi_cell_batching",
)


@dataclass(frozen=True)
class SurfaceLedgerRow:
    surface_name: str
    historical_contract_exact: bool
    current_live_probe_status: str
    current_live_authority: str
    checkpoint_continuation: str
    public_output_parity: str
    cpu_parity: str
    cuda_parity: str
    whole_run_measurement: str
    current_blocker: str


def git_head(root: Path | None = None) -> str:
    cwd = root if root is not None else Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(cwd), text=True).strip()
    except Exception:
        return ""


def sha256_json(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def live_policy_boundary_map_payload(*, head: str = "") -> dict[str, Any]:
    return {
        "schema_version": BOUNDARY_SCHEMA_VERSION,
        "head": str(head),
        "production_entry": "SimulationRunner._run_ticks",
        "policy_entry": "simulation.runnercore.action_cache.runnercore1_actions_strided",
        "refill_entry": "simulation.runnercore.action_cache._compute_actions_k",
        "native_math_authority_flags": [
            "cxxbackbone3_policy_aten_gru_authority_enable",
            "cxxbackbone4_policy_aten_mlp_authority_enable",
        ],
        "full_native_policy_envelope_supported": False,
        "full_native_policy_envelope_blocker": (
            "cache service, observation construction, policy grouping, row maps, policy RNG, "
            "action CPU copy, cache writeback, and action application remain Python-orchestrated"
        ),
        "steps": list(POLICY_BOUNDARY_STEPS),
    }


def master_surface_ledger_rows() -> tuple[SurfaceLedgerRow, ...]:
    rows: list[SurfaceLedgerRow] = []
    for name in SURFACE_NAMES:
        if name in {"execution_cell_metadata", "observations_staging", "population_slots_stable_ids"}:
            live_authority = "inherited_live_integrated_from_runnercore"
            live_probe = "live_integrated"
            blocker = ""
        elif name in {"mlp_gru_policy_banks", "hmem"}:
            live_authority = "default_off_cxxbackbone3_4_aten_math_only"
            live_probe = "cpu_cuda_bitexact_policy_math"
            blocker = "surrounding policy service and full resident checkpoint are not native-authoritative"
        elif name in {"action_cache", "policy_row_mappings", "rng_state_indices"}:
            live_authority = "not_live_native"
            live_probe = "blocked_by_python_policy_orchestration"
            blocker = "CXXBACKBONE5 native policy envelope not implemented for this surface"
        else:
            live_authority = "not_live_native"
            live_probe = "not_attempted_or_blocked_in_cxxbackbone5_slice"
            blocker = "requires later environment/lifecycle/output/checkpoint/multicell composition"
        rows.append(
            SurfaceLedgerRow(
                surface_name=name,
                historical_contract_exact=True,
                current_live_probe_status=live_probe,
                current_live_authority=live_authority,
                checkpoint_continuation=(
                    "focused_policy_boundary_exact" if name in {"mlp_gru_policy_banks", "hmem"} else "not_proven_live"
                ),
                public_output_parity="not_applicable" if name not in {"public_output_compatibility"} else "not_proven_live",
                cpu_parity="exact_for_cxxbackbone4_policy_math" if name in {"mlp_gru_policy_banks", "hmem"} else "not_retested",
                cuda_parity="exact_for_cxxbackbone4_policy_math" if name in {"mlp_gru_policy_banks", "hmem"} else "not_retested",
                whole_run_measurement="measured_as_current_python_orchestrated_lane_only",
                current_blocker=blocker,
            )
        )
    return tuple(rows)


def master_surface_ledger_payload(*, head: str = "", timing_digest: str = "") -> dict[str, Any]:
    rows = [asdict(row) for row in master_surface_ledger_rows()]
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "head": str(head),
        "historical_contract_exact_count": sum(1 for row in rows if row["historical_contract_exact"]),
        "surface_count": len(rows),
        "current_live_authority_count": sum(
            1
            for row in rows
            if row["current_live_authority"]
            in {"inherited_live_integrated_from_runnercore", "default_off_cxxbackbone3_4_aten_math_only"}
        ),
        "full_native_policy_envelope_live": False,
        "timing_digest": str(timing_digest),
        "rows": rows,
    }


def phase_attempt_ledger_payload(*, head: str = "") -> dict[str, Any]:
    attempts = [
        ("phase0_live_boundary_and_cost_map", "completed", ""),
        ("phase1_unified_native_policy_service", "blocked", "implemented math exists, but native cache/obs/grouping/row-map service is absent"),
        ("phase2_resident_cuda_policy_state", "blocked", "CUDA policy banks/Hmem can execute, but obs/actions/cache are copied per refill"),
        ("phase3_exact_policy_service_parity", "blocked", "no complete policy service to run the required matrix"),
        ("phase4_real_run_ticks_policy_profile", "partial", "real _run_ticks timing measured for current Python-orchestrated CXXBACKBONE4 lane"),
        ("phase5_policy_whole_run_performance", "partial", "same-commit current-lane disabled/enabled timing measured where harness completes"),
        ("phase6_patch_wdl_jungle_environment_composition", "blocked", "environment authority remains Python/env-owned"),
        ("phase7_lifecycle_reproduction_mutation_composition", "blocked", "lifecycle authority remains Python-owned"),
        ("phase8_typed_events_public_output_composition", "blocked", "event/output authority remains Python-owned"),
        ("phase9_full_resident_checkpoint_authority", "blocked", "focused policy checkpoint exists; full resident checkpoint is absent"),
        ("phase10_full_18_surface_single_cell_composition", "blocked", "required live authority groups are absent"),
        ("phase11_long_continuation", "blocked", "requires phase10 live composition first"),
        ("phase12_real_multi_cell_campaign_composition", "blocked", "multi-cell native generation call is absent"),
        ("phase13_complete_before_after", "blocked", "requires accepted live policy/environment/lifecycle/output composition first"),
    ]
    return {
        "schema_version": PHASE_SCHEMA_VERSION,
        "head": str(head),
        "decision": "CXXBACKBONE5-I/PARTIAL-AUTONOMOUS-CLOSEOUT",
        "attempts": [
            {"phase": phase, "status": status, "blocker": blocker}
            for phase, status, blocker in attempts
        ],
    }


def counter_snapshot_payload(
    *,
    buckets: Mapping[str, Any],
    device: str,
    population: int,
    grid: int,
    steps: int,
    generations: int,
    enabled: bool,
) -> dict[str, Any]:
    numeric = {str(k): float(v) for k, v in buckets.items() if isinstance(v, (int, float))}
    prefix = "cxxbackbone5_"
    cxx = {k: v for k, v in numeric.items() if k.startswith(prefix)}
    return {
        "schema_version": COUNTER_SCHEMA_VERSION,
        "device": str(device),
        "population": int(population),
        "grid": int(grid),
        "steps": int(steps),
        "generations": int(generations),
        "cxxbackbone4_policy_math_enabled": bool(enabled),
        "cxxbackbone5_counters": cxx,
        "runner_buckets": numeric,
        "python_policy_orchestration_calls": int(cxx.get("cxxbackbone5_policy_python_entry_calls", 0.0)),
        "python_observation_callbacks": int(cxx.get("cxxbackbone5_observation_python_callbacks", 0.0)),
        "h2d_bytes": int(cxx.get("cxxbackbone5_h2d_bytes", 0.0)),
        "d2h_bytes": int(cxx.get("cxxbackbone5_d2h_bytes", 0.0)),
        "cuda_launches": "not available from Python bucket instrumentation",
        "gpu_active_fraction": "not available from Python bucket instrumentation",
        "hidden_fallback": False,
        "full_native_policy_window": False,
    }


def tensor_device_name() -> str:
    if torch.cuda.is_available():
        try:
            return str(torch.cuda.get_device_name(0))
        except Exception:
            return "cuda_available_unknown_device"
    return "cuda_unavailable"


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_payload(paths: Iterable[Path]) -> dict[str, Any]:
    rows = []
    for path in paths:
        if path.exists():
            rows.append({"path": str(path), "sha256": digest_file(path), "bytes": int(path.stat().st_size)})
    return {"schema_version": "cxxbackbone5_evidence_manifest_v1", "files": rows}
