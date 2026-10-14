"""Source-bound fail-closed audit for composed C1+C2+C3+C4 residency.

The accepted R10 driver proves five B48/C32 transaction boundaries, but its
Python owner snapshots state and materializes status each tick.  It is not a
resident R11 owner.  This audit records the exact current source blockers and
deliberately provides no sequential-authority fallback.
"""

from __future__ import annotations

import ast
from hashlib import sha256
from pathlib import Path
from typing import Final

from .contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[2]
R10_ADAPTER = ROOT / "engine/tensor_world/c4_resident_multitick_r10.py"
R9_ADAPTER = ROOT / "engine/tensor_world/c4_resident_population_r9.py"
R9_KERNEL = ROOT / "engine/tensor_world/native/tensor_world_c4_resident_population_r9.cu"
C4_STATE = ROOT / "engine/tensor_world/c4_state.py"
C3_NATIVE = ROOT / "engine/tensor_world/c3_native.py"
C3_KERNEL = ROOT / "engine/perf/cuda_architecture_b_persistent_generation_kernel.cu"
C1_STATE = ROOT / "engine/tensor_world/state.py"
C5_REPLAY = ROOT / "engine/tensor_world/c5_determinism.py"

C4_RESIDENT_COMPOSITION_R11_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_composition_r11_audit_v1"
)
C4_RESIDENT_COMPOSITION_R11_DECISION_V1: Final[str] = "NO_GATE_DECISION"


def _function(path: Path, symbol: str) -> tuple[ast.FunctionDef, str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == symbol:
            if node.end_lineno is None:
                raise TensorWorldContractError(f"C4 R11 cannot bound {symbol}")
            lines = source.splitlines()
            return node, "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise TensorWorldContractError(f"C4 R11 source symbol is missing: {symbol}")


def _call_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def audit_c4_resident_composition_r11_v1(
    repo_root: Path | str = ROOT,
) -> dict[str, object]:
    """Audit the accepted R10 and current C3/C4 owners without executing them."""

    root = Path(repo_root).resolve()
    paths = {
        "r10_adapter": root / R10_ADAPTER.relative_to(ROOT),
        "r9_adapter": root / R9_ADAPTER.relative_to(ROOT),
        "r9_kernel": root / R9_KERNEL.relative_to(ROOT),
        "c4_state_owner": root / C4_STATE.relative_to(ROOT),
        "c3_persistent_owner": root / C3_NATIVE.relative_to(ROOT),
        "c3_kernel": root / C3_KERNEL.relative_to(ROOT),
        "c1_state_owner": root / C1_STATE.relative_to(ROOT),
        "c5_replay_owner": root / C5_REPLAY.relative_to(ROOT),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise TensorWorldContractError(f"C4 R11 audit sources are missing: {missing}")

    r10_run_node, r10_run = _function(
        paths["r10_adapter"], "run_c4_resident_multitick_r10_cuda_v1"
    )
    r10_helpers = "\n".join(
        _function(paths["r10_adapter"], symbol)[1]
        for symbol in (
            "_snapshot_tensor_leaves_v1",
            "_capture_boundary_v1",
            "_restore_boundary_v1",
            "_prepare_tick_v1",
            "_tick_receipt_v1",
        )
    )
    r10_boundary_source = r10_run + r10_helpers
    boundary_tokens = (
        "detach().clone()",
        ".cpu()",
        ".tolist()",
        "torch.cuda.synchronize",
    )
    r10_boundary_hits = [token for token in boundary_tokens if token in r10_boundary_source]
    r10_tick_loops = sum(
        isinstance(node, ast.For)
        and "tick" in ast.unparse(node.target).lower()
        for node in ast.walk(r10_run_node)
    )

    c4_source = paths["c4_state_owner"].read_text(encoding="utf-8")
    c3_source = paths["c3_persistent_owner"].read_text(encoding="utf-8")
    c1_source = paths["c1_state_owner"].read_text(encoding="utf-8")
    c5_source = paths["c5_replay_owner"].read_text(encoding="utf-8")
    required_fragments = {
        "c4_outside_c3_graph": "population mutations remain outside the C3 tick graph",
        "c1_position_i64": '("agents.pos_y", self.agents.pos_y, slot_shape, torch.int64)',
        "c3_position_i32_copy": (
            '"pos_y": _owned(source.agents.pos_y, target, dtype=torch.int32)'
        ),
        "c3_death_output_owner": 'output("death_agent_ids"',
        "c3_event_output_owner": 'event_tensors["active_count"] = _zeros',
        "c3_runner_rng_owner": '"runner_pcg64_state_words": runner_state',
        "c5_stream_rng_owner": "rng_state_words must have shape [batch, streams, words]",
    }
    owners = {
        "c4_outside_c3_graph": c4_source,
        "c1_position_i64": c1_source,
        "c3_position_i32_copy": c3_source,
        "c3_death_output_owner": c3_source,
        "c3_event_output_owner": c3_source,
        "c3_runner_rng_owner": c3_source,
        "c5_stream_rng_owner": c5_source,
    }
    absent = [
        name for name, fragment in required_fragments.items() if fragment not in owners[name]
    ]
    if absent:
        raise TensorWorldContractError(f"C4 R11 blocker source drifted: {absent}")

    composed_callsites: list[str] = []
    for path in (root / "engine").rglob("*.py"):
        if path.resolve() == Path(__file__).resolve():
            continue
        calls = _call_names(path)
        if {
            "execute_c3_persistent_native_v1",
            "execute_c4_resident_population_r9_cuda_v1",
        }.issubset(calls):
            composed_callsites.append(path.relative_to(root).as_posix())

    return {
        "schema_version": C4_RESIDENT_COMPOSITION_R11_SCHEMA_VERSION_V1,
        "decision": C4_RESIDENT_COMPOSITION_R11_DECISION_V1,
        "status": "FAIL_CLOSED_COMPOSITION_UNAVAILABLE",
        "accepted_r10_route": "R10_B48_C32_FIVE_TICK_R9_DRIVER",
        "accepted_r10_residency_classification": "C_host_dependent",
        "r10_tick_python_loops": r10_tick_loops,
        "r10_host_boundary_hits": r10_boundary_hits,
        "r10_zero_boundary_eligible": False,
        "composed_callsites": sorted(composed_callsites),
        "blockers": [
            {
                "code": "R10_HOST_STATUS_AND_ROLLBACK_AUTHORITY",
                "detail": "accepted R10 snapshots, materializes status, and synchronizes on failure",
            },
            {
                "code": "C4_OUTSIDE_C3_TICK_GRAPH",
                "detail": "V4 source explicitly leaves population mutation outside C3 graph",
            },
            {
                "code": "C3_POSITION_AUTHORITY_IS_A_DTYPE_COPY",
                "detail": "C1/V4 int64 positions are projected into independent C3 int32 tensors",
            },
            {
                "code": "C3_LIFECYCLE_AND_EVENT_OUTPUTS_NOT_BOUND_TO_V4",
                "detail": "C3 death/event outputs have no resident device projection into V4 owners",
            },
            {
                "code": "C3_C5_RNG_CONTINUATION_UNMAPPED",
                "detail": "C3 named PCG owners are not mapped to C5 stream cursors/counters used by R9",
            },
            {
                "code": "NO_COMPOSED_NATIVE_OR_GRAPH_CALLSITE",
                "detail": "no engine callsite invokes C3 persistent native and R9 C4 together",
            },
        ],
        "sequential_host_fallback": "forbidden_not_implemented",
        "runtime_boundary_counters": "not_measured_for_nonexistent_composed_route",
        "gate_claim": False,
        "sources": {
            name: {
                "path": str(path.relative_to(root)),
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in paths.items()
        },
    }


__all__ = [
    "C4_RESIDENT_COMPOSITION_R11_DECISION_V1",
    "C4_RESIDENT_COMPOSITION_R11_SCHEMA_VERSION_V1",
    "audit_c4_resident_composition_r11_v1",
]
