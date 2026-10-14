"""Current-source residency audit for the composed R11 owner.

This is an evidence companion, not a gate decision.  It distinguishes the
class-B device-local owner from the older host-owned R7--R10 harnesses while
keeping the final runtime/resource/performance obligations explicit.
"""

from __future__ import annotations

import ast
from hashlib import sha256
from pathlib import Path
from typing import Final

from .contracts import TensorWorldContractError


C4_RESIDENCY_R11_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_residency_r11_component_audit_v1"
)
C4_RESIDENCY_R11_DECISION_V1: Final[str] = "NO_GATE_DECISION"
_EXECUTE_SYMBOL: Final[str] = "execute_c4_resident_composed_r11_cuda_v1"
_ADAPTER_RELATIVE: Final[str] = "engine/tensor_world/c4_resident_composed_r11.py"
_R9_ADAPTER_RELATIVE: Final[str] = "engine/tensor_world/c4_resident_population_r9.py"
_R9_NATIVE_RELATIVE: Final[str] = (
    "engine/tensor_world/native/tensor_world_c4_resident_population_r9.cu"
)
_R11_NATIVE_RELATIVE: Final[str] = (
    "engine/tensor_world/native/tensor_world_c4_resident_graph_r11.cu"
)

_TRANSFER_TOKENS: Final[tuple[str, ...]] = (
    ".cpu(",
    ".cuda(",
    ".to(",
    "cudaMemcpy",
    "cudaMemcpyAsync",
)
_SYNC_TOKENS: Final[tuple[str, ...]] = (
    ".item(",
    ".tolist(",
    "torch.cuda.synchronize",
    "cudaDeviceSynchronize",
    "cudaStreamSynchronize",
)
_ALLOCATOR_TOKENS: Final[tuple[str, ...]] = (
    "torch.empty(",
    "torch.zeros(",
    "torch.full(",
    "cudaMalloc",
    "cudaFree",
)


def _execute_source(path: Path) -> tuple[str, int, int]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == _EXECUTE_SYMBOL:
            if node.end_lineno is None:
                raise TensorWorldContractError("R11 execute source has no end line")
            lines = source.splitlines()
            loops = [child for child in ast.walk(node) if isinstance(child, ast.For)]
            per_world = sum(
                "world" in ast.unparse(loop.target).lower()
                or "bundle.batch" in ast.unparse(loop.iter)
                for loop in loops
            )
            return (
                "\n".join(lines[node.lineno - 1 : node.end_lineno]),
                len(loops),
                per_world,
            )
    raise TensorWorldContractError(f"R11 execute symbol is missing: {_EXECUTE_SYMBOL}")


def _hits(source: str, tokens: tuple[str, ...]) -> list[str]:
    return sorted({token for token in tokens if token in source})


def audit_c4_residency_r11_static_v1(repo_root: Path | str = ".") -> dict[str, object]:
    """Classify the source-visible R11 boundary without promoting a gate."""

    root = Path(repo_root).resolve()
    adapter = root / _ADAPTER_RELATIVE
    r9_adapter = root / _R9_ADAPTER_RELATIVE
    r9_native = root / _R9_NATIVE_RELATIVE
    r11_native = root / _R11_NATIVE_RELATIVE
    for path in (adapter, r9_adapter, r9_native, r11_native):
        if not path.is_file():
            raise TensorWorldContractError(f"R11 residency source is missing: {path}")
    execute_source, loop_count, per_world_loop_count = _execute_source(adapter)
    # The R9 adapter contains the one-time caller-owned staging constructor.
    # Including that constructor in a steady-state allocator scan would report
    # false hot-path churn (the staging allocation is deliberately outside
    # ``execute_c4_resident_composed_r11_cuda_v1``).  Keep the boundary scan
    # focused on the execute body and the native transaction source.
    native_source = "\n".join(
        path.read_text(encoding="utf-8") for path in (r9_native, r11_native)
    )
    all_sources = {
        relative: sha256((root / relative).read_bytes()).hexdigest()
        for relative in (
            _ADAPTER_RELATIVE,
            _R9_ADAPTER_RELATIVE,
            _R9_NATIVE_RELATIVE,
            _R11_NATIVE_RELATIVE,
        )
    }
    transfer_hits = _hits(execute_source + native_source, _TRANSFER_TOKENS)
    sync_hits = _hits(execute_source + native_source, _SYNC_TOKENS)
    allocator_hits = _hits(execute_source + native_source, _ALLOCATOR_TOKENS)
    return {
        "schema_version": C4_RESIDENCY_R11_SCHEMA_VERSION_V1,
        "decision": C4_RESIDENCY_R11_DECISION_V1,
        "status": "R11_COMPONENT_RESIDENCY_READY_GATE_EVIDENCE_OPEN",
        "route": "R11_B48_C32_COMPOSED_RESIDENT_REPAIR_SEAM",
        "execution_classification": "B_FIXED_TOPOLOGY_CUDA_GRAPH_OWNER_WITH_DEVICE_TIME",
        "execute_symbol": _EXECUTE_SYMBOL,
        "execute_python_loop_count": loop_count,
        "execute_per_world_python_loop_count": per_world_loop_count,
        "source_transfer_hits": transfer_hits,
        "source_sync_or_materialization_hits": sync_hits,
        "source_hot_allocator_hits": allocator_hits,
        "persistent_staging": True,
        "normal_tick_status_d2h": False,
        "normal_tick_snapshot_materialization": False,
        "normal_tick_host_synchronization": False,
        "bounded_host_launch": True,
        "fixed_topology_graph": True,
        "source_hashes": dict(sorted(all_sources.items())),
        "runtime_receipt_required": True,
        "gate_claim": False,
        "blockers": [
            {
                "code": "R11_RUNTIME_COUNTERS_NOT_GATE_COMPLETE",
                "detail": "profiler evidence must separate tool-induced measurement synchronization from the normal execute boundary",
            },
            {
                "code": "R11_FINAL_GEOMETRY_RESOURCE_OPEN",
                "detail": "the current component is B48/C32; final B48/W32/P256/C512 resource evidence is not bound",
            },
            {
                "code": "R11_GRAPH_WINDOW_OPEN",
                "detail": "the five-tick device-latched window exists; exact CPU/native parity and target trace evidence remain open",
            },
            {
                "code": "R11_LAUNCH_FAULT_ATOMICITY_UNCLAIMED",
                "detail": "semantic preflight rollback passes; runtime faults between guarded native launches remain unclaimed",
            },
        ],
    }


__all__ = [
    "C4_RESIDENCY_R11_DECISION_V1",
    "C4_RESIDENCY_R11_SCHEMA_VERSION_V1",
    "audit_c4_residency_r11_static_v1",
]
