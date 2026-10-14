"""Static R9 residency classifier for the current C4 R1-R7 routes.

This is source evidence only.  Absence of a transfer/synchronization token is
not promoted to a measured zero-boundary receipt, and no C4 gate is implied.
"""

from __future__ import annotations

import ast
from hashlib import sha256
from pathlib import Path
from typing import Final

from .contracts import TensorWorldContractError


C4_RESIDENCY_R9_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_residency_r9_static_audit_v1"
)
C4_RESIDENCY_R9_DECISION_V1: Final[str] = "NO_GATE_DECISION"

_RUNG_SPECS: Final[tuple[dict[str, object], ...]] = (
    {
        "rung": "R1",
        "adapter": "engine/tensor_world/c4_resident_death.py",
        "execute": "execute_c4_resident_death_transaction_cuda_v1",
        "native": ("engine/tensor_world/native/tensor_world_c4_resident_death.cu",),
        "admission": "batched_death_component_only",
    },
    {
        "rung": "R2",
        "adapter": "engine/tensor_world/c4_resident_birth_r2.py",
        "execute": "execute_c4_resident_birth_r2_cuda_v1",
        "native": ("engine/tensor_world/native/tensor_world_c4_resident_birth_r2.cu",),
        "admission": "B1_ordinary_birth_component_only",
    },
    {
        "rung": "R3",
        "adapter": "engine/tensor_world/c4_resident_turnover_r3.py",
        "execute": "execute_c4_resident_turnover_r3_cuda_v1",
        "native": ("engine/tensor_world/native/tensor_world_c4_resident_turnover_r3.cu",),
        "admission": "B1_death_ordinary_birth_component_only",
    },
    {
        "rung": "R4",
        "adapter": "engine/tensor_world/c4_resident_mutation_r4.py",
        "execute": "execute_c4_resident_mutation_r4_cuda_v1",
        "native": ("engine/tensor_world/native/tensor_world_c4_resident_mutation_r4.cu",),
        "admission": "B1_mate_mutation_component_only",
    },
    {
        "rung": "R5",
        "adapter": "engine/tensor_world/c4_resident_population_r5.py",
        "execute": "execute_c4_resident_population_r5_cuda_v1",
        "native": ("engine/tensor_world/native/tensor_world_c4_resident_population_r5.cu",),
        "admission": "B4_complete_one_transaction_component",
    },
    {
        "rung": "R6",
        "adapter": "engine/tensor_world/c4_resident_preflight_r6.py",
        "execute": "execute_c4_resident_preflight_r6_cuda_v1",
        "native": ("engine/tensor_world/native/tensor_world_c4_resident_preflight_r6.cu",),
        "admission": "B4_complete_one_transaction_with_all_live_preflight",
    },
    {
        "rung": "R7",
        "adapter": "engine/tensor_world/c4_resident_turnover_r7_native.py",
        "execute": "execute_c4_resident_turnover_r7_cuda_tick_v1",
        "native": ("engine/tensor_world/native/tensor_world_c4_resident_population_r5.cu",),
        "admission": "B4_C32_host_launched_five_tick_test_harness",
    },
    {
        "rung": "R8",
        "adapter": "engine/tensor_world/c4_resident_maxload_r8_native.py",
        "execute": "execute_c4_resident_maxload_r8_cuda_v1",
        "native": ("engine/tensor_world/native/tensor_world_c4_resident_maxload_r8.cu",),
        "admission": "B4_C32_native_exact_8_death_8_birth_maxload_probe",
    },
    {
        "rung": "R9",
        "adapter": "engine/tensor_world/c4_resident_population_r9.py",
        "execute": "execute_c4_resident_population_r9_cuda_v1",
        "native": ("engine/tensor_world/native/tensor_world_c4_resident_population_r9.cu",),
        "admission": "B48_native_one_tick_dynamic_batch_probe",
    },
)

_TRANSFER_TOKENS: Final[tuple[str, ...]] = (
    ".cpu(", ".cuda(", ".to(", "cudaMemcpy", "cudaMemcpyAsync"
)
_SYNC_TOKENS: Final[tuple[str, ...]] = (
    ".item(", ".tolist(", "torch.cuda.synchronize", "cudaDeviceSynchronize",
    "cudaStreamSynchronize",
)
_ALLOCATION_TOKENS: Final[tuple[str, ...]] = (
    "torch.empty(", "torch.zeros(", "torch.full(", "cudaMalloc", "cudaFree",
)
_GRAPH_TOKENS: Final[tuple[str, ...]] = (
    "torch.cuda.CUDAGraph", "cudaGraph", "graph_capture", "capture_begin",
)


def _function_source(path: Path, symbol: str) -> tuple[str, int, int]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol:
            if node.end_lineno is None:
                raise TensorWorldContractError(f"C4 R9 cannot bound {symbol}")
            lines = source.splitlines()
            loops = [
                child for child in ast.walk(node)
                if isinstance(child, (ast.For, ast.AsyncFor))
            ]
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
    raise TensorWorldContractError(f"C4 R9 execute symbol is missing: {symbol}")


def _token_hits(source: str, tokens: tuple[str, ...]) -> list[str]:
    return [token for token in tokens if token in source]


def _production_callsites(root: Path, symbols: tuple[str, ...]) -> list[str]:
    callsites: list[str] = []
    engine = root / "engine"
    for path in engine.rglob("*.py"):
        relative = path.relative_to(root).as_posix()
        if (
            path.name == "__init__.py"
            or path.name.startswith("c4_resident_")
            or path.name == "c4_residency_r9.py"
        ):
            continue
        source = path.read_text(encoding="utf-8")
        if any(symbol in source for symbol in symbols):
            callsites.append(relative)
    return sorted(callsites)


def audit_c4_residency_r9_static_v1(
    repo_root: Path | str = ".",
) -> dict[str, object]:
    """Classify source-visible residency properties without runtime promotion."""

    root = Path(repo_root).resolve()
    rung_rows: list[dict[str, object]] = []
    execute_symbols: list[str] = []
    all_sources: dict[str, str] = {}
    for spec in _RUNG_SPECS:
        adapter_relative = str(spec["adapter"])
        execute_symbol = str(spec["execute"])
        adapter = root / adapter_relative
        if not adapter.is_file():
            raise TensorWorldContractError(f"C4 R9 adapter is missing: {adapter_relative}")
        execute_source, python_loop_count, per_world_loop_count = _function_source(
            adapter, execute_symbol
        )
        execute_symbols.append(execute_symbol)
        all_sources[adapter_relative] = sha256(adapter.read_bytes()).hexdigest()
        native_text = ""
        for native_relative in spec["native"]:
            native = root / str(native_relative)
            if not native.is_file():
                raise TensorWorldContractError(
                    f"C4 R9 native source is missing: {native_relative}"
                )
            all_sources[str(native_relative)] = sha256(native.read_bytes()).hexdigest()
            native_text += native.read_text(encoding="utf-8") + "\n"
        transfer_hits = sorted(set(
            _token_hits(execute_source, _TRANSFER_TOKENS)
            + _token_hits(native_text, _TRANSFER_TOKENS)
        ))
        sync_hits = sorted(set(
            _token_hits(execute_source, _SYNC_TOKENS)
            + _token_hits(native_text, _SYNC_TOKENS)
        ))
        allocation_hits = sorted(set(
            _token_hits(execute_source, _ALLOCATION_TOKENS)
            + _token_hits(native_text, _ALLOCATION_TOKENS)
        ))
        graph_hits = sorted(set(
            _token_hits(execute_source, _GRAPH_TOKENS)
            + _token_hits(native_text, _GRAPH_TOKENS)
        ))
        rung_rows.append({
            "rung": spec["rung"],
            "adapter": adapter_relative,
            "execute_symbol": execute_symbol,
            "native_sources": list(spec["native"]),
            "admission": spec["admission"],
            "execute_python_loop_count": python_loop_count,
            "execute_per_world_python_loop_count": per_world_loop_count,
            "source_transfer_hits": transfer_hits,
            "source_sync_or_materialization_hits": sync_hits,
            "source_hot_allocation_hits": allocation_hits,
            "source_cuda_graph_hits": graph_hits,
            "source_classification": (
                "NO_FORBIDDEN_BOUNDARY_TOKEN_OBSERVED"
                if not transfer_hits and not sync_hits and not allocation_hits
                else "SOURCE_REVIEW_REQUIRED"
            ),
        })

    callsites = _production_callsites(root, tuple(execute_symbols))
    return {
        "schema_version": C4_RESIDENCY_R9_SCHEMA_VERSION_V1,
        "decision": C4_RESIDENCY_R9_DECISION_V1,
        "status": "STATIC_RESIDENCY_BLOCKERS_OPEN",
        "scope": "C4_R1_R9_ADAPTER_AND_NATIVE_SOURCE_ONLY",
        "rungs": rung_rows,
        "source_hashes": dict(sorted(all_sources.items())),
        "production_composed_callsites": callsites,
        "blockers": [
            {
                "code": "C1_C2_C3_C4_COMPOSED_TICK_NOT_WIRED",
                "detail": "no production engine callsite owns a composed C1+C2+C3+C4 resident tick",
            },
            {
                "code": "CUDA_GRAPH_CLASSIFICATION_UNPROVEN",
                "detail": "no audited execute/native source binds graph capture or replay",
            },
            {
                "code": "R7_HOST_TICK_OWNERSHIP",
                "detail": (
                    "R7 is explicitly a host-launched test harness with Python tick_index "
                    "and one adapter invocation per tick; R8/R9 remain host-launched probes"
                ),
            },
            {
                "code": "STEADY_STATE_COUNTERS_UNMEASURED",
                "detail": (
                    "H2D, D2H, callbacks, synchronizations, materializations, allocator "
                    "calls, and Python-loop counts have no runtime receipt"
                ),
            },
            {
                "code": "PREALLOCATED_STAGING_LIFETIME_UNBOUND",
                "detail": (
                    "staging factories allocate device tensors outside execute functions, "
                    "but no production owner proves one-time allocation and reuse"
                ),
            },
        ],
        "interpretation": (
            "clean source scans are component evidence only and cannot be reported as "
            "zero measured steady-state boundaries"
        ),
        "gate_claim": False,
    }


__all__ = [
    "C4_RESIDENCY_R9_DECISION_V1",
    "C4_RESIDENCY_R9_SCHEMA_VERSION_V1",
    "audit_c4_residency_r9_static_v1",
]
