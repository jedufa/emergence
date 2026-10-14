"""Campaign-owned integration surface for CUDA Full Evolution V2.

All Campaign consumers reach the shape-specialized CUDA implementation through
this one explicit boundary.  The module deliberately exports module handles:
the performance package continues to own kernels and runtime contracts, while
Campaign owns request, checkpoint, and execution composition.
"""

from __future__ import annotations

from engine.perf.cuda_full_evolution_v2 import (
    benchmark as benchmark_v2,
    contract as contract_v2,
    direct_sparse_runtime as direct_sparse_runtime_v2,
    layout as layout_v2,
    native_loader as native_loader_v2,
    runtime as runtime_v2,
    workhorse as workhorse_v2,
)


__all__ = [
    "benchmark_v2",
    "contract_v2",
    "direct_sparse_runtime_v2",
    "layout_v2",
    "native_loader_v2",
    "runtime_v2",
    "workhorse_v2",
]
