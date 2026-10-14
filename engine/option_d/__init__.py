"""Standalone Option D resident RNG challenger.

This package is intentionally isolated from :mod:`engine.tensor_world`.  Its
public contract starts with an authoritative CPython MT19937 state bundle and
a bounded per-world draw schedule and returns generated material, immutable
successor state, and a hash-bound execution receipt.
"""

from .checkpoint import (
    CheckpointSnapshot,
    capture_checkpoint,
    checkpoint_bytes,
    load_checkpoint,
    restore_checkpoint,
    write_snapshot_atomic,
    write_checkpoint_atomic,
)
from .contracts import (
    CPYTHON_ALGORITHM_ID_V1,
    OPTION_D_API_SCHEMA_V1,
    OPTION_D_CHECKPOINT_SCHEMA_V1,
    CpythonRNGState,
    DrawPredicate,
    DrawOperation,
    DrawSchedule,
    ExecutionReceipt,
    GeneratedMaterial,
    GeneratedValue,
    GenerationResult,
    Operator,
    PredicateKind,
    OptionDContractError,
    OptionDExecutionError,
    RNGCounters,
    RNGStateBundle,
    WorldExecutionReceipt,
    float64_bits,
    float64_from_bits,
)
from .cpu import execute_cpu, state_from_seed
from .cuda import CudaResidentEngine, execute_cuda

__all__ = [
    "CPYTHON_ALGORITHM_ID_V1",
    "OPTION_D_API_SCHEMA_V1",
    "OPTION_D_CHECKPOINT_SCHEMA_V1",
    "CpythonRNGState",
    "CudaResidentEngine",
    "CheckpointSnapshot",
    "DrawPredicate",
    "DrawOperation",
    "DrawSchedule",
    "ExecutionReceipt",
    "GeneratedMaterial",
    "GeneratedValue",
    "GenerationResult",
    "Operator",
    "PredicateKind",
    "OptionDContractError",
    "OptionDExecutionError",
    "RNGCounters",
    "RNGStateBundle",
    "WorldExecutionReceipt",
    "checkpoint_bytes",
    "capture_checkpoint",
    "execute_cpu",
    "execute_cuda",
    "float64_bits",
    "float64_from_bits",
    "load_checkpoint",
    "restore_checkpoint",
    "state_from_seed",
    "write_checkpoint_atomic",
    "write_snapshot_atomic",
]
