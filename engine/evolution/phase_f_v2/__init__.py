"""Planning-only Phase F v2 matrix contracts."""

from .matrix import (
    AUTHORIZED_BASE_COMMIT,
    CASE_COUNT,
    CHECKPOINT_CASE_COUNT,
    EXPECTED_ACCELERATED_INVOCATIONS,
    EXPECTED_CPU_ORACLE_EVALUATIONS,
    EXPECTED_TOTAL_REQUIRED_EVALUATIONS,
    MATRIX_ID,
    SCHEMA_VERSION,
    PhaseFV2MatrixError,
    build_phase_f_v2_matrix,
    canonical_json_bytes,
    canonical_sha256,
    load_phase_f_v2_matrix,
    validate_phase_f_v2_matrix,
)

__all__ = [
    "AUTHORIZED_BASE_COMMIT",
    "CASE_COUNT",
    "CHECKPOINT_CASE_COUNT",
    "EXPECTED_ACCELERATED_INVOCATIONS",
    "EXPECTED_CPU_ORACLE_EVALUATIONS",
    "EXPECTED_TOTAL_REQUIRED_EVALUATIONS",
    "MATRIX_ID",
    "SCHEMA_VERSION",
    "PhaseFV2MatrixError",
    "build_phase_f_v2_matrix",
    "canonical_json_bytes",
    "canonical_sha256",
    "load_phase_f_v2_matrix",
    "validate_phase_f_v2_matrix",
]
