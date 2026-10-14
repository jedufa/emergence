"""Versioned, declarative qualification-plan definitions."""

from .v1 import (
    DEFAULT_MAX_MATRIX_CASES,
    MATRIX_SCHEMA_VERSION,
    SCHEMA_VERSION,
    QualificationPlanError,
    compile_matrix,
    load_document,
    load_matrix,
    load_plan,
    validate_plan,
)

__all__ = [
    "DEFAULT_MAX_MATRIX_CASES",
    "MATRIX_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "QualificationPlanError",
    "compile_matrix",
    "load_document",
    "load_matrix",
    "load_plan",
    "validate_plan",
]
