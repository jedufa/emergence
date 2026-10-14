"""EIEL evidence librarian V1R.

The librarian scans evidence as inert files only.  It does not import or
execute evidence payloads and it does not infer scientific or qualification
authority from filenames, tests, or nearby reports.
"""

from __future__ import annotations

from .core import (
    AuditConfig,
    LibrarianError,
    build_catalog,
    canonical_json_bytes,
    inspect_zip_safety,
    run_audit,
    validate_rel_path,
)

__all__ = [
    "AuditConfig",
    "LibrarianError",
    "build_catalog",
    "canonical_json_bytes",
    "inspect_zip_safety",
    "run_audit",
    "validate_rel_path",
]
