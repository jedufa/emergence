"""Optional Assignment Linker and worker scheduler for EIEL.

Importing the normal operator managers never imports this package.  Removing
``tools/scheduler`` disables unattended execution without changing repository
truth or manual workflows.
"""

from .contracts import (
    ASSIGNMENT_SCHEMA_VERSION,
    DISPATCH_SCHEMA_VERSION,
    LEASE_SCHEMA_VERSION,
    WORKER_RESULT_SCHEMA_VERSION,
)

__all__ = [
    "ASSIGNMENT_SCHEMA_VERSION",
    "DISPATCH_SCHEMA_VERSION",
    "LEASE_SCHEMA_VERSION",
    "WORKER_RESULT_SCHEMA_VERSION",
]
