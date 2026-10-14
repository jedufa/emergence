"""Isolated HAPB V2 CPU/code-prep scaffold.

Nothing in this package is a production route, a qualification claim, or a
replacement for the active Gate-6 V1 kernel.  Experimental GPU entrypoints
require an explicit non-production flag and are not registered on the HAPB
production artifact.
"""

from __future__ import annotations

HAPB_V2_EXPERIMENTAL_ENV_V1 = "EIEL_HAPB_V2_EXPERIMENTAL"
HAPB_V2_PACKAGE_SCHEMA_V1 = "eiel_hapb_v2_cpu_prep_package_v1"

__all__ = [
    "HAPB_V2_EXPERIMENTAL_ENV_V1",
    "HAPB_V2_PACKAGE_SCHEMA_V1",
]
