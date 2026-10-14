"""Read-only evidence tooling for campaign decision support."""

from __future__ import annotations

FORBIDDEN_SIMULATION_PATH_PREFIXES: tuple[str, ...] = (
    "engine/",
    "engine/simulation/",
    "engine/agents/",
    "engine/evolution/",
)

# Compatibility export: no machine-specific default. Consumers must configure a root.
SHARED_EVIDENCE_ROOT: str | None = None

# Evidence modules are readers by default. Tests assert this stays empty unless a
# future writer declares an explicit, reviewed evidence-only target.
WRITABLE_TARGET_PATHS: tuple[str, ...] = ()

__all__ = [
    "FORBIDDEN_SIMULATION_PATH_PREFIXES",
    "SHARED_EVIDENCE_ROOT",
    "WRITABLE_TARGET_PATHS",
]
