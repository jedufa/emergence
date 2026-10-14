"""simulation/viz_pkg

Phase 5d: Engine-side observability artifacts for the Web UI.

This package is intentionally *not* part of the determinism substrate.
Everything written here goes under <RUN_DIR>/viz/.

Rules
-----
- Write-only by engine/orchestrator; UI is read-only.
- Must not modify determinism-critical artifacts (metrics.csv, behaviors.csv,
  wal.jsonl, checkpoint bundles, SafeBoundary ordering, RNG usage).
"""

from .viz_writer import VizWriter, VizConfig, get_viz_writer

__all__ = ["VizWriter", "VizConfig", "get_viz_writer"]
