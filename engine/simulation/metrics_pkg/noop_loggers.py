"""simulation/metrics_pkg/noop_loggers.py

No-op logger implementations used by *internal* tooling (e.g., shadow-clone metrics
evaluation) where we intentionally want:

- Zero filesystem writes
- Zero side effects on determinism-critical artifacts

These are deliberately tiny and permissive: they accept extra args/kwargs and
silently ignore them.

IMPORTANT:
- Do not import simulation core modules from here to avoid circular imports.
"""

from __future__ import annotations

from typing import Any, Dict


class NoOpMetricsLogger:
    """A drop-in replacement for MetricsLogger that performs no I/O."""

    def __init__(self, *_: Any, **__: Any) -> None:
        self.metrics_csv = ""

    def write_run_config(self, *_: Any, **__: Any) -> None:
        return

    def log_generation(self, *_: Any, **__: Any) -> None:
        return

    def flush(self) -> None:
        return

    def close(self) -> None:
        return


class NoOpBehaviorLogger:
    """A BehaviorLogger-compatible stub that records nothing."""

    def __init__(self, *_: Any, **__: Any) -> None:
        # Keep an attribute compatible with selection's fallback access pattern.
        self._data: Dict[int, Any] = {}

    def start_generation(self, *_: Any, **__: Any) -> None:
        return

    def register_agent(self, *_: Any, **__: Any) -> None:
        return

    def record_actions(self, *_: Any, **__: Any) -> None:
        return

    def set_outcomes(self, *_: Any, **__: Any) -> None:
        return

    def end_generation(self, *_: Any, **__: Any) -> None:
        return

    def injury_start_for(self, _agent_id: int, default: float = 0.0) -> float:
        return float(default)
