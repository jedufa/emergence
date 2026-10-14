from __future__ import annotations

"""
simulation/metrics_pkg/metrics_logger_io.py

CSV I/O helpers for MetricsLogger.

Keeps all file-writing behavior in one place so schema changes do not force
touching metric computation code.

Public functions:
- init_metrics_csv(path)
- append_metrics_row(path, row)

Determinism:
- Pure file I/O, no randomness.
"""

import csv
import io
import os
from simulation.metrics_pkg.rows import MetricsRow
from simulation.output_cost_timing import current_flat_output_cost_timing


def init_metrics_csv(path: str) -> None:
    """Initialize metrics.csv with header.

    Important: if the file already exists (e.g. a branched run carrying over
    history), do NOT truncate it.
    """
    if os.path.exists(path):
        return
    timing = current_flat_output_cost_timing()
    if timing is not None:
        with timing.serialization_phase():
            buffer = io.StringIO(newline="")
            csv.writer(buffer).writerow(list(MetricsRow.__annotations__.keys()))
            encoded = buffer.getvalue()
        with timing.write_phase():
            with open(path, "w", newline="", encoding="utf-8") as f:
                f.write(encoded)
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(MetricsRow.__annotations__.keys()))


def append_metrics_row(path: str, row: MetricsRow) -> None:
    """Append one MetricsRow to metrics.csv using the MetricsRow annotation order."""
    timing = current_flat_output_cost_timing()
    if timing is not None:
        with timing.serialization_phase():
            buffer = io.StringIO(newline="")
            csv.writer(buffer).writerow(
                [getattr(row, k) for k in MetricsRow.__annotations__.keys()]
            )
            encoded = buffer.getvalue()
        with timing.write_phase():
            with open(path, "a", newline="", encoding="utf-8") as f:
                f.write(encoded)
        return
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([getattr(row, k) for k in MetricsRow.__annotations__.keys()])
