from __future__ import annotations

"""simulation/runner_pkg/perf_writer.py

Lightweight performance bucket writer.

Why this exists
--------------
The runner already maintains an in-memory `buckets` dict (seconds spent in
named regions). That is useful during interactive debugging, but it is painful
to compare runs across code changes.

This module writes a stable, append-only CSV in *long* format:

    gen,total_s,bucket,seconds

The long format avoids the trap of having columns that appear/disappear as
instrumentation changes.

Important
---------
- Only used when `--perf` is enabled.
- Never changes simulation semantics.
"""

from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Dict, Optional


@dataclass
class PerfBucketWriter:
    """Append-only writer for per-generation perf buckets."""

    run_dir: str
    filename: str = "perf_buckets.csv"

    def __post_init__(self) -> None:
        self._path = Path(self.run_dir) / self.filename
        self._initialized = False

    def _ensure_header(self) -> None:
        if self._initialized:
            return
        # Create parent if needed
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            with self._path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["gen", "total_s", "bucket", "seconds"])
        self._initialized = True

    def write_generation(self, gen: int, total_s: float, buckets: Optional[Dict[str, float]]) -> None:
        """Write one generation worth of bucket timings.

        Args:
            gen: Generation index.
            total_s: Total wall time for the generation.
            buckets: Dict[str, float] mapping bucket name -> seconds.
        """
        self._ensure_header()
        b = buckets or {}
        with self._path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for k, v in sorted(b.items()):
                try:
                    w.writerow([int(gen), float(total_s), str(k), float(v)])
                except Exception:
                    continue
