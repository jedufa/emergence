from __future__ import annotations

"""simulation/metrics_pkg/epi_av_timeseries.py

Phase 5a follow-up: optional human-friendly EPI/AV visibility.

Provides:
- A deterministic per-generation time series CSV writer for EPI + AV
- Stable float formatting (rounded to 1e-8) for reproducible logs

Design goals
------------
- Does NOT touch determinism-critical artifacts (metrics.csv, behaviors.csv)
- Best-effort atomicity:
    - Writes are appended in a single line and fsynced.
    - This prevents partial-line reads in normal conditions.
- No RNG usage.

File format
-----------
epi_av_timeseries.csv

Header:
generation,epi_value,av_value

Rows:
- generation: int
- epi_value: float (rounded to 8 decimals)
- av_value: float (rounded to 8 decimals) or empty if undefined/null
"""

import os
from dataclasses import dataclass
from typing import Optional


def _fmt_f(x: Optional[float]) -> str:
    """Format float with a stable policy (rounded to 1e-8).

    Notes
    -----
    - Uses fixed-point with 8 decimals for reproducibility.
    - Empty string represents null/undefined values.
    """
    if x is None:
        return ""
    try:
        v = float(x)
    except Exception:
        return ""
    # Round then format in fixed decimals.
    v = round(v, 8)
    return f"{v:.8f}"




def _read_last_gen_from_csv_tail(path: str) -> int:
    """Return last generation in a generation-indexed CSV, or -1."""
    try:
        if not path or (not os.path.exists(path)):
            return -1
        with open(path, "rb") as f:
            try:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                tail = min(size, 64 * 1024)
                f.seek(size - tail)
            except Exception:
                f.seek(0)
            data = f.read().decode("utf-8", errors="ignore")
        lines = [ln.strip() for ln in data.splitlines() if ln.strip()]
        for ln in reversed(lines):
            if ln.lower().startswith("generation,"):
                continue
            parts = ln.split(",")
            if not parts:
                continue
            try:
                return int(parts[0])
            except Exception:
                continue
        return -1
    except Exception:
        return -1

@dataclass
class EPITimeSeriesWriter:
    """Append-only CSV writer for EPI/AV time series."""

    run_dir: str
    filename: str = "epi_av_timeseries.csv"
    _last_written_gen: int = -1

    def path(self) -> str:
        return os.path.join(self.run_dir, self.filename)

    def ensure_header(self) -> None:
        """Create file with header if it does not exist."""
        p = self.path()
        if os.path.exists(p):
            return
        os.makedirs(self.run_dir, exist_ok=True)
        header = "generation,epi_value,av_value\n"
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(header)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)

    def append_row(self, *, generation: int, epi_value: float, av_value: Optional[float]) -> None:
        """Append one row for a generation (fsync for durability).

        Append-safe for branch/continuation: if the file already contains history,
        skip any write for generation <= last_written_gen.
        """
        self.ensure_header()
        p = self.path()

        # Lazy init of last_written_gen from existing history.
        if self._last_written_gen < 0 and os.path.exists(p):
            self._last_written_gen = _read_last_gen_from_csv_tail(p)

        if int(generation) <= int(self._last_written_gen):
            return

        line = f"{int(generation)},{_fmt_f(epi_value)},{_fmt_f(av_value)}\n"
        with open(p, "a", encoding="utf-8", newline="\n") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        self._last_written_gen = int(generation)
