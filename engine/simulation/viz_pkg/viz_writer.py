from __future__ import annotations

"""simulation/viz_pkg/viz_writer.py

Phase 5d (UI prerequisites): run-local viz/ observability artifacts.

This module implements the DSIP v1 contract for "observability, not physics":

- All artifacts are written under <RUN_DIR>/viz/
- These files are NOT determinism-critical and MUST NOT be used for strict
  equivalence or fingerprinting.
- No RNG usage, no SafeBoundary reordering, no ControlPlane/WAL semantic changes.

Files
-----
All paths are relative to <RUN_DIR>/viz/

Required:
- run_status.json (atomic JSON)
- run_determinism_status.json (atomic JSON)
- checkpoints_index.json (atomic canonical JSON)
- epi_av_summary.csv (append-only CSV, created when EPI/AV computed)
- metrics_summary.csv (append-only CSV)
- applied_commands_log.jsonl (append-only JSONL)

Optional:
- behaviors_agg_summary.csv (append-only CSV)

Implementation notes
--------------------
- JSON writes are atomic via "*.tmp then os.replace".
- CSV/JSONL writes are append-only. We enforce newline discipline so a prior
  partial trailing line does not poison subsequent appends.
- These artifacts may include timestamps. They are explicitly non-determinism-
  critical and must never feed back into simulation decisions.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from simulation.output_cost_timing import current_flat_output_cost_timing


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


def _env_bool(key: str, default: str) -> bool:
    """Parse a boolean-ish env var.

    Accepts: 1/0, true/false, yes/no (case-insensitive).
    """
    raw = os.environ.get(key, default)
    s = str(raw).strip().lower()
    if s in ("", "0", "false", "no", "off"):
        return False
    return True


def _env_int(key: str, default: int) -> int:
    """Parse an int env var with a safe floor of 1."""
    raw = os.environ.get(key, str(default))
    try:
        v = int(str(raw).strip())
    except Exception:
        v = int(default)
    if v <= 0:
        v = 1
    return v


def _parse_csv_allowlist(raw: str) -> List[str]:
    """Parse comma-separated allowlists (e.g. "a,b,c")."""
    if raw is None:
        return []
    parts = [p.strip() for p in str(raw).split(",")]
    return [p for p in parts if p]


@dataclass(frozen=True)
class VizConfig:
    """Configuration for viz/ instrumentation.

    NOTE: These knobs are explicitly non-determinism-critical.
    """

    enable: bool = True
    summary_every_n: int = 10
    enable_behaviors_agg: bool = False
    behaviors_traits_allowlist: Tuple[str, ...] = ()


# -----------------------------------------------------------------------------
# Core writer
# -----------------------------------------------------------------------------


def _iso8601_utc_now() -> str:
    """Return an ISO8601 timestamp string in UTC with seconds precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_bytes_canonical(obj: Any, *, sort_keys: bool = True) -> bytes:
    """Encode canonical JSON bytes (UTF-8, sorted keys, minified)."""
    timing = current_flat_output_cost_timing()
    if timing is not None:
        with timing.serialization_phase():
            s = json.dumps(obj, sort_keys=sort_keys, separators=(",", ":"), ensure_ascii=False)
            return (s + "\n").encode("utf-8")
    s = json.dumps(obj, sort_keys=sort_keys, separators=(",", ":"), ensure_ascii=False)
    return (s + "\n").encode("utf-8")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomic write: write to *.tmp then os.replace."""
    tmp = Path(str(path) + ".tmp")
    # Ensure parent exists
    path.parent.mkdir(parents=True, exist_ok=True)

    timing = current_flat_output_cost_timing()
    if timing is None:
        with tmp.open("wb") as f:
            f.write(data)
            f.flush()
            # Deliberately avoid fsync here: these files are non-critical and can be
            # rewritten on the next tick. Atomicity is the contract.

        os.replace(str(tmp), str(path))
        return

    with timing.write_phase():
        with tmp.open("wb") as f:
            f.write(data)
            with timing.flush_phase():
                f.flush()
        os.replace(str(tmp), str(path))


def _file_ends_with_newline(path: Path) -> bool:
    """Return True if file is empty or ends with a newline."""
    try:
        if not path.exists():
            return True
        sz = path.stat().st_size
        if sz == 0:
            return True
        with path.open("rb") as f:
            f.seek(-1, os.SEEK_END)
            last = f.read(1)
        return last == b"\n"
    except Exception:
        # Best-effort; if unsure, don't block the append.
        return True


def _append_line(path: Path, line: str) -> None:
    """Append a single line to a file, enforcing newline discipline.

    This is append-only. If the file exists but does not end in '\n', we insert
    a '\n' before the new line so the next record is not concatenated.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    data = line
    if not data.endswith("\n"):
        data += "\n"

    # Newline discipline: repair prior partial trailing line *separator*.
    if path.exists() and (not _file_ends_with_newline(path)):
        timing = current_flat_output_cost_timing()
        if timing is None:
            with path.open("ab") as f:
                f.write(b"\n")
        else:
            with timing.write_phase():
                with path.open("ab") as f:
                    f.write(b"\n")

    timing = current_flat_output_cost_timing()
    if timing is None:
        with path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(data)
            f.flush()
        return
    with timing.write_phase():
        with path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(data)
            with timing.flush_phase():
                f.flush()


def _ensure_csv_header(path: Path, header: str) -> None:
    """Create a CSV with header if missing, using atomic create."""
    if path.exists():
        return
    timing = current_flat_output_cost_timing()
    if timing is None:
        data = (header.rstrip("\n") + "\n").encode("utf-8")
    else:
        with timing.serialization_phase():
            data = (header.rstrip("\n") + "\n").encode("utf-8")
    _atomic_write_bytes(path, data)


def _safe_float_str(v: Optional[float]) -> str:
    """Format floats for CSV output.

    We keep this simple: these files are not determinism-critical.
    """
    if v is None:
        return ""
    try:
        return str(float(v))
    except Exception:
        return ""


class VizWriter:
    """Write Phase 5d UI viz/ artifacts for a single run directory."""

    def __init__(self, run_dir: str | Path, cfg: VizConfig):
        self.run_dir = Path(run_dir)
        self.viz_dir = self.run_dir / "viz"
        self.cfg = cfg

    @property
    def enabled(self) -> bool:
        """Whether viz/ writes are enabled."""
        return bool(self.cfg.enable)

    @classmethod
    def from_env(cls, run_dir: str | Path) -> "VizWriter":
        """Construct a VizWriter from env vars.

        Knobs (safe defaults):
        - VIZ_ENABLE=1
        - VIZ_SUMMARY_EVERY_N=10
        - VIZ_ENABLE_BEHAVIORS_AGG=0
        - VIZ_BEHAVIORS_TRAITS_ALLOWLIST="traitA,traitB"
        """
        enable = _env_bool("VIZ_ENABLE", "1")
        summary_every_n = _env_int("VIZ_SUMMARY_EVERY_N", 10)
        enable_beh = _env_bool("VIZ_ENABLE_BEHAVIORS_AGG", "0")

        allow_raw = os.environ.get("VIZ_BEHAVIORS_TRAITS_ALLOWLIST", "")
        allow = _parse_csv_allowlist(allow_raw)
        if not allow:
            # Safe, small default allowlist (only used if behaviors agg is enabled).
            allow = [
                "fitness",
                "energy_collected",
                "offspring",
                "attacks_made",
                "injury_end",
                "injury_delta",
                "aggression_bias",
            ]

        cfg = VizConfig(
            enable=enable,
            summary_every_n=int(summary_every_n),
            enable_behaviors_agg=bool(enable_beh),
            behaviors_traits_allowlist=tuple(allow),
        )
        return cls(run_dir=run_dir, cfg=cfg)

    # ------------------------------------------------------------------
    # Init / status
    # ------------------------------------------------------------------

    def ensure_viz_dir(self) -> None:
        """Ensure <RUN_DIR>/viz exists."""
        if not self.enabled:
            return
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    def initialize_run(
        self,
        *,
        start_generation: int,
        target_generation: int,
        run_kind: Optional[str] = None,
    ) -> None:
        """Create initial viz/ files required by the UI.

        This is safe to call multiple times.
        """
        if not self.enabled:
            return

        self.ensure_viz_dir()

        # Required append-only artifacts: ensure they exist with headers.
        _ensure_csv_header(
            self.viz_dir / "metrics_summary.csv",
            "generation,alive,births,deaths,avg_gen_time_s",
        )

        # applied_commands_log.jsonl: empty file is fine.
        try:
            p = self.viz_dir / "applied_commands_log.jsonl"
            if not p.exists():
                _atomic_write_bytes(p, b"")
        except Exception:
            pass

        # checkpoints_index.json: create (possibly empty) from filesystem.
        self.update_checkpoints_index()

        # determinism status: write once at run start (UNKNOWN by default).
        self.write_run_determinism_status()

        # run status: running.
        rk = run_kind if run_kind in ("base", "branch", "continuation", "replay") else None
        self.write_run_status(
            status="running",
            current_generation=int(start_generation),
            target_generation=int(target_generation),
            last_error=None,
            branch_state="RUNNING",
            run_kind=rk,
        )

    def write_run_status(
        self,
        *,
        status: str,
        current_generation: int,
        target_generation: int,
        last_error: Optional[str],
        branch_state: Optional[str] = None,
        run_kind: Optional[str] = None,
    ) -> None:
        """Write viz/run_status.json (atomic)."""
        if not self.enabled:
            return

        obj: Dict[str, Any] = {
            "schema_version": "run_status_v1",
            "status": str(status),
            "current_generation": int(current_generation),
            "target_generation": int(target_generation),
            "last_update_timestamp": _iso8601_utc_now(),
            "last_error": (str(last_error) if last_error is not None else None),
        }

        if branch_state in ("PLANNED", "RUNNING", "COMPLETE", "FAILED", "ARCHIVED"):
            obj["branch_state"] = str(branch_state)

        if run_kind in ("base", "branch", "continuation", "replay"):
            obj["run_kind"] = str(run_kind)

        _atomic_write_bytes(self.viz_dir / "run_status.json", _json_bytes_canonical(obj, sort_keys=True))

    def write_run_determinism_status(self) -> None:
        """Write viz/run_determinism_status.json (atomic).

        NOTE: phase4_* fields are owned by the harness/verification pipeline.
        The engine only sets strict_mode and initializes fields to UNKNOWN.
        """
        if not self.enabled:
            return

        strict_mode = _env_bool("DEBUG_DETERMINISM", "0")
        obj: Dict[str, Any] = {
            "schema_version": "run_det_status_v1",
            "strict_mode": bool(strict_mode),
            "phase4_last_result": "UNKNOWN",
            "last_verified_generation": -1,
            "harness_profile": None,
            "last_update_timestamp": _iso8601_utc_now(),
        }
        _atomic_write_bytes(self.viz_dir / "run_determinism_status.json", _json_bytes_canonical(obj, sort_keys=True))

    # ------------------------------------------------------------------
    # Checkpoints index
    # ------------------------------------------------------------------

    def update_checkpoints_index(self) -> None:
        """Rescan checkpoint bundle v2 directories and rewrite checkpoints_index.json.

        Rule: only list checkpoints that are present and have a readable metadata.json.
        """
        if not self.enabled:
            return

        subdir = "checkpoint_bundles_v2"
        root = self.run_dir / subdir
        checkpoints: List[Dict[str, Any]] = []

        if root.exists() and root.is_dir():
            for child in sorted(root.iterdir()):
                if not child.is_dir():
                    continue
                name = child.name
                # Ignore temp dirs.
                if name.endswith(".tmp"):
                    continue

                meta_path = child / "metadata.json"
                if not meta_path.exists():
                    continue

                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    continue

                # Only index checkpoint bundle v2.
                if str(meta.get("schema_version")) != "checkpoint_bundle_v2":
                    continue

                try:
                    gen = int(meta.get("generation_index"))
                except Exception:
                    continue

                ckpt_id = str(meta.get("checkpoint_id", "") or "")

                rel = os.path.relpath(str(child), str(self.run_dir))
                rel = rel.replace("\\", "/")

                size_bytes: Optional[int] = None
                try:
                    total = 0
                    for dp, _dns, fns in os.walk(str(child)):
                        for fn in fns:
                            fp = os.path.join(dp, fn)
                            try:
                                total += int(os.path.getsize(fp))
                            except Exception:
                                pass
                    size_bytes = int(total)
                except Exception:
                    size_bytes = None

                checkpoints.append(
                    {
                        "generation": int(gen),
                        "checkpoint_id": ckpt_id,
                        "relative_path": rel,
                        "label": None,
                        "size_bytes": size_bytes,
                    }
                )

        # Stable order
        checkpoints.sort(key=lambda d: int(d.get("generation", 0)))

        obj = {
            "schema_version": "checkpoints_index_v1",
            "checkpoints": checkpoints,
        }

        # Canonical JSON (sorted keys)
        _atomic_write_bytes(self.viz_dir / "checkpoints_index.json", _json_bytes_canonical(obj, sort_keys=True))

    # ------------------------------------------------------------------
    # Timeseries summaries
    # ------------------------------------------------------------------

    def _should_sample(self, generation: int) -> bool:
        """Return True when we should emit a summary row."""
        n = int(self.cfg.summary_every_n) if int(self.cfg.summary_every_n) > 0 else 1
        return (int(generation) % int(n)) == 0

    def append_metrics_summary(
        self,
        *,
        generation: int,
        alive: int,
        births: int,
        deaths: int,
        avg_gen_time_s: Optional[float],
    ) -> None:
        """Append to viz/metrics_summary.csv (append-only)."""
        if not self.enabled:
            return
        if not self._should_sample(int(generation)):
            return

        path = self.viz_dir / "metrics_summary.csv"
        _ensure_csv_header(path, "generation,alive,births,deaths,avg_gen_time_s")

        timing = current_flat_output_cost_timing()
        if timing is None:
            row = ",".join(
                [
                    str(int(generation)),
                    str(int(alive)),
                    str(int(births)),
                    str(int(deaths)),
                    _safe_float_str(avg_gen_time_s),
                ]
            )
        else:
            with timing.serialization_phase():
                row = ",".join(
                    [
                        str(int(generation)),
                        str(int(alive)),
                        str(int(births)),
                        str(int(deaths)),
                        _safe_float_str(avg_gen_time_s),
                    ]
                )
        _append_line(path, row)

    def append_epi_av_summary(
        self,
        *,
        generation: int,
        epi: Optional[float],
        av: Optional[float],
    ) -> None:
        """Append to viz/epi_av_summary.csv (append-only).

        This file is created when EPI/AV are computed, even if cadence skips the
        first row.
        """
        if not self.enabled:
            return

        path = self.viz_dir / "epi_av_summary.csv"
        _ensure_csv_header(path, "generation,epi,av")

        if not self._should_sample(int(generation)):
            return

        row = ",".join([str(int(generation)), _safe_float_str(epi), _safe_float_str(av)])
        _append_line(path, row)

    def append_behaviors_agg_summary(
        self,
        *,
        generation: int,
        values_by_trait: Dict[str, List[float]],
    ) -> None:
        """Append a behaviors aggregation row (append-only).

        Caller is expected to provide only allowlisted traits.
        """
        if not self.enabled:
            return
        if not bool(self.cfg.enable_behaviors_agg):
            return
        if not self._should_sample(int(generation)):
            return

        traits = list(self.cfg.behaviors_traits_allowlist)
        if not traits:
            return

        # Lazy import: avoid numpy overhead unless enabled.
        try:
            import numpy as np  # type: ignore
        except Exception:
            np = None  # type: ignore

        # Header
        cols: List[str] = ["generation"]
        for t in traits:
            cols.extend([f"{t}_mean", f"{t}_std", f"{t}_p05", f"{t}_p50", f"{t}_p95"])

        path = self.viz_dir / "behaviors_agg_summary.csv"
        _ensure_csv_header(path, ",".join(cols))

        # Row
        out: List[str] = [str(int(generation))]
        for t in traits:
            vals = list(values_by_trait.get(t, []) or [])
            if not vals:
                out.extend(["", "", "", "", ""])  # empty stats
                continue

            if np is not None:
                arr = np.asarray(vals, dtype=float)
                mean = float(arr.mean())
                std = float(arr.std(ddof=0))
                p05, p50, p95 = [float(x) for x in np.percentile(arr, [5, 50, 95])]
            else:
                # Fallback: simple python stats.
                arr = sorted(float(x) for x in vals)
                mean = float(sum(arr)) / float(len(arr))
                std = 0.0
                if len(arr) > 1:
                    m = mean
                    std = (sum((x - m) ** 2 for x in arr) / float(len(arr))) ** 0.5

                def _pct(p: float) -> float:
                    if not arr:
                        return 0.0
                    idx = int(round((p / 100.0) * (len(arr) - 1)))
                    idx = max(0, min(len(arr) - 1, idx))
                    return float(arr[idx])

                p05, p50, p95 = _pct(5), _pct(50), _pct(95)

            out.extend([
                _safe_float_str(mean),
                _safe_float_str(std),
                _safe_float_str(p05),
                _safe_float_str(p50),
                _safe_float_str(p95),
            ])

        _append_line(path, ",".join(out))

    # ------------------------------------------------------------------
    # ControlPlane command timeline
    # ------------------------------------------------------------------

    def append_applied_command_event(
        self,
        *,
        command_uuid: str,
        command_type: str,
        apply_phase: str,
        applied_generation: int,
        payload_hash: str,
        ack_status: str,
        error_reason: Optional[str],
        timestamp: Optional[str] = None,
    ) -> None:
        """Append to viz/applied_commands_log.jsonl (append-only)."""
        if not self.enabled:
            return

        obj = {
            "schema_version": "applied_cmd_v1",
            "command_uuid": str(command_uuid),
            "command_type": str(command_type),
            "apply_phase": str(apply_phase),
            "applied_generation": int(applied_generation),
            "payload_hash": str(payload_hash),
            "ack_status": str(ack_status),
            "error_reason": (str(error_reason) if error_reason is not None else None),
            "timestamp": str(timestamp or _iso8601_utc_now()),
        }

        path = self.viz_dir / "applied_commands_log.jsonl"
        # Ensure file exists (empty is ok)
        if not path.exists():
            try:
                _atomic_write_bytes(path, b"")
            except Exception:
                pass

        _append_line(path, json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


# -----------------------------------------------------------------------------
# Cache helper
# -----------------------------------------------------------------------------


_VIZ_WRITER_CACHE: Dict[str, VizWriter] = {}


def get_viz_writer(run_dir: str | Path) -> VizWriter:
    """Return a cached VizWriter for run_dir."""
    key = os.path.abspath(str(run_dir))
    vw = _VIZ_WRITER_CACHE.get(key)
    if vw is None:
        vw = VizWriter.from_env(key)
        _VIZ_WRITER_CACHE[key] = vw
    return vw
