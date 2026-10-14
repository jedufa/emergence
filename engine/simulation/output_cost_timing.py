from __future__ import annotations

"""Opt-in exclusive timing for Flat benchmark output costs."""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import json
import os
from pathlib import Path
import threading
import time
from typing import Callable, Iterator, Sequence


SIDECAR_ENV = "EIEL_FLAT_OUTPUT_COST_SIDECAR"
SCHEMA_VERSION = "eiel_flat_output_cost_sample_v1"
PHASES = (
    "simulation",
    "reporting_projection",
    "reporting_reducer",
    "serialization",
    "write",
    "flush",
)
CALL_COUNT_KEYS = PHASES

_PHASE_CALL_COUNTS: dict[str, tuple[str, ...]] = {
    "simulation": ("simulation",),
    "reporting_projection": ("reporting_projection",),
    "reporting_reducer": ("reporting_reducer",),
    "serialization": ("serialization",),
    "write": ("write",),
    "flush": ("flush",),
}


@dataclass
class _PhaseFrame:
    phase: str
    started_ns: int
    child_ns: int = 0


class FlatOutputCostTiming:
    """Accumulate exclusive phase timings for one engine process run."""

    def __init__(
        self,
        sidecar_path: str | os.PathLike[str],
        *,
        clock: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        raw_target = os.fspath(sidecar_path)
        if not str(raw_target).strip():
            raise ValueError("Flat output-cost sidecar path must not be empty")
        self.sidecar_path = Path(raw_target).expanduser().resolve(strict=False)
        self._clock = clock
        self._exclusive_ns = {phase: 0 for phase in PHASES}
        self._call_counts = {key: 0 for key in CALL_COUNT_KEYS}
        self._phase_entries = {phase: 0 for phase in PHASES}
        self._errors: list[str] = []
        self._run_started_ns: int | None = None
        self._run_total_ns = 0
        self._finalized = False
        self._local = threading.local()
        self._lock = threading.Lock()

    def begin(self) -> None:
        if self._run_started_ns is not None:
            raise RuntimeError("Flat output-cost timing has already begun")
        if self._finalized:
            raise RuntimeError("Flat output-cost timing has already finalized")
        self._run_started_ns = int(self._clock())

    def _stack(self) -> list[_PhaseFrame]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        return stack

    @contextmanager
    def phase(
        self,
        phase: str,
        *,
        call_count_keys: Sequence[str] | None = None,
    ) -> Iterator[None]:
        if phase not in self._exclusive_ns:
            raise ValueError(f"Unknown Flat output-cost phase: {phase}")
        if self._run_started_ns is None:
            raise RuntimeError("Flat output-cost timing phase entered before begin")
        if self._finalized:
            raise RuntimeError("Flat output-cost timing phase entered after finalize")

        count_keys = tuple(
            _PHASE_CALL_COUNTS[phase]
            if call_count_keys is None
            else call_count_keys
        )
        unknown_counts = [key for key in count_keys if key not in self._call_counts]
        if unknown_counts:
            raise ValueError(f"Unknown Flat output-cost call count keys: {unknown_counts}")

        frame = _PhaseFrame(phase=phase, started_ns=int(self._clock()))
        stack = self._stack()
        stack.append(frame)
        with self._lock:
            self._phase_entries[phase] += 1
            for key in count_keys:
                self._call_counts[key] += 1
        try:
            yield
        except BaseException as exc:
            self.record_error(
                f"phase_error:{phase}:{type(exc).__name__}:{_error_text(exc)}"
            )
            raise
        finally:
            ended_ns = int(self._clock())
            popped = stack.pop()
            if popped is not frame:
                self.record_error(f"phase_stack_corruption:{phase}")
            elapsed_ns = max(0, ended_ns - frame.started_ns)
            exclusive_ns = max(0, elapsed_ns - frame.child_ns)
            with self._lock:
                self._exclusive_ns[phase] += exclusive_ns
            if stack:
                stack[-1].child_ns += elapsed_ns

    def simulation_phase(self):
        return self.phase("simulation")

    def reporting_projection_phase(self):
        return self.phase("reporting_projection")

    def reporting_reducer_phase(self):
        return self.phase("reporting_reducer")

    def serialization_phase(self):
        return self.phase("serialization")

    def write_phase(self):
        return self.phase("write")

    def checkpoint_write_phase(self):
        return self.phase("write")

    def flush_phase(self):
        return self.phase("flush")

    def record_error(self, error: str) -> None:
        value = str(error).strip()
        if not value:
            return
        with self._lock:
            if value not in self._errors:
                self._errors.append(value)

    def finalize(
        self,
        *,
        run_dir: str | os.PathLike[str] | None = None,
        run_complete: bool = True,
        error: BaseException | None = None,
    ) -> dict[str, object]:
        """Stop the run clock and emit the strict sidecar exactly once."""

        if self._finalized:
            raise RuntimeError("Flat output-cost timing has already finalized")
        if self._run_started_ns is None:
            raise RuntimeError("Flat output-cost timing finalized before begin")

        ended_ns = int(self._clock())
        self._run_total_ns = max(0, ended_ns - self._run_started_ns)
        if self._stack():
            self.record_error("active_phase_contexts_at_finalize")
        if error is not None:
            self.record_error(f"run_error:{type(error).__name__}:{_error_text(error)}")
        if not run_complete and error is None:
            self.record_error("run_incomplete")

        missing_phases = [
            phase for phase in PHASES if self._phase_entries[phase] == 0
        ]
        if missing_phases:
            self.record_error(f"missing_phase_timing:{','.join(missing_phases)}")

        exclusive_total = sum(self._exclusive_ns.values())
        if exclusive_total > self._run_total_ns:
            self.record_error("exclusive_ns_sum_exceeds_run_total_ns")

        self._validate_sidecar_location(run_dir)
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "exclusive_ns": dict(self._exclusive_ns),
            "call_counts": dict(self._call_counts),
            "run_total_ns": int(self._run_total_ns),
            "complete": not self._errors,
            "errors": list(self._errors),
        }
        self._finalized = True
        self.sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        self.sidecar_path.write_text(encoded + "\n", encoding="utf-8")
        return payload

    def _validate_sidecar_location(
        self,
        run_dir: str | os.PathLike[str] | None,
    ) -> None:
        if run_dir is None:
            return
        run_root = Path(run_dir).expanduser().resolve(strict=False)
        try:
            self.sidecar_path.relative_to(run_root)
        except ValueError:
            return
        raise ValueError(
            "EIEL_FLAT_OUTPUT_COST_SIDECAR must point outside run_dir: "
            f"sidecar={self.sidecar_path} run_dir={run_root}"
        )


_ACTIVE_LOCK = threading.Lock()
_ACTIVE: FlatOutputCostTiming | None = None


def begin_flat_output_cost_timing(
    sidecar_path: str | os.PathLike[str] | None = None,
    *,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> FlatOutputCostTiming | None:
    """Activate one timing session, or return ``None`` when the env is unset."""

    target = sidecar_path
    if target is None:
        target = os.environ.get(SIDECAR_ENV)
        if not target:
            return None
    timing = FlatOutputCostTiming(target, clock=clock)
    timing.begin()
    global _ACTIVE
    with _ACTIVE_LOCK:
        if _ACTIVE is not None:
            raise RuntimeError("A Flat output-cost timing session is already active")
        _ACTIVE = timing
    return timing


def current_flat_output_cost_timing() -> FlatOutputCostTiming | None:
    return _ACTIVE


def finalize_flat_output_cost_timing(
    timing: FlatOutputCostTiming | None,
    *,
    run_dir: str | os.PathLike[str] | None = None,
    run_complete: bool = True,
    error: BaseException | None = None,
) -> dict[str, object] | None:
    """Finalize and deactivate ``timing``; disabled sessions stay no-op."""

    if timing is None:
        return None
    global _ACTIVE
    with _ACTIVE_LOCK:
        if _ACTIVE is not timing:
            raise RuntimeError("Flat output-cost timing session is not active")
        _ACTIVE = None
    return timing.finalize(
        run_dir=run_dir,
        run_complete=run_complete,
        error=error,
    )


def _active_phase(name: str):
    timing = current_flat_output_cost_timing()
    return nullcontext() if timing is None else timing.phase(name)


def simulation_phase():
    return _active_phase("simulation")


def reporting_projection_phase():
    return _active_phase("reporting_projection")


def reporting_reducer_phase():
    return _active_phase("reporting_reducer")


def serialization_phase():
    return _active_phase("serialization")


def write_phase():
    return _active_phase("write")


def checkpoint_write_phase():
    timing = current_flat_output_cost_timing()
    return nullcontext() if timing is None else timing.checkpoint_write_phase()


def flush_phase():
    return _active_phase("flush")


def _error_text(exc: BaseException) -> str:
    return " ".join(str(exc).split()) or "no_message"


__all__ = [
    "CALL_COUNT_KEYS",
    "FlatOutputCostTiming",
    "PHASES",
    "SCHEMA_VERSION",
    "SIDECAR_ENV",
    "begin_flat_output_cost_timing",
    "checkpoint_write_phase",
    "current_flat_output_cost_timing",
    "finalize_flat_output_cost_timing",
    "flush_phase",
    "reporting_projection_phase",
    "reporting_reducer_phase",
    "serialization_phase",
    "simulation_phase",
    "write_phase",
]
