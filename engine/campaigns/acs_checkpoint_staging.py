"""Bounded asynchronous persistence for detached checkpoint snapshots.

The pipeline deliberately knows nothing about live simulation state.  A caller
may submit work only after it owns an immutable host snapshot.  One FIFO writer
executes persistence work, while the caller may advance compute until the
admitted snapshot slots are full.  Completion is not durability authority until
the caller consumes the result and publishes its checkpoint catalog entry.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar


ACS_CHECKPOINT_STAGING_SCHEMA_VERSION = "eiel_acs_checkpoint_staging_v1"
ACS_CHECKPOINT_SLOTS_DEFAULT_V1 = 2


class ACSCheckpointStagingError(RuntimeError):
    """Raised when bounded checkpoint staging fails closed."""


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ACSCheckpointToken(Generic[T]):
    """Identity for one admitted immutable checkpoint snapshot."""

    sequence: int
    compute_frontier: int


@dataclass(slots=True)
class _Job(Generic[T]):
    token: ACSCheckpointToken[T]
    persist: Callable[[], T] | None
    enqueued_ns: int
    captured_bytes: int = 0
    started_ns: int | None = None
    finished_ns: int | None = None
    result: T | None = None
    failure: BaseException | None = None
    consumed: bool = False


class ACSCheckpointStaging(Generic[T]):
    """One-writer checkpoint family with one to three bounded snapshot slots.

    One and two slots are the supported implementation candidates.  Three slots
    exist only for the master program's explicitly gated comparison and is never
    selected by default.
    """

    def __init__(
        self,
        *,
        slots: int,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        thread_factory: Callable[..., threading.Thread] = threading.Thread,
    ) -> None:
        if type(slots) is not int or slots not in {1, 2, 3}:
            raise ACSCheckpointStagingError(
                "ACS checkpoint slots must be exactly one, two, or three"
            )
        self._slots = slots
        self._clock_ns = clock_ns
        self._condition = threading.Condition()
        self._jobs: deque[_Job[T]] = deque()
        self._by_sequence: dict[int, _Job[T]] = {}
        self._released_tokens: deque[ACSCheckpointToken[T]] = deque()
        self._next_sequence = 1
        self._compute_frontier = -1
        self._durable_checkpoint_frontier = -1
        self._occupancy_high_water = 0
        self._producer_blocked_ns = 0
        self._writer_busy_ns = 0
        self._closed = False
        self._aborted = False
        self._abandoned_snapshot_count = 0
        self._failure: BaseException | None = None
        self._writer = thread_factory(
            target=self._writer_main,
            name="eiel-acs-checkpoint-writer",
            daemon=True,
        )
        self._writer.start()

    @property
    def slots(self) -> int:
        return self._slots

    def submit(
        self,
        *,
        compute_frontier: int,
        captured_bytes: int,
        persist: Callable[[], T],
    ) -> ACSCheckpointToken[T]:
        if type(captured_bytes) is not int or captured_bytes < 0:
            raise ACSCheckpointStagingError(
                "ACS captured bytes must be a non-negative integer"
            )
        if not callable(persist):
            raise ACSCheckpointStagingError("ACS persistence task must be callable")
        token = self.begin_capture(compute_frontier=compute_frontier)
        self.seal_capture(
            token,
            captured_bytes=captured_bytes,
            persist=persist,
        )
        return token

    def begin_capture(self, *, compute_frontier: int) -> ACSCheckpointToken[T]:
        """Reserve one bounded slot before settlement creates a host snapshot."""

        if type(compute_frontier) is not int or compute_frontier < 0:
            raise ACSCheckpointStagingError(
                "ACS compute frontier must be a non-negative integer"
            )

        blocked_started: int | None = None
        with self._condition:
            self._raise_if_failed_locked()
            if self._closed:
                raise ACSCheckpointStagingError("ACS checkpoint staging is closed")
            if compute_frontier <= self._compute_frontier:
                raise ACSCheckpointStagingError(
                    "ACS compute frontiers must be strictly increasing"
                )
            while self._active_slot_count_locked() >= self._slots:
                if blocked_started is None:
                    blocked_started = self._clock_ns()
                self._condition.wait()
                self._raise_if_failed_locked()
                if self._closed:
                    raise ACSCheckpointStagingError(
                        "ACS checkpoint staging closed during backpressure"
                    )
            if blocked_started is not None:
                self._producer_blocked_ns += self._clock_ns() - blocked_started
            token = ACSCheckpointToken[T](
                sequence=self._next_sequence,
                compute_frontier=compute_frontier,
            )
            self._next_sequence += 1
            self._compute_frontier = compute_frontier
            job = _Job(token=token, persist=None, enqueued_ns=self._clock_ns())
            self._jobs.append(job)
            self._by_sequence[token.sequence] = job
            self._occupancy_high_water = max(
                self._occupancy_high_water,
                self._active_slot_count_locked(),
            )
            self._condition.notify_all()
            return token

    def seal_capture(
        self,
        token: ACSCheckpointToken[T],
        *,
        captured_bytes: int,
        persist: Callable[[], T],
    ) -> None:
        """Freeze a reserved capture and make it eligible for FIFO persistence."""

        if type(captured_bytes) is not int or captured_bytes < 0:
            raise ACSCheckpointStagingError(
                "ACS captured bytes must be a non-negative integer"
            )
        if not callable(persist):
            raise ACSCheckpointStagingError("ACS persistence task must be callable")
        with self._condition:
            self._raise_if_failed_locked()
            if self._closed:
                raise ACSCheckpointStagingError("ACS checkpoint staging is closed")
            job = self._job_for_token_locked(token)
            if job.persist is not None or job.started_ns is not None:
                raise ACSCheckpointStagingError(
                    "ACS checkpoint capture was already sealed"
                )
            job.captured_bytes = captured_bytes
            job.persist = persist
            self._condition.notify_all()

    def result(
        self,
        token: ACSCheckpointToken[T],
        *,
        producer_backpressure: bool = False,
    ) -> T:
        with self._condition:
            self._raise_if_aborted_locked()
            job = self._job_for_token_locked(token)
            if not self._jobs or self._jobs[0] is not job:
                raise ACSCheckpointStagingError(
                    "ACS checkpoint results must be consumed in FIFO order"
                )
            if job.persist is None:
                raise ACSCheckpointStagingError(
                    "ACS checkpoint capture is not sealed for persistence"
                )
            blocked_started = (
                self._clock_ns()
                if producer_backpressure and job.finished_ns is None
                else None
            )
            while job.finished_ns is None:
                self._condition.wait()
                self._raise_if_aborted_locked()
            self._raise_if_aborted_locked()
            if blocked_started is not None:
                self._producer_blocked_ns += self._clock_ns() - blocked_started
            if job.failure is not None:
                self._raise_failure(job.failure)
            if job.consumed:
                raise ACSCheckpointStagingError(
                    "ACS checkpoint result was already consumed"
                )
            job.consumed = True
            self._released_tokens.append(token)
            self._retire_consumed_prefix_locked()
            self._condition.notify_all()
            return job.result  # type: ignore[return-value]

    def acknowledge_durable(self, token: ACSCheckpointToken[T]) -> None:
        """Advance durability only after the caller publishes the catalog record."""

        with self._condition:
            self._raise_if_aborted_locked()
            if not self._released_tokens or self._released_tokens[0] != token:
                raise ACSCheckpointStagingError(
                    "ACS durable frontiers must be acknowledged in FIFO order"
                )
            self._released_tokens.popleft()
            self._durable_checkpoint_frontier = token.compute_frontier
            self._condition.notify_all()

    def poll_failure(self) -> None:
        with self._condition:
            self._raise_if_failed_locked()

    def ready(self, token: ACSCheckpointToken[T]) -> bool:
        """Return whether a FIFO result is sealed or failed without consuming it."""

        with self._condition:
            self._raise_if_aborted_locked()
            job = self._job_for_token_locked(token)
            return job.finished_ns is not None

    def close(self) -> None:
        with self._condition:
            if not self._closed:
                unconsumed = [job for job in self._jobs if not job.consumed]
                if unconsumed:
                    raise ACSCheckpointStagingError(
                        "ACS checkpoint staging cannot close with unconsumed snapshots"
                    )
                if self._released_tokens:
                    raise ACSCheckpointStagingError(
                        "ACS checkpoint staging cannot close with unpublished snapshots"
                    )
                self._closed = True
                self._condition.notify_all()
        self._writer.join()

    def abort(self) -> None:
        """Stop accepting work after an outer failure and join completed work.

        Already-running persistence is allowed to finish so its atomic
        transaction can seal or quarantine itself.  Pending tasks are abandoned
        without being advertised durable.
        """

        with self._condition:
            if not self._aborted and (
                not self._closed
                or self._jobs
                or self._released_tokens
                or self._failure is not None
            ):
                self._closed = True
                self._aborted = True
                self._abandoned_snapshot_count += len(self._jobs) + len(
                    self._released_tokens
                )
                self._jobs.clear()
                self._by_sequence.clear()
                self._released_tokens.clear()
                self._condition.notify_all()
        self._writer.join()

    def telemetry(self) -> dict[str, Any]:
        with self._condition:
            now = self._clock_ns()
            active = [job for job in self._jobs if not job.consumed]
            backlog_age_ns = max(
                (now - job.enqueued_ns for job in active),
                default=0,
            )
            states = []
            for slot in range(self._slots):
                if slot >= len(active):
                    states.append("FREE")
                    continue
                job = active[slot]
                if job.failure is not None:
                    states.append("FAILED")
                elif job.finished_ns is not None:
                    states.append("SEALED/RELEASABLE")
                elif job.started_ns is not None:
                    states.append("PERSISTING")
                else:
                    states.append("CAPTURING")
            return {
                "schema_version": ACS_CHECKPOINT_STAGING_SCHEMA_VERSION,
                "slots": self._slots,
                "writer_count": 1,
                "slot_states": states,
                "queue_occupancy": len(active),
                "queue_occupancy_high_water": self._occupancy_high_water,
                "captured_bytes_active": sum(job.captured_bytes for job in active),
                "producer_blocked_time_ns": self._producer_blocked_ns,
                "writer_busy_time_ns": self._writer_busy_ns,
                "backlog_age_ns": backlog_age_ns,
                "compute_frontier": self._compute_frontier,
                "durable_checkpoint_frontier": self._durable_checkpoint_frontier,
                "frontier_lag": max(
                    0,
                    self._compute_frontier - self._durable_checkpoint_frontier,
                ),
                "failure_latched": self._failure is not None,
                "closed": self._closed,
                "aborted": self._aborted,
                "abandoned_snapshot_count": self._abandoned_snapshot_count,
                "writer_alive": self._writer.is_alive(),
            }

    def _writer_main(self) -> None:
        while True:
            with self._condition:
                job = self._next_fifo_job_locked()
                while job is None:
                    if self._closed:
                        return
                    self._condition.wait()
                    job = self._next_fifo_job_locked()
                job.started_ns = self._clock_ns()
                persist = job.persist
                assert persist is not None
            try:
                result = persist()
            except BaseException as exc:
                with self._condition:
                    job.failure = exc
                    job.finished_ns = self._clock_ns()
                    self._writer_busy_ns += job.finished_ns - job.started_ns
                    self._failure = exc
                    self._closed = True
                    self._condition.notify_all()
                return
            with self._condition:
                job.result = result
                job.finished_ns = self._clock_ns()
                self._writer_busy_ns += job.finished_ns - job.started_ns
                self._condition.notify_all()

    def _active_slot_count_locked(self) -> int:
        return sum(not job.consumed for job in self._jobs)

    def _next_fifo_job_locked(self) -> _Job[T] | None:
        for candidate in self._jobs:
            if candidate.started_ns is None:
                return candidate if candidate.persist is not None else None
        return None

    def _job_for_token_locked(self, token: ACSCheckpointToken[T]) -> _Job[T]:
        job = self._by_sequence.get(token.sequence)
        if job is None or job.token != token:
            raise ACSCheckpointStagingError("ACS checkpoint token is unknown")
        return job

    def _retire_consumed_prefix_locked(self) -> None:
        while self._jobs and self._jobs[0].consumed:
            retired = self._jobs.popleft()
            self._by_sequence.pop(retired.token.sequence, None)

    def _raise_if_failed_locked(self) -> None:
        if self._failure is not None:
            self._raise_failure(self._failure)

    def _raise_if_aborted_locked(self) -> None:
        if self._aborted:
            raise ACSCheckpointStagingError("ACS checkpoint staging was aborted")

    @staticmethod
    def _raise_failure(cause: BaseException) -> None:
        raise ACSCheckpointStagingError(
            f"EIEL-ACS-CHECKPOINT-STAGING-FAILED: {type(cause).__name__}: {cause}"
        ) from cause


__all__ = [
    "ACS_CHECKPOINT_STAGING_SCHEMA_VERSION",
    "ACSCheckpointStaging",
    "ACSCheckpointStagingError",
    "ACSCheckpointToken",
]
