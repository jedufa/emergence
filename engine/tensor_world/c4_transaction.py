"""Preparation-bound C4 population transaction coordinator.

This module gives the existing CPU source oracles one explicit transaction
boundary.  It does not put population mutation into the C3 tick graph and it
does not claim resident CUDA qualification.  The coordinator is useful for
parity work because it makes the source phase order, per-world counts, and
failure boundary observable in one machine-readable receipt.

The rollback boundary is intentionally explicit: this bounded preparation
route snapshots all tensor leaves in the C4 bundle (and an optional C5 replay
context) before running the CPU oracle phases.  Any phase failure restores the
snapshot and raises :class:`C4PopulationTransactionError` carrying the failed
receipt.  A future resident implementation must replace this snapshot with a
device-side staged commit; this helper must not be mistaken for that route.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from hashlib import sha256
from typing import Mapping, TYPE_CHECKING, Final

import torch

from .c4_contract import (
    C4_GATE_AUTHORIZATION_V1,
    C4_PARITY_BOUNDARY_SURFACES_V1,
    C4_PHASE_ORDER,
)
from .c4_population import (
    TensorWorldC4BirthConfigV1,
    TensorWorldC4PopulationConfigV1,
    apply_c4_death_settlement_oracle_v1,
    commit_c4_ordinary_births_oracle_v1,
    plan_c4_ordinary_births_oracle_v1,
    release_c4_free_slots_oracle_v1,
)
from .c4_state import TensorWorldStateBundleV4
from .contracts import TensorWorldContractError

if TYPE_CHECKING:
    from .c4_mating import TensorWorldC4MateBirthConfigV1
    from .c5_determinism import TensorWorldC5ReplayContextV1


C4_POPULATION_TRANSACTION_RECEIPT_SCHEMA_V1: Final[str] = (
    "tensor_world_c4_population_transaction_receipt_v1"
)
C4_TRANSACTION_PREPARATION_STATUS_V1: Final[str] = "PREPARATION_PASS"
C4_TRANSACTION_ROLLED_BACK_STATUS_V1: Final[str] = (
    "PREPARATION_FAILED_ROLLED_BACK"
)
C4_TRANSACTION_DECISION_V1: Final[str] = "NO_GATE_DECISION"
C4_TRANSACTION_ROLLBACK_SCOPE_V1: Final[str] = "cpu_oracle_tensor_snapshot"
C4_POPULATION_BOUNDARY_SCHEMA_V1: Final[str] = (
    "tensor_world_c4_population_boundary_snapshot_v1"
)


def _boundary_surface_paths_v1() -> tuple[str, ...]:
    """Return the declared surfaces once, preserving contract order."""

    paths: list[str] = []
    for group in C4_PARITY_BOUNDARY_SURFACES_V1.values():
        for path in group:
            if path not in paths:
                paths.append(path)
    return tuple(paths)


C4_BOUNDARY_SURFACE_PATHS_V1: Final[tuple[str, ...]] = _boundary_surface_paths_v1()


@dataclass(frozen=True, slots=True)
class C4PopulationBoundarySnapshotV1:
    """CPU observation of the declared C4 transaction-boundary surfaces.

    This is an oracle/parity preparation object.  Capturing it intentionally
    performs a host copy and therefore cannot be called from the steady-state
    resident transaction.  A CUDA implementation must provide an equivalent
    device-local receipt before C4 residency evidence can be admitted.
    """

    phase: str
    surfaces: Mapping[str, torch.Tensor]
    digest: str
    schema_version: str = C4_POPULATION_BOUNDARY_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema_version != C4_POPULATION_BOUNDARY_SCHEMA_V1:
            raise TensorWorldContractError(
                "C4 population boundary snapshot schema is unsupported"
            )
        if not self.phase:
            raise TensorWorldContractError("C4 population boundary phase is required")
        if tuple(self.surfaces) != tuple(
            path for path in C4_BOUNDARY_SURFACE_PATHS_V1 if path in self.surfaces
        ):
            raise TensorWorldContractError(
                "C4 population boundary surfaces diverge from the declared contract"
            )
        if any(tensor.device.type != "cpu" for tensor in self.surfaces.values()):
            raise TensorWorldContractError(
                "C4 population boundary snapshots must be CPU preparation observations"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "phase": self.phase,
            "surface_names": list(self.surfaces),
            "digest": self.digest,
            "host_observation": True,
            "gate_claim": False,
        }


def _resolve_boundary_surface_v1(bundle: TensorWorldStateBundleV4, path: str) -> torch.Tensor | None:
    owner: object = bundle
    for component in path.split("."):
        if component == "events" and getattr(owner, component, None) is None:
            return None
        owner = getattr(owner, component)
    if not isinstance(owner, torch.Tensor):
        raise TensorWorldContractError(f"C4 boundary surface is not a tensor: {path}")
    return owner


def _boundary_digest_v1(surfaces: Mapping[str, torch.Tensor]) -> str:
    digest = sha256()
    for path, tensor in surfaces.items():
        digest.update(path.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def capture_c4_population_boundary_v1(
    bundle: TensorWorldStateBundleV4, *, phase: str
) -> C4PopulationBoundarySnapshotV1:
    """Capture declared C4 surfaces for CPU oracle parity preparation."""

    if not isinstance(bundle, TensorWorldStateBundleV4):
        raise TensorWorldContractError("C4 boundary capture requires a C4 state bundle")
    bundle.validate()
    if bundle.device.type != "cpu":
        raise TensorWorldContractError(
            "C4 boundary capture is a CPU preparation observation; resident CUDA capture is not wired"
        )
    surfaces: dict[str, torch.Tensor] = {}
    for path in C4_BOUNDARY_SURFACE_PATHS_V1:
        tensor = _resolve_boundary_surface_v1(bundle, path)
        if tensor is not None:
            surfaces[path] = tensor.detach().clone().contiguous()
    return C4PopulationBoundarySnapshotV1(
        phase=phase,
        surfaces=surfaces,
        digest=_boundary_digest_v1(surfaces),
    )


def compare_c4_population_boundaries_v1(
    expected: C4PopulationBoundarySnapshotV1,
    actual: C4PopulationBoundarySnapshotV1,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> dict[str, object]:
    """Compare two declared boundary observations without slot reordering."""

    if not isinstance(expected, C4PopulationBoundarySnapshotV1) or not isinstance(
        actual, C4PopulationBoundarySnapshotV1
    ):
        raise TensorWorldContractError("C4 boundary comparison requires matching snapshots")
    if atol < 0.0 or rtol < 0.0:
        raise TensorWorldContractError("C4 boundary tolerances must be nonnegative")
    mismatches: list[dict[str, object]] = []
    names = tuple(dict.fromkeys((*expected.surfaces, *actual.surfaces)))
    for path in names:
        lhs = expected.surfaces.get(path)
        rhs = actual.surfaces.get(path)
        if lhs is None or rhs is None:
            mismatches.append({"surface": path, "reason": "missing_surface"})
            continue
        if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
            mismatches.append(
                {
                    "surface": path,
                    "reason": "shape_or_dtype",
                    "expected_shape": list(lhs.shape),
                    "actual_shape": list(rhs.shape),
                    "expected_dtype": str(lhs.dtype),
                    "actual_dtype": str(rhs.dtype),
                }
            )
            continue
        equal = (
            torch.allclose(lhs, rhs, atol=atol, rtol=rtol, equal_nan=True)
            if lhs.is_floating_point()
            else torch.equal(lhs, rhs)
        )
        if not bool(equal):
            mismatches.append({"surface": path, "reason": "value_mismatch"})
    return {
        "pass": not mismatches,
        "expected_phase": expected.phase,
        "actual_phase": actual.phase,
        "expected_digest": expected.digest,
        "actual_digest": actual.digest,
        "surface_count": len(names),
        "mismatches": mismatches,
        "gate_claim": False,
    }


@dataclass(frozen=True, slots=True)
class C4PopulationTransactionReceiptV1:
    """Machine-readable outcome of one bounded C4 oracle transaction."""

    status: str
    phase_status: tuple[tuple[str, str], ...]
    phase_counts: Mapping[str, Mapping[str, tuple[int, ...]]]
    counts_before: Mapping[str, tuple[int, ...]]
    counts_after: Mapping[str, tuple[int, ...]]
    failure_phase: str | None = None
    failure_type: str | None = None
    failure_message: str | None = None
    rollback_applied: bool = False
    schema_version: str = C4_POPULATION_TRANSACTION_RECEIPT_SCHEMA_V1
    gate_authorization: str = C4_GATE_AUTHORIZATION_V1
    decision: str = C4_TRANSACTION_DECISION_V1
    phase_order: tuple[str, ...] = C4_PHASE_ORDER
    rollback_scope: str = C4_TRANSACTION_ROLLBACK_SCOPE_V1
    gate_claim: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != C4_POPULATION_TRANSACTION_RECEIPT_SCHEMA_V1:
            raise TensorWorldContractError(
                "C4 transaction receipt schema version is unsupported"
            )
        if tuple(name for name, _status in self.phase_status) != self.phase_order:
            raise TensorWorldContractError(
                "C4 transaction receipt phase status order diverges from source"
            )
        if self.gate_authorization != C4_GATE_AUTHORIZATION_V1:
            raise TensorWorldContractError(
                "C4 transaction receipt gate authorization is invalid"
            )
        if self.decision != C4_TRANSACTION_DECISION_V1 or self.gate_claim:
            raise TensorWorldContractError(
                "C4 transaction receipt cannot claim a gate decision"
            )
        if self.status == C4_TRANSACTION_ROLLED_BACK_STATUS_V1 and not self.rollback_applied:
            raise TensorWorldContractError(
                "a rolled-back C4 transaction must record rollback_applied"
            )

    @property
    def phase_status_map(self) -> dict[str, str]:
        return dict(self.phase_status)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible receipt without losing phase ordering."""

        def lists(value: object) -> object:
            if isinstance(value, tuple):
                return [lists(item) for item in value]
            if isinstance(value, Mapping):
                return {str(key): lists(item) for key, item in value.items()}
            return value

        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "gate_authorization": self.gate_authorization,
            "decision": self.decision,
            "gate_claim": self.gate_claim,
            "phase_order": list(self.phase_order),
            "phase_status": {name: status for name, status in self.phase_status},
            "phase_counts": lists(self.phase_counts),
            "counts_before": lists(self.counts_before),
            "counts_after": lists(self.counts_after),
            "rollback_scope": self.rollback_scope,
            "rollback_applied": self.rollback_applied,
            "failure_phase": self.failure_phase,
            "failure_type": self.failure_type,
            "failure_message": self.failure_message,
        }


class C4PopulationTransactionError(TensorWorldContractError):
    """Raised after a C4 oracle failure has restored its transaction snapshot."""

    def __init__(self, message: str, receipt: C4PopulationTransactionReceiptV1):
        super().__init__(message)
        self.receipt = receipt


def _snapshot_tensor_leaves(value: object, snapshot: dict[int, tuple[torch.Tensor, torch.Tensor]]) -> None:
    if isinstance(value, torch.Tensor):
        snapshot.setdefault(id(value), (value, value.detach().clone()))
        return
    if is_dataclass(value):
        for field in fields(value):
            _snapshot_tensor_leaves(getattr(value, field.name), snapshot)


def _restore_tensor_leaves(snapshot: dict[int, tuple[torch.Tensor, torch.Tensor]]) -> None:
    with torch.no_grad():
        for target, saved in snapshot.values():
            target.copy_(saved)


def _counts_v1(bundle: TensorWorldStateBundleV4) -> dict[str, tuple[int, ...]]:
    core = bundle.base.base.core
    p = bundle.population
    counts: dict[str, tuple[int, ...]] = {
        "alive_count": tuple(int(value) for value in core.agents.alive.sum(dim=1).tolist()),
        "active_count": tuple(int(value) for value in p.active_count.tolist()),
        "free_queue_count": tuple(int(value) for value in p.free_queue_count.tolist()),
        "pending_free_count": tuple(int(value) for value in p.pending_free_count.tolist()),
        "death_count": tuple(int(value) for value in p.death_count.tolist()),
        "planned_birth_count": tuple(int(value) for value in p.planned_count.tolist()),
        "birth_count": tuple(int(value) for value in p.birth_count.tolist()),
        "decision_count": tuple(int(value) for value in p.decision_count.tolist()),
        "spawn_rng_draws_added": tuple(int(value) for value in p.spawn_rng_draws_added.tolist()),
        "spawn_rng_semantic_draws_added": tuple(
            int(value) for value in p.spawn_rng_semantic_draws_added.tolist()
        ),
        "runner_rng_draws_added": tuple(int(value) for value in p.runner_rng_draws_added.tolist()),
        "runner_rng_semantic_draws_added": tuple(
            int(value) for value in p.runner_rng_semantic_draws_added.tolist()
        ),
    }
    if bundle.events is not None:
        counts["event_active_count"] = tuple(
            int(value) for value in bundle.events.active_count.tolist()
        )
    return counts


def _candidate_death_counts_v1(bundle: TensorWorldStateBundleV4) -> tuple[int, ...]:
    core = bundle.base.base.core
    lifecycle = bundle.base.lifecycle
    return tuple(
        int(
            (
                core.agents.alive[world]
                & (core.agents.agent_id[world] >= 0)
                & lifecycle.death_candidate[world]
            ).sum()
        )
        for world in range(bundle.batch)
    )


def _phase_counts_v1(bundle: TensorWorldStateBundleV4) -> dict[str, dict[str, tuple[int, ...]]]:
    p = bundle.population
    return {
        C4_PHASE_ORDER[0]: {"death_candidates": _candidate_death_counts_v1(bundle)},
        C4_PHASE_ORDER[1]: {"death_count": tuple(int(value) for value in p.death_count.tolist())},
        C4_PHASE_ORDER[2]: {"free_queue_count": tuple(int(value) for value in p.free_queue_count.tolist())},
        C4_PHASE_ORDER[3]: {
            "decision_count": tuple(int(value) for value in p.decision_count.tolist()),
            "planned_birth_count": tuple(int(value) for value in p.planned_count.tolist()),
            "spawn_rng_draws_added": tuple(
                int(value) for value in p.spawn_rng_draws_added.tolist()
            ),
            "runner_rng_draws_added": tuple(
                int(value) for value in p.runner_rng_draws_added.tolist()
            ),
        },
        C4_PHASE_ORDER[4]: {"birth_count": tuple(int(value) for value in p.birth_count.tolist())},
        C4_PHASE_ORDER[5]: {"active_count": tuple(int(value) for value in p.active_count.tolist())},
        C4_PHASE_ORDER[6]: {"active_count": tuple(int(value) for value in p.active_count.tolist())},
        C4_PHASE_ORDER[7]: {"alive_count": tuple(int(value) for value in bundle.base.base.core.agents.alive.sum(dim=1).tolist())},
        C4_PHASE_ORDER[8]: {
            "event_active_count": (
                tuple(int(value) for value in bundle.events.active_count.tolist())
                if bundle.events is not None
                else tuple(0 for _ in range(bundle.batch))
            )
        },
    }


def _receipt_v1(
    *,
    status: str,
    phase_status: Mapping[str, str],
    phase_counts: Mapping[str, Mapping[str, tuple[int, ...]]],
    counts_before: Mapping[str, tuple[int, ...]],
    counts_after: Mapping[str, tuple[int, ...]],
    failure_phase: str | None = None,
    failure: BaseException | None = None,
    rollback_applied: bool = False,
) -> C4PopulationTransactionReceiptV1:
    return C4PopulationTransactionReceiptV1(
        status=status,
        phase_status=tuple((phase, phase_status[phase]) for phase in C4_PHASE_ORDER),
        phase_counts=phase_counts,
        counts_before=counts_before,
        counts_after=counts_after,
        failure_phase=failure_phase,
        failure_type=None if failure is None else type(failure).__name__,
        failure_message=None if failure is None else str(failure),
        rollback_applied=rollback_applied,
    )


def execute_c4_population_transaction_oracle_v1(
    bundle: TensorWorldStateBundleV4,
    *,
    death_config: TensorWorldC4PopulationConfigV1 | None = None,
    birth_config: TensorWorldC4BirthConfigV1 | None = None,
    spawn_uniforms: torch.Tensor | None = None,
    rng_context: "TensorWorldC5ReplayContextV1 | None" = None,
    rng_stream: int = 0,
    mate_config: "TensorWorldC4MateBirthConfigV1 | None" = None,
) -> C4PopulationTransactionReceiptV1:
    """Execute one ordered C4 CPU-oracle population transaction.

    Death settlement always precedes an optional ordinary or mate birth route.
    Supplying both birth configurations is rejected.  On success the receipt
    records every source phase and per-world counts.  On any failure all tensor
    leaves in ``bundle`` and ``rng_context`` are restored before a
    :class:`C4PopulationTransactionError` is raised.
    """

    if not isinstance(bundle, TensorWorldStateBundleV4):
        raise TensorWorldContractError("C4 transaction requires a C4 state bundle")
    bundle.validate()
    if bundle.device.type != "cpu":
        raise TensorWorldContractError(
            "C4 population transaction oracle requires a CPU preparation fixture"
        )
    if birth_config is not None and mate_config is not None:
        raise TensorWorldContractError(
            "C4 transaction cannot combine ordinary and mate birth configurations"
        )
    if rng_context is not None:
        from .c5_determinism import TensorWorldC5ReplayContextV1

        if not isinstance(rng_context, TensorWorldC5ReplayContextV1):
            raise TensorWorldContractError("C4 transaction RNG context has an unsupported type")

    snapshot: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    _snapshot_tensor_leaves(bundle, snapshot)
    if rng_context is not None:
        _snapshot_tensor_leaves(rng_context, snapshot)
    counts_before = _counts_v1(bundle)
    phase_status = {phase: "PENDING" for phase in C4_PHASE_ORDER}
    current_phase = C4_PHASE_ORDER[0]
    phase_status[current_phase] = "PASS"
    phase_counts = _phase_counts_v1(bundle)
    phase_counts[current_phase] = {"death_candidates": _candidate_death_counts_v1(bundle)}

    try:
        current_phase = C4_PHASE_ORDER[1]
        effective_death_config = death_config or TensorWorldC4PopulationConfigV1()
        apply_c4_death_settlement_oracle_v1(
            bundle,
            replace(effective_death_config, release_free_slots=False),
            event_sequence=(
                None if rng_context is None else rng_context.event_sequence
            ),
        )
        phase_status[current_phase] = "PASS"
        phase_counts[current_phase] = {
            "death_count": tuple(int(value) for value in bundle.population.death_count.tolist())
        }

        current_phase = C4_PHASE_ORDER[2]
        if effective_death_config.release_free_slots:
            release_c4_free_slots_oracle_v1(bundle)
            phase_status[current_phase] = "PASS"
        else:
            phase_status[current_phase] = "SKIPPED"
        phase_counts[current_phase] = {
            "free_queue_count": tuple(
                int(value) for value in bundle.population.free_queue_count.tolist()
            )
        }

        if birth_config is None and mate_config is None:
            for phase in C4_PHASE_ORDER[3:]:
                phase_status[phase] = "SKIPPED"
                phase_counts[phase] = {}
            # Death settlement publishes its typed lifecycle rows even when
            # no birth route is selected.  Keep the receipt faithful to the
            # executed phase rather than reporting event emission as skipped.
            if bundle.events is not None:
                phase_status[C4_PHASE_ORDER[8]] = "PASS"
                phase_counts[C4_PHASE_ORDER[8]] = {
                    "event_active_count": tuple(
                        int(value) for value in bundle.events.active_count.tolist()
                    )
                }
        else:
            current_phase = C4_PHASE_ORDER[3]
            if mate_config is not None:
                from .c4_mating import plan_c4_mate_births_oracle_v1

                plan_c4_mate_births_oracle_v1(
                    bundle,
                    mate_config,
                    rng_context=rng_context,
                )
            else:
                assert birth_config is not None
                plan_c4_ordinary_births_oracle_v1(
                    bundle,
                    birth_config,
                    spawn_uniforms=spawn_uniforms,
                    rng_context=rng_context,
                    rng_stream=rng_stream,
                )
            phase_status[current_phase] = "PASS"
            phase_counts[current_phase] = {
                "decision_count": tuple(
                    int(value) for value in bundle.population.decision_count.tolist()
                ),
                "planned_birth_count": tuple(
                    int(value) for value in bundle.population.planned_count.tolist()
                ),
                "spawn_rng_draws_added": tuple(
                    int(value) for value in bundle.population.spawn_rng_draws_added.tolist()
                ),
                "runner_rng_draws_added": tuple(
                    int(value) for value in bundle.population.runner_rng_draws_added.tolist()
                ),
            }

            current_phase = C4_PHASE_ORDER[4]
            if mate_config is not None:
                from .c4_mating import commit_c4_mate_births_oracle_v1

                commit_c4_mate_births_oracle_v1(
                    bundle,
                    mate_config,
                    rng_context=rng_context,
                )
            else:
                assert birth_config is not None
                commit_c4_ordinary_births_oracle_v1(
                    bundle,
                    birth_config,
                    replay_context=rng_context,
                )
            phase_status[current_phase] = "PASS"
            phase_counts[current_phase] = {
                "birth_count": tuple(
                    int(value) for value in bundle.population.birth_count.tolist()
                )
            }
            for phase in C4_PHASE_ORDER[5:]:
                phase_status[phase] = "PASS"
            if bundle.events is None:
                phase_status[C4_PHASE_ORDER[8]] = "SKIPPED"
            post_commit_counts = _phase_counts_v1(bundle)
            for phase in C4_PHASE_ORDER[5:]:
                phase_counts[phase] = post_commit_counts[phase]

        bundle.validate()
        counts_after = _counts_v1(bundle)
        return _receipt_v1(
            status=C4_TRANSACTION_PREPARATION_STATUS_V1,
            phase_status=phase_status,
            phase_counts=phase_counts,
            counts_before=counts_before,
            counts_after=counts_after,
        )
    except Exception as exc:
        phase_status[current_phase] = "FAILED"
        for phase in C4_PHASE_ORDER:
            if phase_status[phase] == "PENDING":
                phase_status[phase] = "ROLLED_BACK"
        _restore_tensor_leaves(snapshot)
        counts_after = _counts_v1(bundle)
        receipt = _receipt_v1(
            status=C4_TRANSACTION_ROLLED_BACK_STATUS_V1,
            phase_status=phase_status,
            phase_counts=phase_counts,
            counts_before=counts_before,
            counts_after=counts_after,
            failure_phase=current_phase,
            failure=exc,
            rollback_applied=True,
        )
        raise C4PopulationTransactionError(
            f"C4 population transaction failed in {current_phase}: {exc}", receipt
        ) from exc


__all__ = [
    "C4_BOUNDARY_SURFACE_PATHS_V1",
    "C4_POPULATION_BOUNDARY_SCHEMA_V1",
    "C4PopulationTransactionError",
    "C4PopulationBoundarySnapshotV1",
    "C4_POPULATION_TRANSACTION_RECEIPT_SCHEMA_V1",
    "C4_TRANSACTION_DECISION_V1",
    "C4_TRANSACTION_PREPARATION_STATUS_V1",
    "C4_TRANSACTION_ROLLED_BACK_STATUS_V1",
    "C4_TRANSACTION_ROLLBACK_SCOPE_V1",
    "C4PopulationTransactionReceiptV1",
    "capture_c4_population_boundary_v1",
    "compare_c4_population_boundaries_v1",
    "execute_c4_population_transaction_oracle_v1",
]
