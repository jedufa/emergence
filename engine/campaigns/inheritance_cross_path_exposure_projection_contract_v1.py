"""Compose the generic resident reproduction-exposure projection contract."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Any, Final, Mapping

from engine.perf.cuda_architecture_b_direct_resident_source_allocator_v1 import (
    PHYSICAL_WINDOW_GENERATIONS_V1,
    TICKS_PER_GENERATION_V1,
)

from .canonical import canonical_sha256, pretty_json


SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_exposure_projection_contract_v3"
)
AGENT_ID_ALLOCATION_SCHEMA_VERSION: Final[str] = (
    "eiel_resident_exposure_compact_key_capacity_envelope_v2"
)
MAX_NATIVE_INT64: Final[int] = (1 << 63) - 1
TURNOVER_ID_GROWTH_BOUND_PER_GENERATION_V1: Final[int] = 64


class InheritanceCrossPathExposureProjectionContractError(ValueError):
    """The projection contract cannot be composed or published safely."""


def _fail(message: str) -> InheritanceCrossPathExposureProjectionContractError:
    return InheritanceCrossPathExposureProjectionContractError(message)


def _positive(value: Any, *, field: str) -> int:
    if type(value) is not int or not 1 <= value <= MAX_NATIVE_INT64:
        raise _fail(f"{field} must be a positive native int64")
    return value


def required_reproduction_exposure_aggregate_capacity_v1(
    current_capacity: object,
    population_next_id: object,
    max_births_per_tick: object,
    slot_capacity: object,
    *,
    allocation_generation_count: object,
    retained_capacity: object,
) -> int:
    """Bound compact exposure keys by slots plus the rounded birth horizon.

    ``population_next_id`` remains validated identity state. Its numeric
    watermark does not size resident aggregate storage.
    """

    for field, value in (
        ("current_capacity", current_capacity),
        ("slot_capacity", slot_capacity),
        ("retained_capacity", retained_capacity),
    ):
        if type(value) is not int or not 1 <= value <= MAX_NATIVE_INT64:
            raise _fail(f"{field} must be a positive native int64")
    if (
        type(population_next_id) is not int
        or not 1 <= population_next_id <= 1 << 31
    ):
        raise _fail("population_next_id must be in [1, INT32_MAX + 1]")
    if (
        type(max_births_per_tick) is not int
        or not 1 <= max_births_per_tick <= slot_capacity
    ):
        raise _fail("max_births_per_tick must be in [1, slot_capacity]")
    if (
        type(allocation_generation_count) is not int
        or not PHYSICAL_WINDOW_GENERATIONS_V1
        <= allocation_generation_count
        <= 128
    ):
        raise _fail("allocation_generation_count must be in [2, 128]")
    if retained_capacity < current_capacity:
        raise _fail("retained_capacity must cover current_capacity")
    rounded_horizon = (
        allocation_generation_count + PHYSICAL_WINDOW_GENERATIONS_V1 - 1
    ) // PHYSICAL_WINDOW_GENERATIONS_V1 * PHYSICAL_WINDOW_GENERATIONS_V1
    if (
        max_births_per_tick
        > (
            MAX_NATIVE_INT64
            - TURNOVER_ID_GROWTH_BOUND_PER_GENERATION_V1
        )
        // TICKS_PER_GENERATION_V1
    ):
        raise _fail("compact-key per-generation growth exceeds native int64")
    growth_per_generation = (
        TICKS_PER_GENERATION_V1 * max_births_per_tick
        + TURNOVER_ID_GROWTH_BOUND_PER_GENERATION_V1
    )
    if rounded_horizon > MAX_NATIVE_INT64 // growth_per_generation:
        raise _fail("compact-key generation horizon exceeds native int64")
    remaining_growth = rounded_horizon * growth_per_generation
    if slot_capacity > MAX_NATIVE_INT64 - remaining_growth:
        raise _fail("compact-key capacity exceeds native int64")
    maximum_growth = 128 * growth_per_generation
    if (
        slot_capacity > MAX_NATIVE_INT64 - maximum_growth
        or retained_capacity > slot_capacity + maximum_growth
    ):
        raise _fail(
            "retained_capacity exceeds the supported 128-generation "
            "compact-key envelope"
        )
    return max(
        current_capacity,
        retained_capacity,
        slot_capacity + remaining_growth,
    )


@dataclass(frozen=True, slots=True)
class ExposureProjectionEnvelopeV1:
    """Typed campaign-independent bounds for one drained resident segment."""

    max_batch: int
    segment_generations: int
    storage_mode: str
    attempt_capacity: int
    snapshot_capacity: int
    eligible_capacity: int
    aggregate_capacity: int
    agent_id_current_capacity: int
    population_next_id: int
    max_births_per_tick: int
    slot_capacity: int
    allocation_generation_count: int
    retained_agent_id_capacity: int

    def __post_init__(self) -> None:
        for field in (
            "max_batch",
            "segment_generations",
            "attempt_capacity",
            "snapshot_capacity",
            "eligible_capacity",
            "aggregate_capacity",
            "agent_id_current_capacity",
            "population_next_id",
            "max_births_per_tick",
            "slot_capacity",
            "allocation_generation_count",
            "retained_agent_id_capacity",
        ):
            _positive(getattr(self, field), field=field)
        if self.storage_mode not in {"exact_full", "compact_complete"}:
            raise _fail("storage_mode is unsupported")
        if self.storage_mode == "exact_full":
            if self.attempt_capacity > MAX_NATIVE_INT64 // 2:
                raise _fail(
                    "twice exact-full attempt_capacity exceeds native int64"
                )
            if self.snapshot_capacity < 2 * self.attempt_capacity:
                raise _fail(
                    "exact-full snapshot_capacity must be at least twice "
                    "attempt_capacity"
                )
        required = self.required_aggregate_capacity
        if self.aggregate_capacity < required:
            raise _fail(
                "aggregate_capacity is below required agent-ID allocation "
                f"capacity: declared={self.aggregate_capacity} required={required}"
            )

    @property
    def required_aggregate_capacity(self) -> int:
        return required_reproduction_exposure_aggregate_capacity_v1(
            self.agent_id_current_capacity,
            self.population_next_id,
            self.max_births_per_tick,
            self.slot_capacity,
            allocation_generation_count=self.allocation_generation_count,
            retained_capacity=self.retained_agent_id_capacity,
        )

    def payload(self) -> dict[str, Any]:
        return {
            "max_batch": self.max_batch,
            "segment_generations": self.segment_generations,
            "storage_mode": self.storage_mode,
            "attempt_capacity": self.attempt_capacity,
            "snapshot_capacity": self.snapshot_capacity,
            "eligible_capacity": self.eligible_capacity,
            "aggregate_capacity": self.aggregate_capacity,
            "agent_id_allocation": {
                "schema_version": AGENT_ID_ALLOCATION_SCHEMA_VERSION,
                "current_capacity": self.agent_id_current_capacity,
                "population_next_id": self.population_next_id,
                "max_births_per_tick": self.max_births_per_tick,
                "slot_capacity": self.slot_capacity,
                "allocation_generation_count": (
                    self.allocation_generation_count
                ),
                "retained_capacity": self.retained_agent_id_capacity,
                "required_aggregate_capacity": (
                    self.required_aggregate_capacity
                ),
            },
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any]
    ) -> ExposureProjectionEnvelopeV1:
        if not isinstance(payload, Mapping):
            raise _fail("projection envelope must be an object")
        expected = {
            "max_batch",
            "segment_generations",
            "storage_mode",
            "attempt_capacity",
            "snapshot_capacity",
            "eligible_capacity",
            "aggregate_capacity",
            "agent_id_allocation",
        }
        if set(payload) != expected:
            raise _fail("projection envelope field set mismatch")
        allocation = payload.get("agent_id_allocation")
        allocation_fields = {
            "schema_version",
            "current_capacity",
            "population_next_id",
            "max_births_per_tick",
            "slot_capacity",
            "allocation_generation_count",
            "retained_capacity",
            "required_aggregate_capacity",
        }
        if (
            not isinstance(allocation, Mapping)
            or set(allocation) != allocation_fields
            or allocation.get("schema_version")
            != AGENT_ID_ALLOCATION_SCHEMA_VERSION
        ):
            raise _fail("agent-ID allocation envelope field set mismatch")
        envelope = cls(
            max_batch=payload["max_batch"],
            segment_generations=payload["segment_generations"],
            storage_mode=payload["storage_mode"],
            attempt_capacity=payload["attempt_capacity"],
            snapshot_capacity=payload["snapshot_capacity"],
            eligible_capacity=payload["eligible_capacity"],
            aggregate_capacity=payload["aggregate_capacity"],
            agent_id_current_capacity=allocation["current_capacity"],
            population_next_id=allocation["population_next_id"],
            max_births_per_tick=allocation["max_births_per_tick"],
            slot_capacity=allocation["slot_capacity"],
            allocation_generation_count=allocation[
                "allocation_generation_count"
            ],
            retained_agent_id_capacity=allocation["retained_capacity"],
        )
        if envelope.payload() != dict(payload):
            raise _fail("agent-ID allocation derived capacity mismatch")
        return envelope


def _source_contract() -> dict[str, Any]:
    from engine.perf import (  # noqa: PLC0415
        cuda_architecture_b_resident_reproduction_exposure_sidecar_v1 as source,
    )

    from .inheritance_cross_path_patch_execution_v1 import (  # noqa: PLC0415
        EXPOSURE_SEGMENT_FIELD,
    )

    return {
        "schemas": {
            "resident_owner": source.SCHEMA_VERSION,
            "checkpoint": source.CHECKPOINT_SCHEMA_VERSION,
            "projector": source.PROJECTOR_SCHEMA_VERSION,
            "continuity": source.CONTINUITY_SCHEMA_VERSION,
            "segment_output": source.SEGMENT_OUTPUT_SCHEMA_VERSION,
            "segment_drain": source.SEGMENT_DRAIN_SCHEMA_VERSION,
        },
        "paths": {
            "generation_boundary_refill": (
                source.PATH_GENERATION_BOUNDARY_REFILL
            ),
            "ordinary_live_birth": source.PATH_ORDINARY_LIVE_BIRTH,
        },
        "parent_roles": {
            "primary": source.PARENT_ROLE_PRIMARY,
            "mate": source.PARENT_ROLE_MATE,
            "count": source.PARENT_ROLE_COUNT,
        },
        "storage_modes": dict(sorted(source.STORAGE_MODE_CODES.items())),
        "tensor_fields": list(source.TENSOR_NAMES_V1),
        "attempt_projection_fields": [
            "attempt_index",
            "path",
            "generation",
            "tick",
            "eligible_snapshot_refs",
            "eligible_parent_counts",
            "selected_parent_ids",
            "success",
            "failure_code",
            "commit_ordinal",
            "authoritative_commit",
        ],
        "authoritative_commit_fields": [
            "generation",
            "tick",
            "path_id",
            "selected_parent_ids",
            "child_agent_id",
            "donor_draw_ordinals",
            "commit_authority_sha256",
        ],
        "cell_projection_fields": [
            "cell_index",
            "enabled",
            "status",
            "attempt_count",
            "success_count",
            "failure_count",
            "snapshot_count",
            "attempts",
            "eligible_parent_snapshots",
            "selected_parent_counts",
            "parent_contribution_counts",
            "primary_eligibility_opportunity_counts",
            "mate_eligibility_opportunity_counts",
            "cumulative_counts_after_segment",
            "segment_sha256",
            "previous_chain_sha256",
            "chain_sha256_after_segment",
        ],
        "continuity_fields": [
            "segment_count",
            "cell_chain_sha256",
            "cumulative_attempt_counts",
            "cumulative_success_counts",
            "cumulative_failure_counts",
            "cumulative_snapshot_counts",
        ],
        "sentinels": {
            "no_path": source.NO_PATH,
            "no_reference": source.NO_REFERENCE,
            "no_agent_id": source.NO_AGENT_ID,
            "no_generation": source.NO_GENERATION,
            "no_tick": source.NO_TICK,
            "no_failure": source.NO_FAILURE,
            "inactive_failure": source.INACTIVE_FAILURE,
            "no_commit_ordinal": source.NO_COMMIT_ORDINAL,
            "no_donor_ordinal": source.NO_DONOR_ORDINAL,
            "donor_ordinal_capacity": source.DONOR_ORDINAL_CAPACITY,
        },
        "cell_statuses": {
            "ok": source.CELL_STATUS_OK,
            "disabled": source.CELL_STATUS_DISABLED,
            "attempt_capacity_exceeded": (
                source.CELL_STATUS_ATTEMPT_CAPACITY_EXCEEDED
            ),
            "snapshot_capacity_exceeded": (
                source.CELL_STATUS_SNAPSHOT_CAPACITY_EXCEEDED
            ),
            "aggregate_capacity_exceeded": (
                source.CELL_STATUS_AGGREGATE_CAPACITY_EXCEEDED
            ),
            "authoritative_commit_mismatch": (
                source.CELL_STATUS_AUTHORITATIVE_COMMIT_MISMATCH
            ),
        },
        "transaction_statuses": {
            "pending_commit": source.FAILURE_PENDING_COMMIT,
            "authoritative_commit_aborted": (
                source.FAILURE_AUTHORITATIVE_COMMIT_ABORTED
            ),
        },
        "checkpoint_projection_field": EXPOSURE_SEGMENT_FIELD,
    }


def compose_inheritance_cross_path_exposure_projection_contract_v1(
    envelope: ExposureProjectionEnvelopeV1,
) -> dict[str, Any]:
    """Reject composition of a new retired-route exposure contract."""

    raise _fail(
        "P12_RETIRED_INHERITANCE_AUTHORITY "
        "cross-path exposure contract composition is disabled"
    )
    # Historical implementation retained temporarily for evidence reading.
    if not isinstance(envelope, ExposureProjectionEnvelopeV1):
        raise _fail("envelope must be ExposureProjectionEnvelopeV1")
    source_contract = _source_contract()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "envelope": envelope.payload(),
        "source_contract": source_contract,
        "source_contract_sha256": canonical_sha256(source_contract),
        "eligibility_semantics": {
            "primary_and_mate_separate": True,
            "exact_full": {
                "eligible_parent_snapshots_present": True,
                "attempt_snapshot_references_present": True,
                "role_specific_snapshots_per_attempt": 2,
            },
            "compact_complete": {
                "eligible_parent_snapshots_present": False,
                "snapshot_allocation_assumed": False,
                "attempt_role_counts_present": True,
                "role_opportunity_aggregates_present": True,
            },
        },
        "outcome_semantics": {
            "attempts_include_failures": True,
            "success_requires_authoritative_commit": True,
            "committed_child_agent_id_present": True,
            "commit_ordinal_present": True,
            "donor_ordinals_present": True,
            "parent_contributions_derived_from_successes": True,
        },
        "aggregate_semantics": {
            "selected_parent_counts": "all_attempts",
            "parent_contribution_counts": "successful_commits",
            "eligibility_opportunities": ["primary", "mate"],
            "compact_index_authority": "stable_agent_id",
        },
        "checkpoint_and_drain": {
            "checkpoint_hash_bound": True,
            "raw_rows_emitted_once_per_segment": True,
            "continuation_retains_empty_fixed_capacity_buffers": True,
            "continuation_retains_cumulative_counts": True,
            "continuation_retains_per_cell_segment_hash_chain": True,
            "overflow_status_cell_local_sticky": True,
        },
        "chronicle_projection": {
            "source": "settlement_segment_artifact",
            "checkpoint_projection_field": source_contract[
                "checkpoint_projection_field"
            ],
            "mode": "post_run_read_only_projection",
            "mechanics_feedback": False,
            "rng_draws": 0,
            "requires_explicit_consumer": True,
        },
        "reporting_independence": {
            "native_capture_owner": source_contract["schemas"]["resident_owner"],
            "projection_read_only": True,
            "reporter_is_not_mechanics_authority": True,
            "mechanics_feedback": False,
            "rng_draws": 0,
        },
        "fail_closed": {
            "cell_statuses": source_contract["cell_statuses"],
            "pending_commit_projection_rejected": True,
            "capacity_overflow_is_not_truncation": True,
            "authoritative_commit_mismatch_is_terminal_status": True,
            "disabled_cells_are_explicit": True,
        },
    }
    payload["contract_sha256"] = canonical_sha256(payload)
    return payload


def publish_inheritance_cross_path_exposure_projection_contract_v1(
    output_path: Path,
    envelope: ExposureProjectionEnvelopeV1,
) -> dict[str, Any]:
    """Reject publication of a new retired-route exposure contract."""

    raise _fail(
        "P12_RETIRED_INHERITANCE_AUTHORITY "
        "cross-path exposure contract publication is disabled"
    )
    # Historical implementation retained temporarily for evidence reading.
    if not output_path.is_absolute():
        raise _fail("output_path must be absolute")
    for candidate in (output_path, *output_path.parents):
        if candidate.is_symlink():
            raise _fail("output_path must not contain symlinks")
    if output_path.exists():
        raise _fail(f"refusing to overwrite projection contract: {output_path}")
    if not output_path.parent.is_dir():
        raise _fail("output_path parent must already exist")

    payload = compose_inheritance_cross_path_exposure_projection_contract_v1(
        envelope
    )
    temporary: Path | None = None
    try:
        descriptor, raw_temporary = tempfile.mkstemp(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
        )
        temporary = Path(raw_temporary)
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(pretty_json(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output_path)
        directory = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as exc:
        raise _fail(
            f"refusing to overwrite projection contract: {output_path}"
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return payload


def validate_inheritance_cross_path_exposure_projection_contract_v1(
    payload: Mapping[str, Any],
) -> None:
    """Fail closed when a durable contract is stale or internally inconsistent."""

    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION:
        raise _fail("projection contract schema mismatch")
    ExposureProjectionEnvelopeV1.from_payload(payload.get("envelope"))
    source_contract = payload.get("source_contract")
    if (
        not isinstance(source_contract, Mapping)
        or dict(source_contract) != _source_contract()
        or payload.get("source_contract_sha256")
        != canonical_sha256(source_contract)
    ):
        raise _fail("projection contract source binding is stale")
    unsigned = dict(payload)
    declared = unsigned.pop("contract_sha256", None)
    if declared != canonical_sha256(unsigned):
        raise _fail("projection contract digest mismatch")


__all__ = [
    "AGENT_ID_ALLOCATION_SCHEMA_VERSION",
    "ExposureProjectionEnvelopeV1",
    "InheritanceCrossPathExposureProjectionContractError",
    "SCHEMA_VERSION",
    "TURNOVER_ID_GROWTH_BOUND_PER_GENERATION_V1",
    "compose_inheritance_cross_path_exposure_projection_contract_v1",
    "publish_inheritance_cross_path_exposure_projection_contract_v1",
    "required_reproduction_exposure_aggregate_capacity_v1",
    "validate_inheritance_cross_path_exposure_projection_contract_v1",
]
