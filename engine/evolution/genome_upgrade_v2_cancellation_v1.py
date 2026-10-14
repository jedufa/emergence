"""Cancellation cut-point policy for the unified Genome V2 package."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final


CANCELLATION_CONTRACT_SCHEMA_VERSION: Final[str] = "genome_v2_cancellation_contract_v1"
CANCELLATION_CUT_POINTS: Final[tuple[str, ...]] = (
    "pre_expression",
    "post_expression",
    "pre_action",
    "post_action",
    "birth_reservation",
    "birth_staging",
    "birth_commit_boundary",
    "turnover_staging",
    "turnover_commit_boundary",
    "checkpoint_serialization",
    "disk_write",
    "sidecar_restore",
    "chronicle_closeout",
    "report_closeout",
    "batch_transition",
)
CANCELLATION_LIFECYCLE_CASES: Final[tuple[str, ...]] = (
    "before_submission_or_owner_start",
    "after_owner_creation_before_first_generation",
    "before_resident_dispatch",
    "before_ordinary_birth_transaction",
    "during_ordinary_birth_transaction",
    "immediately_after_ordinary_birth_commit",
    "before_turnover_transaction",
    "during_turnover_transaction",
    "immediately_after_turnover_commit",
    "before_generation_boundary_sealing",
    "supported_generation_boundary_cancellation",
    "checkpoint_on_cancel_preparation",
    "after_checkpoint_publication_before_terminal_response",
    "repeated_cancellation_after_acceptance",
    "contradictory_operation_after_terminal_sealing",
)

HISTORICAL_CUT_POINT_REPLACEMENTS: Final[dict[str, tuple[str, ...]]] = {
    "pre_expression": ("before_resident_dispatch",),
    "post_expression": ("before_ordinary_birth_transaction",),
    "pre_action": ("before_ordinary_birth_transaction",),
    "post_action": ("during_ordinary_birth_transaction",),
    "birth_reservation": ("before_ordinary_birth_transaction",),
    "birth_staging": ("during_ordinary_birth_transaction",),
    "birth_commit_boundary": ("immediately_after_ordinary_birth_commit",),
    "turnover_staging": ("during_turnover_transaction",),
    "turnover_commit_boundary": ("immediately_after_turnover_commit",),
    "checkpoint_serialization": ("checkpoint_on_cancel_preparation",),
    "disk_write": ("checkpoint_on_cancel_preparation",),
    "sidecar_restore": ("after_checkpoint_publication_before_terminal_response",),
    "chronicle_closeout": ("after_checkpoint_publication_before_terminal_response",),
    "report_closeout": ("after_checkpoint_publication_before_terminal_response",),
    "batch_transition": ("contradictory_operation_after_terminal_sealing",),
}

_TRAJECTORY_PHASES: Final[frozenset[str]] = frozenset(
    {
        "pre_expression",
        "post_expression",
        "pre_action",
        "post_action",
    }
)
_TRANSACTION_PHASES: Final[frozenset[str]] = frozenset(
    {
        "birth_reservation",
        "birth_staging",
        "birth_commit_boundary",
        "turnover_staging",
        "turnover_commit_boundary",
    }
)
_CHECKPOINT_PHASES: Final[frozenset[str]] = frozenset(
    {"checkpoint_serialization", "disk_write", "sidecar_restore"}
)
_CLOSEOUT_PHASES: Final[frozenset[str]] = frozenset(
    {"chronicle_closeout", "report_closeout"}
)


class GenomeV2CancellationContractError(ValueError):
    """A cancellation cut-point or policy value is invalid."""


@dataclass(frozen=True, slots=True)
class GenomeV2CancellationCutPointV1:
    case_id: str
    request_arrival_phase: str
    observation_boundary: str
    arrival_disposition: str
    atomic_owner: str
    checkpoint_on_cancel: str
    post_observation_rng: str = "forbidden"
    post_observation_mutation: str = "forbidden"
    post_terminal_submission: str = "forbidden"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GenomeV2CancellationLifecycleCaseV1:
    case_id: str
    request_arrival_phase: str
    authoritative_observation_phase: str
    transaction_settlement_phase: str
    terminal_sealing_phase: str
    required_evidence: str
    expected_disposition: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def cancellation_cut_point_v1(
    case_id: str,
) -> GenomeV2CancellationCutPointV1:
    """Resolve one request-arrival phase to its legal observation boundary."""

    if case_id not in CANCELLATION_CUT_POINTS:
        raise GenomeV2CancellationContractError(
            f"unsupported Genome V2 cancellation cut-point: {case_id!r}"
        )
    if case_id in _TRAJECTORY_PHASES:
        return GenomeV2CancellationCutPointV1(
            case_id=case_id,
            request_arrival_phase=case_id,
            observation_boundary="next_completed_generation_terminal",
            arrival_disposition="queued_until_native_safe_boundary",
            atomic_owner="native_cuda_generation_transaction",
            checkpoint_on_cancel="required_at_observation_boundary",
        )
    if case_id in _TRANSACTION_PHASES:
        owner = (
            "unified_resident_ordinary_birth_transaction_v2"
            if case_id.startswith("birth_")
            else "unified_resident_generation_turnover_v1"
        )
        return GenomeV2CancellationCutPointV1(
            case_id=case_id,
            request_arrival_phase=case_id,
            observation_boundary="next_completed_generation_terminal",
            arrival_disposition="deferred_until_atomic_transaction_settles",
            atomic_owner=owner,
            checkpoint_on_cancel="required_at_observation_boundary",
        )
    if case_id in _CHECKPOINT_PHASES:
        disposition = {
            "checkpoint_serialization": ("finish_or_discard_unpublished_serialization"),
            "disk_write": "finish_or_discard_atomic_temporary_file",
            "sidecar_restore": "finish_or_discard_unpublished_restore_set",
        }[case_id]
        return GenomeV2CancellationCutPointV1(
            case_id=case_id,
            request_arrival_phase=case_id,
            observation_boundary="checkpoint_publication_boundary",
            arrival_disposition=disposition,
            atomic_owner="canonical_direct_resident_checkpoint_owner",
            checkpoint_on_cancel=(
                "existing_exact_checkpoint_required_no_partial_acceptance"
            ),
        )
    if case_id in _CLOSEOUT_PHASES:
        return GenomeV2CancellationCutPointV1(
            case_id=case_id,
            request_arrival_phase=case_id,
            observation_boundary="closeout_publication_boundary",
            arrival_disposition="abort_unfinalized_permanent_history",
            atomic_owner="unified_campaign_closeout_owner",
            checkpoint_on_cancel="terminal_checkpoint_already_required",
        )
    return GenomeV2CancellationCutPointV1(
        case_id=case_id,
        request_arrival_phase=case_id,
        observation_boundary="before_next_batch_submission",
        arrival_disposition="reject_next_batch_submission",
        atomic_owner="unified_campaign_batch_dispatch_owner",
        checkpoint_on_cancel="terminal_checkpoint_already_required",
    )


def cancellation_lifecycle_case_v1(
    case_id: str,
) -> GenomeV2CancellationLifecycleCaseV1:
    """Resolve one material request lifecycle case to its evidence boundary."""

    if case_id not in CANCELLATION_LIFECYCLE_CASES:
        raise GenomeV2CancellationContractError(
            f"unsupported Genome V2 cancellation lifecycle case: {case_id!r}"
        )
    if case_id == "before_submission_or_owner_start":
        return GenomeV2CancellationLifecycleCaseV1(
            case_id=case_id,
            request_arrival_phase="production_submission_admission",
            authoritative_observation_phase="before_native_owner_start",
            transaction_settlement_phase="no_transaction_started",
            terminal_sealing_phase="no_terminal_transition",
            required_evidence="production_admission_and_native_not_running_receipts",
            expected_disposition="cancel_before_owner_creation",
        )
    if case_id == "after_owner_creation_before_first_generation":
        return GenomeV2CancellationLifecycleCaseV1(
            case_id=case_id,
            request_arrival_phase="native_owner_initialized",
            authoritative_observation_phase="native_owner_initialized",
            transaction_settlement_phase="no_generation_transaction_started",
            terminal_sealing_phase="no_terminal_transition",
            required_evidence="native_not_running_receipt",
            expected_disposition="reject_not_running",
        )
    if case_id in {
        "before_resident_dispatch",
        "before_ordinary_birth_transaction",
        "during_ordinary_birth_transaction",
        "immediately_after_ordinary_birth_commit",
        "before_turnover_transaction",
        "during_turnover_transaction",
        "immediately_after_turnover_commit",
        "before_generation_boundary_sealing",
    }:
        transaction = (
            "ordinary_birth_transaction_settled"
            if "birth" in case_id
            else (
                "turnover_transaction_settled"
                if "turnover" in case_id
                else "active_generation_transaction_settled"
            )
        )
        evidence = (
            "live_async_latch_plus_birth_atomicity_receipt"
            if "birth" in case_id
            else (
                "live_async_latch_plus_turnover_atomicity_receipt"
                if "turnover" in case_id
                else "live_async_latch_and_generation_boundary_receipts"
            )
        )
        return GenomeV2CancellationLifecycleCaseV1(
            case_id=case_id,
            request_arrival_phase=case_id,
            authoritative_observation_phase="completed_generation_boundary",
            transaction_settlement_phase=transaction,
            terminal_sealing_phase="after_durable_checkpoint_acknowledgement",
            required_evidence=evidence,
            expected_disposition="accepted_then_cancelled_at_safe_boundary",
        )
    if case_id == "supported_generation_boundary_cancellation":
        return GenomeV2CancellationLifecycleCaseV1(
            case_id=case_id,
            request_arrival_phase="post_tick_zero_control_window",
            authoritative_observation_phase="completed_generation_boundary",
            transaction_settlement_phase="active_generation_transaction_settled",
            terminal_sealing_phase="after_durable_checkpoint_acknowledgement",
            required_evidence="live_control_window_cancellation_receipt",
            expected_disposition="accepted_then_cancelled_at_safe_boundary",
        )
    if case_id == "checkpoint_on_cancel_preparation":
        return GenomeV2CancellationLifecycleCaseV1(
            case_id=case_id,
            request_arrival_phase="checkpoint_on_cancel_preparation",
            authoritative_observation_phase="completed_generation_boundary",
            transaction_settlement_phase="checkpoint_publication_atomic",
            terminal_sealing_phase="after_durable_checkpoint_acknowledgement",
            required_evidence="checkpoint_on_cancel_publication_receipt",
            expected_disposition="await_checkpoint_then_cancelled",
        )
    if case_id == "after_checkpoint_publication_before_terminal_response":
        return GenomeV2CancellationLifecycleCaseV1(
            case_id=case_id,
            request_arrival_phase=case_id,
            authoritative_observation_phase="already_observed_at_generation_boundary",
            transaction_settlement_phase="durable_checkpoint_published",
            terminal_sealing_phase="single_sticky_cancelled_receipt",
            required_evidence="checkpoint_acknowledgement_and_terminal_receipts",
            expected_disposition="seal_cancelled_without_further_mutation",
        )
    if case_id == "repeated_cancellation_after_acceptance":
        return GenomeV2CancellationLifecycleCaseV1(
            case_id=case_id,
            request_arrival_phase=case_id,
            authoritative_observation_phase="first_request_identity_retained",
            transaction_settlement_phase="unchanged",
            terminal_sealing_phase="first_request_only",
            required_evidence="idempotent_repeat_request_receipt",
            expected_disposition="already_requested",
        )
    return GenomeV2CancellationLifecycleCaseV1(
        case_id=case_id,
        request_arrival_phase=case_id,
        authoritative_observation_phase="terminal_already_sealed",
        transaction_settlement_phase="no_additional_transaction",
        terminal_sealing_phase="unchanged_sticky_terminal_receipt",
        required_evidence="no_post_terminal_mutation_receipt",
        expected_disposition="already_terminal",
    )


def cancellation_contract_v1() -> dict[str, object]:
    rows = [
        cancellation_cut_point_v1(case_id).to_dict()
        for case_id in CANCELLATION_CUT_POINTS
    ]
    return {
        "schema_version": CANCELLATION_CONTRACT_SCHEMA_VERSION,
        "provider_id": "eiel-production-provider-v1",
        "runtime_profile_id": "eiel-production-runtime-v1",
        "global_genome_profile_id": "genome_upgrade_v2",
        "request_arrival_semantics": (
            "arrival is not authoritative observation; the first request is "
            "observed only at the named atomic boundary"
        ),
        "polling_semantics": (
            "no Python, host, or device cancellation polling inside a CUDA tick "
            "or transaction"
        ),
        "cut_points": rows,
        "lifecycle_cases": [
            cancellation_lifecycle_case_v1(case_id).to_dict()
            for case_id in CANCELLATION_LIFECYCLE_CASES
        ],
        "historical_cut_point_replacements": [
            {
                "historical_cut_point": historical,
                "replacement_lifecycle_cases": list(replacements),
                "disposition": "superseded_by_stronger_lifecycle_evidence",
            }
            for historical, replacements in HISTORICAL_CUT_POINT_REPLACEMENTS.items()
        ],
        "invariants": [
            "no_rng_draw_after_authoritative_observation",
            "no_genome_mutation_after_authoritative_observation",
            "no_partial_birth_or_turnover_publication",
            "no_partial_checkpoint_or_restore_set_accepted",
            "cancelled_or_failed_closeout_is_not_permanent_scientific_history",
            "terminal_state_is_sticky",
            "no_post_terminal_batch_submission",
        ],
    }


__all__ = [
    "CANCELLATION_CONTRACT_SCHEMA_VERSION",
    "CANCELLATION_CUT_POINTS",
    "CANCELLATION_LIFECYCLE_CASES",
    "HISTORICAL_CUT_POINT_REPLACEMENTS",
    "GenomeV2CancellationContractError",
    "GenomeV2CancellationCutPointV1",
    "GenomeV2CancellationLifecycleCaseV1",
    "cancellation_contract_v1",
    "cancellation_cut_point_v1",
    "cancellation_lifecycle_case_v1",
]
