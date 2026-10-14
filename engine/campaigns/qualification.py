"""Explicit qualification identity, reuse, and invalidation receipts.

Qualification belongs to fixed runtime invariants and an admitted physical-batch
range.  Requested batch extent and the broader planning capability are described
by receipts but never added to fixed invariants.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any, Final

from .canonical import canonical_sha256
from .contracts import (
    CampaignPlanV1,
    ExecutionRouteContractV1,
    ExecutionRouteQualificationReceiptV1,
)
from .long_horizon_qualification import (
    BOUNDED_ROUTE_QUALIFIED,
    LONG_HORIZON_RUNABILITY_QUALIFIED,
    SCIENCE_ROUTE_QUALIFIED,
    LongHorizonQualificationError,
    validate_long_horizon_evidence_v1,
    validate_qualification_evidence_binding_v1,
)

QUALIFICATION_REUSE_SCHEMA_VERSION: Final[str] = "eiel_campaign_qualification_reuse_v1"
QUALIFICATION_COMPARISON_SCHEMA_VERSION: Final[str] = (
    "eiel_campaign_qualification_comparison_v1"
)
EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION: Final[str] = "eiel_execution_route_contract_v1"
EXECUTION_ROUTE_QUALIFICATION_SCHEMA_VERSION: Final[str] = (
    "eiel_execution_route_qualification_v1"
)
EXECUTION_ROUTE_COMPARISON_SCHEMA_VERSION: Final[str] = (
    "eiel_execution_route_comparison_v1"
)
EXECUTION_ROUTE_OBLIGATIONS_V1: Final[frozenset[str]] = frozenset(
    {
        "cold_start",
        "first_generation",
        "first_tick",
        "intra_generation",
        "generation_boundary",
        "first_post_boundary_tick",
        "second_post_boundary_tick",
        "post_boundary_policy_cache_refill",
        "ordinary_birth",
        "ordinary_birth_disabled",
        "turnover_refill",
        "multicell_execution",
        "odd_physical_tail",
        "checkpoint_export",
        "disk_checkpoint_reload",
        "fresh_owner_restore",
        "first_post_restore_tick",
        "first_post_restore_refill",
        "terminal_settlement",
        "reporting_parity",
        "alternate_batch_partition",
        "non_generation_divisible_terminal_tick",
        "non_generation_divisible_checkpoint_tick",
        "non_lattice_divisible_terminal_tick",
        "non_lattice_divisible_checkpoint_tick",
        "exact_absolute_tick_checkpoint_export",
        "fresh_owner_partial_generation_restore",
        "fresh_owner_partial_tick_restore",
        "first_committed_tick_after_partial_restore",
        "acs_capture_barrier_measurement",
        "acs2_host_only_capture",
        "async_compact_science_backpressure",
        "async_compact_science_failure_latch",
        "compact_science_host_only_capture",
        "compact_science_checkpoint_prefix_drain",
    }
)
ACS_FOCUSED_ROUTE_OBLIGATIONS_V1: Final[frozenset[str]] = frozenset(
    {
        "async_checkpoint_backpressure",
        "async_checkpoint_failure_latch",
    }
)
COMPACT_SCIENCE_FOCUSED_ROUTE_OBLIGATIONS_V1: Final[frozenset[str]] = frozenset(
    {
        "async_compact_science_backpressure",
        "async_compact_science_failure_latch",
    }
)


class QualificationReceiptError(ValueError):
    """A qualification key or plan cannot produce an honest receipt."""


def acs_checkpoint_staging_source_sha256_v1() -> str:
    source = Path(__file__).with_name("acs_checkpoint_staging.py")
    return hashlib.sha256(source.read_bytes()).hexdigest()


def acs_native_detached_slots_source_sha256_v1() -> str:
    source = Path(__file__).with_name("acs_native_detached_slots_v1.py")
    return hashlib.sha256(source.read_bytes()).hexdigest()


def validate_execution_route_acs_focused_evidence_v1(
    *,
    execution_route_sha256: str,
    covered_obligations: set[str],
    route_obligation_results: Any,
    immutable_snapshot_slots: int,
) -> tuple[set[str], dict[str, Any] | None]:
    focused = covered_obligations & set(ACS_FOCUSED_ROUTE_OBLIGATIONS_V1)
    if not focused:
        return set(), None
    rows = (
        list(route_obligation_results)
        if isinstance(route_obligation_results, Sequence)
        and not isinstance(route_obligation_results, (str, bytes))
        else []
    )
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("schema_version") == "eiel_acs_checkpoint_focused_route_evidence_v1"
    ]
    if len(matches) != 1:
        return set(), None
    row = dict(matches[0])
    raw_covered = row.get("covered_obligations")
    evidence_covered = (
        {value for value in raw_covered if isinstance(value, str) and value}
        if isinstance(raw_covered, Sequence)
        and not isinstance(raw_covered, (str, bytes))
        else set()
    )
    legacy_source_exact = (
        row.get("source_kind") == "current_process_cpu_staging_probe_v1"
        and row.get("implementation")
        == "engine.campaigns.acs_checkpoint_staging.ACSCheckpointStaging"
        and row.get("implementation_source_sha256")
        == acs_checkpoint_staging_source_sha256_v1()
    )
    detached_source_exact = (
        row.get("source_kind") == "detached_file_backed_native_probe_v1"
        and row.get("implementation")
        == (
            "engine.campaigns.acs_native_detached_slots_v1."
            "ACSNativeDetachedSlotsV1"
        )
        and row.get("implementation_source_sha256")
        == acs_native_detached_slots_source_sha256_v1()
    )
    exact = (
        (legacy_source_exact or detached_source_exact)
        and row.get("execution_route_sha256") == execution_route_sha256
        and row.get("target_immutable_snapshot_slots") == immutable_snapshot_slots
        and evidence_covered == focused
        and row.get("missing_obligations") == []
        and row.get("exact") is True
    )
    for obligation, key in (
        ("async_checkpoint_backpressure", "backpressure"),
        ("async_checkpoint_failure_latch", "failure_latch"),
    ):
        if obligation not in focused:
            continue
        detail = row.get(key)
        telemetry = (
            detail.get("terminal_telemetry") if isinstance(detail, Mapping) else None
        )
        if not isinstance(detail, Mapping) or not isinstance(telemetry, Mapping):
            exact = False
            continue
        exact = (
            exact
            and detail.get("exact") is True
            and telemetry.get("slots") == immutable_snapshot_slots
            and telemetry.get("queue_occupancy_high_water") == immutable_snapshot_slots
            and telemetry.get("closed") is True
            and telemetry.get("writer_alive") is False
        )
        if obligation == "async_checkpoint_backpressure":
            fifo_results = detail.get("fifo_results")
            expected_fifo_results = [
                "first",
                *[
                    f"queued-{frontier}"
                    for frontier in range(2, immutable_snapshot_slots + 1)
                ],
                "blocked",
            ]
            exact = (
                exact
                and detail.get("blocked_before_release") is True
                and detail.get("post_release_submit_completed") is True
                and detail.get("producer_thread_joined") is True
                and detail.get("all_slots_occupied_before_block") is True
                and detail.get("blocked_submission_compute_frontier")
                == immutable_snapshot_slots + 1
                and fifo_results == expected_fifo_results
                and detail.get("fifo_persist_order") == expected_fifo_results
                and detail.get("no_overwrite_or_drop") is True
                and type(detail.get("producer_blocked_time_ns")) is int
                and detail.get("producer_blocked_time_ns") > 0
                and telemetry.get("failure_latched") is False
                and telemetry.get("compute_frontier") == immutable_snapshot_slots + 1
                and telemetry.get("durable_checkpoint_frontier")
                == telemetry.get("compute_frontier")
                and telemetry.get("frontier_lag") == 0
                and telemetry.get("queue_occupancy") == 0
                and telemetry.get("slot_states") == ["FREE"] * immutable_snapshot_slots
                and telemetry.get("abandoned_snapshot_count") == 0
                and telemetry.get("aborted") is False
            )
        else:
            exact = (
                exact
                and detail.get("failure_observed") is True
                and detail.get("post_latch_capture_rejected") is True
                and detail.get("abort_joined_writer") is True
                and detail.get("all_slots_occupied_before_failure") is True
                and detail.get("queued_token_count") == immutable_snapshot_slots - 1
                and detail.get("queued_callbacks_never_started") is True
                and detail.get("all_tokens_inaccessible_after_abort") is True
                and telemetry.get("abandoned_snapshot_count")
                == immutable_snapshot_slots
                and telemetry.get("queue_occupancy") == 0
                and telemetry.get("slot_states") == ["FREE"] * immutable_snapshot_slots
                and telemetry.get("failure_latched") is True
                and telemetry.get("aborted") is True
                and telemetry.get("durable_checkpoint_frontier") == -1
            )
    return (focused, row) if exact else (set(), None)


def _execution_route_immutable_snapshot_slots_v1(
    contract: ExecutionRouteContractV1,
) -> int | None:
    components = [
        component
        for component in contract.components
        if component.get("component_id") == "acs_checkpoint_persistence"
    ]
    if len(components) != 1:
        return None
    slots = components[0].get("immutable_snapshot_slots")
    return slots if type(slots) is int and slots in {1, 2, 3} else None


def execution_route_qualification_sidecar_path_v1(
    qualification_receipt_path: str | Path,
    execution_route_sha256: str | None = None,
) -> Path:
    """Return the deterministic route-evidence sidecar for a mechanic receipt.

    The legacy V1 sidecar path is one file beside the mechanic receipt.  A
    single artifact can legitimately support multiple route hashes, so callers
    that know the route hash should use the route-specific path.
    """

    path = Path(qualification_receipt_path)
    if isinstance(execution_route_sha256, str) and execution_route_sha256:
        if len(execution_route_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in execution_route_sha256
        ):
            raise QualificationReceiptError(
                "execution route SHA-256 must be a lowercase 64-character hex string"
            )
        return (
            path.parent
            / "execution_route_qualifications"
            / f"{execution_route_sha256}.json"
        )
    return path.parent / "execution_route_qualification.json"


def execution_route_science_qualification_projection_path_v1(
    qualification_receipt_path: str | Path,
    execution_route_sha256: str | None = None,
) -> Path:
    """Return the immutable-source sibling used for Gate27 promotion.

    Science promotion must never rewrite the route-qualified source sidecar:
    its pre-promotion bytes are an evidence input.  The projection carries
    the additional science scope and is deliberately addressed separately.
    """

    source = execution_route_qualification_sidecar_path_v1(
        qualification_receipt_path,
        execution_route_sha256,
    )
    return source.with_suffix(".science.json")


def read_execution_route_qualification_sidecar_v1(
    qualification_receipt_path: str | Path,
    execution_route_sha256: str | None = None,
) -> dict[str, Any] | None:
    """Read a route sidecar without granting authority when it is absent."""

    route_path = execution_route_qualification_sidecar_path_v1(
        qualification_receipt_path,
        execution_route_sha256,
    )
    science_path = execution_route_science_qualification_projection_path_v1(
        qualification_receipt_path,
        execution_route_sha256,
    )
    paths = [science_path, route_path]
    legacy_path = execution_route_qualification_sidecar_path_v1(
        qualification_receipt_path,
    )
    legacy_science_path = execution_route_science_qualification_projection_path_v1(
        qualification_receipt_path,
    )
    for candidate in (legacy_science_path, legacy_path):
        if candidate not in paths:
            paths.append(candidate)
    path = next((candidate for candidate in paths if candidate.is_file()), None)
    if path is None and execution_route_sha256 is not None:
        # A source-only compatible delta has a different current route hash,
        # so its historic receipt is intentionally stored under the prior
        # hash.  Use it only when this qualification root has one unambiguous
        # route sidecar; status evaluation still compares every component.
        route_dir = legacy_path.parent / "execution_route_qualifications"
        candidates = (
            sorted(
                candidate
                for candidate in route_dir.glob("*.json")
                if candidate.is_file()
                and not candidate.name.endswith(".science.json")
            )
            if route_dir.is_dir()
            else []
        )
        if len(candidates) == 1:
            path = candidates[0]
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationReceiptError(
            f"execution route qualification sidecar is unreadable: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise QualificationReceiptError(
            "execution route qualification sidecar must be an object"
        )
    return dict(value)


def execution_route_qualification_status_v1(
    contract: ExecutionRouteContractV1,
    receipt_or_envelope: (
        ExecutionRouteQualificationReceiptV1 | Mapping[str, Any] | None
    ),
) -> dict[str, Any]:
    """Evaluate exact route authority and report every missing obligation.

    A legacy mechanic receipt with no nested route section remains readable but
    is explicitly unbound.  An invalid, partial, or mismatched route receipt has
    qualification-only authority and cannot authorize execution.
    """

    raw: Any
    if isinstance(receipt_or_envelope, ExecutionRouteQualificationReceiptV1):
        raw = receipt_or_envelope.payload()
    elif isinstance(receipt_or_envelope, Mapping):
        if (
            receipt_or_envelope.get("schema_version")
            == EXECUTION_ROUTE_QUALIFICATION_SCHEMA_VERSION
        ):
            raw = receipt_or_envelope
        else:
            raw = receipt_or_envelope.get("execution_route_qualification")
    else:
        raw = None
    required = sorted(set(contract.required_obligations))
    if not isinstance(raw, Mapping):
        return {
            "authority_state": "legacy_unbound",
            "execution_route_sha256": contract.execution_route_sha256,
            "route_hash_exact": False,
            "provider_binding_exact": False,
            "profile_binding_exact": False,
            "required_obligations": required,
            "covered_obligations": [],
            "missing_obligations": required,
            "qualification_dimensions": [],
            "qualification_evidence_binding": None,
            "qualification_authority_scope": "legacy_unbound",
            "reasons": [
                {
                    "code": "execution_route_qualification_missing",
                    "message": (
                        "the qualification evidence has no execution-route binding"
                    ),
                }
            ],
        }

    covered_raw = raw.get("covered_obligations")
    declared_covered = (
        sorted({value for value in covered_raw if isinstance(value, str) and value})
        if isinstance(covered_raw, Sequence)
        and not isinstance(covered_raw, (str, bytes))
        else []
    )
    focused_claimed = (
        set(declared_covered) & set(required) & set(ACS_FOCUSED_ROUTE_OBLIGATIONS_V1)
    )
    compact_focused_claimed = (
        set(declared_covered)
        & set(required)
        & set(COMPACT_SCIENCE_FOCUSED_ROUTE_OBLIGATIONS_V1)
    )
    immutable_snapshot_slots = _execution_route_immutable_snapshot_slots_v1(contract)
    focused_validated, _focused_result = (
        validate_execution_route_acs_focused_evidence_v1(
            execution_route_sha256=contract.execution_route_sha256,
            covered_obligations=set(declared_covered),
            route_obligation_results=raw.get("route_obligation_results"),
            immutable_snapshot_slots=immutable_snapshot_slots or 0,
        )
    )
    focused_evidence_exact = not focused_claimed or (
        immutable_snapshot_slots is not None and focused_validated == focused_claimed
    )
    # Compact-publication success telemetry intentionally carries no fault
    # injection proof.  A future focused validator must be added before these
    # obligations can be admitted from a route receipt; a claimed name alone
    # can never turn ordinary completion into qualification evidence.
    compact_focused_evidence_exact = not compact_focused_claimed
    covered = sorted(
        set(declared_covered)
        - (focused_claimed if not focused_evidence_exact else set())
        - compact_focused_claimed
    )
    missing = sorted(set(required) - set(covered))
    route_hash_exact = (
        raw.get("execution_route_sha256") == contract.execution_route_sha256
    )
    route_hash_equivalence = _source_only_legacy_route_equivalence_v1(contract, raw)
    route_hash_bound = route_hash_exact or route_hash_equivalence is not None
    provider_exact = raw.get("provider_id") == contract.provider_id
    profile_exact = raw.get("profile_id") == contract.profile_id
    schema_exact = (
        raw.get("schema_version") == EXECUTION_ROUTE_QUALIFICATION_SCHEMA_VERSION
    )
    declared_route_authority = raw.get("authority_state") == "route_qualified"
    reasons: list[dict[str, str]] = []
    for exact, code, message in (
        (
            schema_exact,
            "execution_route_qualification_schema_mismatch",
            "the route qualification schema is unsupported",
        ),
        (
            route_hash_bound,
            "execution_route_hash_mismatch",
            "the route qualification binds another execution route or a "
            "material route component changed",
        ),
        (
            provider_exact,
            "execution_route_provider_mismatch",
            "the route qualification binds another provider",
        ),
        (
            profile_exact,
            "execution_route_profile_mismatch",
            "the route qualification binds another profile",
        ),
        (
            declared_route_authority,
            "execution_route_authority_not_final",
            "the route qualification does not declare final route authority",
        ),
        (
            focused_evidence_exact,
            "execution_route_focused_acs_evidence_invalid",
            "focused ACS route obligations lack exact current-source evidence",
        ),
        (
            compact_focused_evidence_exact,
            "execution_route_focused_compact_science_evidence_missing",
            "compact-science backpressure/failure obligations require dedicated "
            "fault-focused evidence",
        ),
    ):
        if not exact:
            reasons.append({"code": code, "message": message})
    reasons.extend(
        {
            "code": "execution_route_obligation_missing",
            "message": f"missing execution-route obligation: {obligation}",
            "obligation": obligation,
        }
        for obligation in missing
    )
    route_qualified = (
        schema_exact
        and route_hash_bound
        and provider_exact
        and profile_exact
        and declared_route_authority
        and focused_evidence_exact
        and compact_focused_evidence_exact
        and not missing
    )
    allowed_dimensions = {
        BOUNDED_ROUTE_QUALIFIED,
        LONG_HORIZON_RUNABILITY_QUALIFIED,
        SCIENCE_ROUTE_QUALIFIED,
    }
    dimensions_raw = raw.get("qualification_dimensions")
    declared_dimensions = (
        [value for value in dimensions_raw if isinstance(value, str)]
        if isinstance(dimensions_raw, Sequence)
        and not isinstance(dimensions_raw, (str, bytes))
        else []
    )
    declared_dimension_count = (
        len(dimensions_raw)
        if isinstance(dimensions_raw, Sequence)
        and not isinstance(dimensions_raw, (str, bytes))
        else 0
    )
    dimensions_well_formed = (
        len(declared_dimensions) == declared_dimension_count
        and len(declared_dimensions) == len(set(declared_dimensions))
        and set(declared_dimensions).issubset(allowed_dimensions)
        and (
            SCIENCE_ROUTE_QUALIFIED not in declared_dimensions
            or {
                BOUNDED_ROUTE_QUALIFIED,
                LONG_HORIZON_RUNABILITY_QUALIFIED,
            }.issubset(declared_dimensions)
        )
        and (
            LONG_HORIZON_RUNABILITY_QUALIFIED not in declared_dimensions
            or BOUNDED_ROUTE_QUALIFIED in declared_dimensions
        )
    )
    if not dimensions_well_formed:
        reasons.append(
            {
                "code": "execution_route_qualification_dimensions_invalid",
                "message": "qualification dimensions are malformed or non-monotonic",
            }
        )
        declared_dimensions = []

    validated_dimensions: list[str] = []
    evidence_binding: dict[str, Any] | None = None
    if route_qualified and BOUNDED_ROUTE_QUALIFIED in declared_dimensions:
        validated_dimensions.append(BOUNDED_ROUTE_QUALIFIED)
    claims_long_horizon = bool(
        set(declared_dimensions)
        & {LONG_HORIZON_RUNABILITY_QUALIFIED, SCIENCE_ROUTE_QUALIFIED}
    )
    if route_qualified and claims_long_horizon:
        scope = raw.get("long_horizon_scope")
        science_route_attestation: Mapping[str, Any] | None = None
        if SCIENCE_ROUTE_QUALIFIED in declared_dimensions:
            science_scope = raw.get("science_route_scope")
            if not isinstance(science_scope, Mapping):
                scope = None
            else:
                scope = science_scope
                candidate = science_scope.get(
                    "canonical_full_production_route_attestation"
                )
                if isinstance(candidate, Mapping):
                    science_route_attestation = candidate
        evidence = (
            scope.get("validated_evidence") if isinstance(scope, Mapping) else None
        )
        declared_binding = (
            scope.get("evidence_binding") if isinstance(scope, Mapping) else None
        )
        expected_identity: dict[str, Any] = {
            "execution_route_sha256": contract.execution_route_sha256,
            "provider_id": contract.provider_id,
            "profile_id": contract.profile_id,
        }
        try:
            artifact_sha256s = raw.get("artifact_sha256s")
            if artifact_sha256s is None:
                artifact_sha256 = raw.get("artifact_sha256")
                if not isinstance(artifact_sha256, str) or len(artifact_sha256) != 64:
                    raise LongHorizonQualificationError(
                        "long-horizon receipt has no exact artifact identity"
                    )
                artifact_sha256s = {"cuda_extension": artifact_sha256}
            if (
                not isinstance(artifact_sha256s, Mapping)
                or not artifact_sha256s
                or any(
                    not isinstance(name, str)
                    or not name
                    or not isinstance(digest, str)
                    or len(digest) != 64
                    for name, digest in artifact_sha256s.items()
                )
            ):
                raise LongHorizonQualificationError(
                    "long-horizon receipt artifact identity map is invalid"
                )
            expected_identity["artifact_sha256s"] = dict(artifact_sha256s)
            validated = validate_long_horizon_evidence_v1(
                evidence,
                expected_identity=expected_identity,
                canonical_full_production_route_attestation=(
                    science_route_attestation
                ),
            )
            evidence_binding = validate_qualification_evidence_binding_v1(
                declared_binding,
                expected_identity=expected_identity,
            )
            if evidence_binding != validated["evidence_binding"]:
                raise LongHorizonQualificationError(
                    "receipt evidence binding differs from validated evidence"
                )
            supported = set(validated["qualification_dimensions"])
            requested = set(declared_dimensions) - {BOUNDED_ROUTE_QUALIFIED}
            if not requested.issubset(supported):
                raise LongHorizonQualificationError(
                    "receipt claims dimensions not supported by evidence"
                )
        except LongHorizonQualificationError as exc:
            reasons.append(
                {
                    "code": "execution_route_long_horizon_evidence_invalid",
                    "message": str(exc),
                }
            )
            evidence_binding = None
        else:
            for dimension in (
                LONG_HORIZON_RUNABILITY_QUALIFIED,
                SCIENCE_ROUTE_QUALIFIED,
            ):
                if dimension in declared_dimensions:
                    validated_dimensions.append(dimension)

    authority_scope = (
        "science_route"
        if SCIENCE_ROUTE_QUALIFIED in validated_dimensions
        else (
            "long_horizon_runability"
            if LONG_HORIZON_RUNABILITY_QUALIFIED in validated_dimensions
            else (
                "bounded_route"
                if BOUNDED_ROUTE_QUALIFIED in validated_dimensions
                else "legacy_compatibility_or_qualification_only"
            )
        )
    )
    return {
        "authority_state": (
            "route_qualified" if route_qualified else "qualification_only"
        ),
        "execution_route_sha256": contract.execution_route_sha256,
        "route_hash_exact": route_hash_exact,
        "route_hash_equivalence": route_hash_equivalence,
        "provider_binding_exact": provider_exact,
        "profile_binding_exact": profile_exact,
        "required_obligations": required,
        "declared_covered_obligations": declared_covered,
        "covered_obligations": covered,
        "missing_obligations": missing,
        "qualification_dimensions": validated_dimensions,
        "qualification_evidence_binding": evidence_binding,
        "qualification_authority_scope": authority_scope,
        "reasons": reasons,
    }


def _source_only_legacy_route_equivalence_v1(
    current: ExecutionRouteContractV1,
    raw_receipt: Mapping[str, Any],
) -> dict[str, str] | None:
    """Accept explicit inheritance for a provider-source-only legacy delta.

    The historic receipt carries its full route contract.  A current route may
    inherit it only if every component and every obligation still matches after
    omitting the provider file digest from the dedicated source-closure
    component.  This implements the delta-qualification policy without
    treating a new source SHA as automatic semantic invalidation.
    """

    if raw_receipt.get("authority_state") != "route_qualified":
        return None
    historic = raw_receipt.get("route_contract")
    if not isinstance(historic, Mapping):
        return None
    current_payload = current.payload()
    if (
        historic.get("schema_version") != current_payload["schema_version"]
        or historic.get("provider_id") != current.provider_id
        or historic.get("profile_id") != current.profile_id
        or historic.get("required_obligations")
        != current_payload["required_obligations"]
    ):
        return None

    def component_map(value: Any) -> dict[str, dict[str, Any]] | None:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return None
        result: dict[str, dict[str, Any]] = {}
        for row in value:
            if not isinstance(row, Mapping):
                return None
            component_id = row.get("component_id")
            if not isinstance(component_id, str) or not component_id:
                return None
            if component_id in result:
                return None
            # Historic route receipts are JSON (tuples become lists) while
            # current in-memory contracts retain tuples. Compare canonical
            # JSON values so that representation alone cannot defeat an
            # otherwise exact semantic contract comparison.
            result[component_id] = json.loads(
                json.dumps(dict(row), allow_nan=False, sort_keys=True)
            )
        return result

    historic_components = component_map(historic.get("components"))
    current_components = component_map(current_payload["components"])
    if historic_components is None or current_components is None:
        return None
    if set(historic_components) != set(current_components):
        return None

    closure_id = "source_dependency_closure"
    old_closure = historic_components.get(closure_id)
    new_closure = current_components.get(closure_id)
    if old_closure is None or new_closure is None:
        return None
    old_provider_sha = old_closure.pop("provider_sha256", None)
    new_provider_sha = new_closure.pop("provider_sha256", None)
    if (
        not isinstance(old_provider_sha, str)
        or not isinstance(new_provider_sha, str)
        or old_provider_sha == new_provider_sha
        or old_closure != new_closure
    ):
        return None
    historic_components[closure_id] = old_closure
    current_components[closure_id] = new_closure
    if historic_components != current_components:
        return None
    return {
        "schema_version": "eiel_legacy_route_source_equivalence_v1",
        "kind": "provider_sha256_only",
        "historic_provider_sha256": old_provider_sha,
        "current_provider_sha256": new_provider_sha,
    }


def compare_execution_routes_v1(
    old: ExecutionRouteContractV1,
    new: ExecutionRouteContractV1,
) -> dict[str, Any]:
    """Explain route invalidation without invalidating mechanic qualification."""

    old_components = {str(row["component_id"]): dict(row) for row in old.components}
    new_components = {str(row["component_id"]): dict(row) for row in new.components}
    changed: list[dict[str, Any]] = []
    for component_id in sorted(set(old_components) | set(new_components)):
        before = old_components.get(component_id)
        after = new_components.get(component_id)
        if before != after:
            changed.append(
                {
                    "component_id": component_id,
                    "old_present": before is not None,
                    "old": before,
                    "new_present": after is not None,
                    "new": after,
                }
            )
    obligation_dependencies = {
        "policy_cache_refill": {
            "first_post_boundary_tick",
            "second_post_boundary_tick",
            "post_boundary_policy_cache_refill",
            "first_post_restore_refill",
        },
        "rng_and_tape": {
            "first_tick",
            "intra_generation",
            "first_post_boundary_tick",
            "second_post_boundary_tick",
            "first_post_restore_tick",
        },
        "birth_and_turnover": {
            "ordinary_birth",
            "ordinary_birth_disabled",
            "turnover_refill",
        },
        "checkpoint_restore": {
            "checkpoint_export",
            "disk_checkpoint_reload",
            "fresh_owner_restore",
            "first_post_restore_tick",
            "first_post_restore_refill",
        },
        "settlement_reporting": {"terminal_settlement", "reporting_parity"},
    }
    route_changed = (
        old.provider_id != new.provider_id
        or old.profile_id != new.profile_id
        or bool(changed)
        or set(old.required_obligations) != set(new.required_obligations)
    )
    if not route_changed:
        affected: set[str] = set()
    elif any(
        row["component_id"]
        in {
            "provider_translation",
            "runner_factory",
            "native_dispatch",
            "world_route",
            "controller",
            "lifecycle_owner",
            "source_dependency_closure",
        }
        for row in changed
    ):
        affected = set(old.required_obligations) | set(new.required_obligations)
    else:
        affected = set().union(
            *(
                obligation_dependencies.get(row["component_id"], set())
                for row in changed
            )
        )
        affected &= set(old.required_obligations) | set(new.required_obligations)
        affected.update(set(old.required_obligations) ^ set(new.required_obligations))
    return {
        "schema_version": EXECUTION_ROUTE_COMPARISON_SCHEMA_VERSION,
        "old_execution_route_sha256": old.execution_route_sha256,
        "new_execution_route_sha256": new.execution_route_sha256,
        "reuses_route_qualification": not route_changed,
        "mechanic_qualification_reusable": True,
        "route_canary_required": route_changed,
        "changed_components": changed,
        "affected_obligations": sorted(affected),
        "invalidation_policy": (
            "none"
            if not route_changed
            else "dependent_route_only_conservative_source_fallback"
        ),
    }


def _plan_member(plan: CampaignPlanV1 | Mapping[str, Any], name: str) -> Any:
    if isinstance(plan, Mapping):
        aliases = {
            "profile_id": "resolved_profile",
            "batches": "physical_batches",
        }
        public_name = aliases.get(name, name)
        if public_name in plan:
            return plan.get(public_name)
        execution = plan.get("execution")
        if isinstance(execution, Mapping) and name in {
            "generations",
            "checkpoints",
        }:
            return execution.get(name)
        return None
    return getattr(plan, name)


def _key(value: CampaignPlanV1 | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, CampaignPlanV1):
        raw: Any = value.qualification_key
    elif "qualification_key" in value:
        raw = value.get("qualification_key")
    else:
        raw = value
    if not isinstance(raw, Mapping) or not all(isinstance(name, str) for name in raw):
        raise QualificationReceiptError("qualification key must be an object")
    return dict(raw)


def _mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(
        isinstance(name, str) for name in value
    ):
        raise QualificationReceiptError(f"{path} must be an object")
    return dict(value)


def _sequence(value: Any, *, path: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise QualificationReceiptError(f"{path} must be an array")
    return list(value)


def qualification_binding_v1(
    value: CampaignPlanV1 | Mapping[str, Any],
) -> dict[str, Any]:
    """Return the auditable ownership categories bound by a qualification key."""

    key = _key(value)
    artifacts = _mapping(key.get("artifact_hashes"), path="artifact_hashes")
    fixed = _mapping(key.get("fixed_invariants"), path="fixed_invariants")
    state = _sequence(key.get("state_components"), path="state_components")
    checkpoints = _sequence(
        key.get("checkpoint_components"), path="checkpoint_components"
    )
    admitted = _sequence(key.get("admitted_batch_range"), path="admitted_batch_range")
    if len(admitted) != 2 or any(type(item) is not int for item in admitted):
        raise QualificationReceiptError(
            "admitted_batch_range must contain two integer bounds"
        )

    rng = {name: fixed[name] for name in sorted(fixed) if "rng" in name}
    checkpoint_fixed = {
        name: fixed[name] for name in sorted(fixed) if "checkpoint" in name
    }
    capacity = {
        name: fixed[name]
        for name in sorted(fixed)
        if name == "capacity" or name.endswith(".capacity")
    }
    mechanics = {
        name: fixed[name]
        for name in sorted(fixed)
        if name not in rng and name not in checkpoint_fixed and name not in capacity
    }
    return {
        "profile": {
            "profile_id": key.get("profile_id"),
            "profile_version": key.get("profile_version"),
        },
        "mechanics": {
            "backend": key.get("backend"),
            "fixed_invariants": mechanics,
        },
        "artifacts": dict(sorted(artifacts.items())),
        "state": {"components": state},
        "observation": {"contract": key.get("observation_contract")},
        "rng": {"fixed_invariants": rng},
        "checkpoint": {
            "components": checkpoints,
            "fixed_invariants": checkpoint_fixed,
        },
        "capacity": {"fixed_invariants": capacity},
        "admitted_batch_range": admitted,
    }


def _comparison_fields(key: Mapping[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name in (
        "schema_version",
        "profile_id",
        "profile_version",
        "backend",
        "observation_contract",
        "state_components",
        "checkpoint_components",
        "admitted_batch_range",
    ):
        if name in key:
            fields[name] = key[name]
    for parent in ("artifact_hashes", "fixed_invariants"):
        raw = key.get(parent)
        if isinstance(raw, Mapping):
            for name in sorted(raw):
                fields[f"{parent}.{name}"] = raw[name]
        elif parent in key:
            fields[parent] = raw
    for name in sorted(
        set(key)
        - {
            "schema_version",
            "profile_id",
            "profile_version",
            "backend",
            "observation_contract",
            "state_components",
            "checkpoint_components",
            "admitted_batch_range",
            "artifact_hashes",
            "fixed_invariants",
        }
    ):
        fields[name] = key[name]
    return fields


def _category(path: str) -> str:
    if path.startswith("artifact_hashes."):
        return "artifact"
    if path.startswith("fixed_invariants."):
        leaf = path.removeprefix("fixed_invariants.")
        if "rng" in leaf:
            return "rng"
        if "checkpoint" in leaf:
            return "checkpoint"
        if leaf == "capacity" or leaf.endswith(".capacity"):
            return "capacity"
        return "mechanic"
    return {
        "profile_id": "profile",
        "profile_version": "profile",
        "backend": "mechanic",
        "observation_contract": "observation",
        "state_components": "state",
        "checkpoint_components": "checkpoint",
        "admitted_batch_range": "capacity",
        "schema_version": "schema",
    }.get(path, "qualification_key")


def compare_qualification_keys_v1(
    old: CampaignPlanV1 | Mapping[str, Any],
    new: CampaignPlanV1 | Mapping[str, Any],
) -> dict[str, Any]:
    """Explain exactly why ``new`` reuses or invalidates ``old`` qualification."""

    old_key = _key(old)
    new_key = _key(new)
    old_fields = _comparison_fields(old_key)
    new_fields = _comparison_fields(new_key)
    changes: list[dict[str, Any]] = []
    for path in sorted(set(old_fields) | set(new_fields)):
        old_present = path in old_fields
        new_present = path in new_fields
        old_value = old_fields.get(path)
        new_value = new_fields.get(path)
        if old_present == new_present and old_value == new_value:
            continue
        changes.append(
            {
                "invariant": _category(path),
                "path": path,
                "old_present": old_present,
                "old_value": old_value,
                "new_present": new_present,
                "new_value": new_value,
            }
        )
    return {
        "schema_version": QUALIFICATION_COMPARISON_SCHEMA_VERSION,
        "reuses_qualification": not changes,
        "old_qualification_key_sha256": canonical_sha256(old_key),
        "new_qualification_key_sha256": canonical_sha256(new_key),
        "changed_invariants": changes,
        "explanation": (
            "qualification is reusable because every fixed invariant is unchanged"
            if not changes
            else f"qualification is invalidated by {len(changes)} changed fixed invariant(s)"
        ),
    }


def _batch_sizes(plan: CampaignPlanV1 | Mapping[str, Any]) -> list[int]:
    batches = _plan_member(plan, "batches")
    if isinstance(batches, (str, bytes)) or not isinstance(batches, Sequence):
        return []
    sizes: list[int] = []
    for batch in batches:
        cell_ids = (
            batch.get("cell_ids")
            if isinstance(batch, Mapping)
            else getattr(batch, "cell_ids", None)
        )
        if isinstance(cell_ids, Sequence) and not isinstance(cell_ids, (str, bytes)):
            sizes.append(len(cell_ids))
    return sizes


def _cells(plan: CampaignPlanV1 | Mapping[str, Any]) -> list[Any]:
    raw = _plan_member(plan, "cells")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return list(raw)
    return []


def _qualification_neutral_treatments(
    plan: CampaignPlanV1 | Mapping[str, Any], fixed: Mapping[str, Any]
) -> list[str]:
    fixed_parameters = {
        name.removeprefix("parameter.")
        for name in fixed
        if name.startswith("parameter.")
    }
    values: dict[str, set[tuple[str, str]]] = {}
    for cell in _cells(plan):
        raw = (
            cell.get("parameters")
            if isinstance(cell, Mapping)
            else getattr(cell, "parameters", None)
        )
        if not isinstance(raw, Mapping):
            continue
        for name, value in raw.items():
            values.setdefault(str(name), set()).add((type(value).__name__, repr(value)))
    return sorted(
        name
        for name, distinct in values.items()
        if len(distinct) > 1 and name not in fixed_parameters
    )


def qualification_reuse_receipt_v1(
    plan: CampaignPlanV1 | Mapping[str, Any],
) -> dict[str, Any]:
    """Describe why campaign-level variation can reuse the bound qualification."""

    key = _key(plan)
    binding = qualification_binding_v1(key)
    fixed = _mapping(key.get("fixed_invariants"), path="fixed_invariants")
    admitted = binding["admitted_batch_range"]
    sizes = _batch_sizes(plan)
    estimate = _plan_member(plan, "resource_estimate")
    requested = (
        estimate.get("physical_batch_cells") if isinstance(estimate, Mapping) else None
    )
    if type(requested) is not int:
        requested = max(sizes, default=None)
    within_range = type(requested) is int and admitted[0] <= requested <= admitted[1]
    computed_hash = canonical_sha256(key)
    declared_hash = _plan_member(plan, "qualification_key_sha256")
    hash_exact = isinstance(declared_hash, str) and declared_hash == computed_hash
    cells = _cells(plan)
    checkpoints = _plan_member(plan, "checkpoints")
    generations = _plan_member(plan, "generations")
    neutral_treatments = _qualification_neutral_treatments(plan, fixed)
    return {
        "schema_version": QUALIFICATION_REUSE_SCHEMA_VERSION,
        "qualification_key_sha256": computed_hash,
        "declared_qualification_key_sha256": declared_hash,
        "qualification_key_hash_exact": hash_exact,
        "reuses_bound_qualification": hash_exact and within_range,
        "identity_binding": binding,
        "campaign_values_outside_identity": {
            "generations": generations,
            "seed_assignments": sum(
                len(
                    cell.get("seeds", {})
                    if isinstance(cell, Mapping)
                    else getattr(cell, "seeds", {})
                )
                for cell in cells
            ),
            "checkpoint_milestones": list(checkpoints or ()),
            "logical_cell_count": len(cells),
            "requested_physical_batch_cells": requested,
            "realized_physical_batch_sizes": sizes,
            "qualification_neutral_treatments": neutral_treatments,
        },
        "batch_admission": {
            "admitted_range": admitted,
            "requested_physical_batch_cells": requested,
            "within_admitted_range": within_range,
            "ordinary_physical_batch_extent_is_identity": False,
        },
        "explanation": (
            "G, seed assignments, checkpoint milestones, logical cell count, and "
            "qualification-neutral treatments reuse this qualification; requested B "
            "is not identity and reuses it only while inside the qualified physical "
            "batch range, which may be narrower than planning capability."
        ),
    }


__all__ = [
    "ACS_FOCUSED_ROUTE_OBLIGATIONS_V1",
    "COMPACT_SCIENCE_FOCUSED_ROUTE_OBLIGATIONS_V1",
    "EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION",
    "EXECUTION_ROUTE_COMPARISON_SCHEMA_VERSION",
    "EXECUTION_ROUTE_OBLIGATIONS_V1",
    "EXECUTION_ROUTE_QUALIFICATION_SCHEMA_VERSION",
    "QUALIFICATION_COMPARISON_SCHEMA_VERSION",
    "QUALIFICATION_REUSE_SCHEMA_VERSION",
    "QualificationReceiptError",
    "acs_checkpoint_staging_source_sha256_v1",
    "compare_qualification_keys_v1",
    "compare_execution_routes_v1",
    "execution_route_qualification_sidecar_path_v1",
    "execution_route_science_qualification_projection_path_v1",
    "execution_route_qualification_status_v1",
    "qualification_binding_v1",
    "qualification_reuse_receipt_v1",
    "read_execution_route_qualification_sidecar_v1",
    "validate_execution_route_acs_focused_evidence_v1",
]
