"""Strict projection of candidate-segment evidence into the mechanic-gate ABI.

The candidate segment owner records most, but not all, observations required by
the direct mechanic gate.  This module deliberately does not synthesize the
missing observations.  The execution owner must bind them in the typed
``direct_projection_evidence`` block before sealing its candidate result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import json
from typing import Any, Final

from engine.perf.cuda_architecture_b_resident_reproduction_exposure_sidecar_v1 import (
    CONTINUITY_SCHEMA_VERSION,
    SEGMENT_OUTPUT_SCHEMA_VERSION,
    STORAGE_MODE_EXACT_FULL,
)

from .canonical import canonical_sha256
from .inheritance_cross_path_candidate_segment_owner_v1 import (
    RESULT_SCHEMA_VERSION,
    SEGMENT_TRACE_SCHEMA_VERSION,
)
from .inheritance_cross_path_native_qualification_observer_v1 import (
    DIRECT_PROJECTION_CAPTURE_SCHEMA_VERSION,
    DIRECT_PROJECTION_CELL_CAPTURE_SCHEMA_VERSION,
    DIRECT_PROJECTION_PATH_INSTRUMENTATION_SCHEMA_VERSION,
)
from .inheritance_cross_path_mechanic_gate_adapter_v1 import (
    DIRECT_ATTEMPT_SCHEMA_VERSION,
    DIRECT_CELL_SCHEMA_VERSION,
    DIRECT_COMMIT_SCHEMA_VERSION,
    DIRECT_PATH_SCHEMA_VERSION,
    DIRECT_RESULT_SCHEMA_VERSION,
    bind_direct_mechanic_segment_result_v1,
)
from .mixed_mode_resident_isolation_runner import (
    ResidentIsolationExecutionRequestV1,
)


EVIDENCE_SCHEMA_VERSION: Final[str] = DIRECT_PROJECTION_CAPTURE_SCHEMA_VERSION
CELL_EVIDENCE_SCHEMA_VERSION: Final[str] = (
    DIRECT_PROJECTION_CELL_CAPTURE_SCHEMA_VERSION
)

_HEX = frozenset("0123456789abcdef")
_OBSERVATION_FIELDS = frozenset(
    {
        "authoritative_attempt_rows",
        "successful_commit_rows",
        "failed_attempt_rows",
        "eligibility_snapshot_rows",
        "donor_draw_rows",
        "sidecar_attempt_rows",
    }
)
class InheritanceCrossPathCandidateDirectProjectionError(ValueError):
    """The candidate result cannot be projected without inventing evidence."""


def _fail(message: str) -> InheritanceCrossPathCandidateDirectProjectionError:
    return InheritanceCrossPathCandidateDirectProjectionError(message)


def _object(
    value: Any, *, name: str, fields: frozenset[str]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(f"{name} must be an object")
    actual = frozenset(value)
    if actual != fields:
        raise _fail(
            f"{name} field set is invalid; missing={sorted(fields - actual)}, "
            f"unexpected={sorted(actual - fields)}"
        )
    return dict(value)


def _sequence(value: Any, *, name: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _fail(f"{name} must be an array")
    return list(value)


def _text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(f"{name} must be non-empty text")
    return value


def _digest(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise _fail(f"{name} must be a lowercase SHA-256 digest")
    return value


def _integer(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise _fail(f"{name} must be a non-negative integer")
    return value


def _json_copy(value: Any, *, name: str) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise _fail(f"{name} must contain only canonical JSON data") from exc


def _string_map(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) or not key for key in value
    ):
        raise _fail(f"{name} must be an object with non-empty text keys")
    return dict(value)


@dataclass(frozen=True, slots=True)
class CandidateDirectProjectionCellEvidenceV1:
    """Truthful observer capture for one resident cell."""

    occurrence_id: str
    reproduction_path: str
    native_path_code: int
    founder_source_authority_sha256: str
    pre_dispatch_state_sha256: str
    cell_authority_sha256: str
    path_observation: Mapping[str, int]
    path_instrumentation: Mapping[str, Mapping[str, int | str]]

    @classmethod
    def from_payload(
        cls, value: Any, *, path_ids: tuple[str, ...]
    ) -> CandidateDirectProjectionCellEvidenceV1:
        row = _object(
            value,
            name="direct projection cell evidence",
            fields=frozenset(
                {
                    "schema_version",
                    "occurrence_id",
                    "reproduction_path",
                    "native_path_code",
                    "pre_dispatch_state_sha256",
                    "founder_source_authority_sha256",
                    "cell_authority_sha256",
                    "path_observation",
                    "path_instrumentation",
                }
            ),
        )
        if row["schema_version"] != CELL_EVIDENCE_SCHEMA_VERSION:
            raise _fail("direct projection cell evidence schema mismatch")
        reproduction_path = _text(
            row["reproduction_path"], name="reproduction_path"
        )
        if reproduction_path not in path_ids:
            raise _fail("cell evidence references an unregistered path")
        observation = _object(
            row["path_observation"],
            name="path_observation",
            fields=_OBSERVATION_FIELDS,
        )
        normalized_observation = {
            name: _integer(value, name=f"path_observation.{name}")
            for name, value in observation.items()
        }
        if (
            normalized_observation["successful_commit_rows"]
            + normalized_observation["failed_attempt_rows"]
            != normalized_observation["authoritative_attempt_rows"]
            or normalized_observation["sidecar_attempt_rows"]
            != normalized_observation["authoritative_attempt_rows"]
        ):
            raise _fail("direct path observation counts do not reconcile")
        instrumentation = _string_map(
            row["path_instrumentation"], name="path_instrumentation"
        )
        if tuple(instrumentation) != path_ids:
            raise _fail("path instrumentation order differs from registered paths")
        normalized_instrumentation: dict[str, dict[str, int | str]] = {}
        seen_codes: set[int] = set()
        for path_id in path_ids:
            path_row = _object(
                instrumentation[path_id],
                name=f"path_instrumentation.{path_id}",
                fields=frozenset(
                    {
                        "schema_version",
                        "path_id",
                        "native_path_code",
                        "rng_counter_state",
                        "counter_write_count",
                        "event_row_write_count",
                        "rng_state_write_count",
                    }
                ),
            )
            if (
                path_row["schema_version"]
                != DIRECT_PROJECTION_PATH_INSTRUMENTATION_SCHEMA_VERSION
                or path_row["path_id"] != path_id
            ):
                raise _fail("path instrumentation identity is invalid")
            code = _integer(
                path_row["native_path_code"],
                name=f"path_instrumentation.{path_id}.native_path_code",
            )
            if code in seen_codes:
                raise _fail("path instrumentation native codes must be unique")
            seen_codes.add(code)
            normalized_instrumentation[path_id] = {
                "schema_version": path_row["schema_version"],
                "path_id": path_id,
                "native_path_code": code,
                "rng_counter_state": _integer(
                    path_row["rng_counter_state"],
                    name=f"path_instrumentation.{path_id}.rng_counter_state",
                ),
                "counter_write_count": _integer(
                    path_row["counter_write_count"],
                    name=f"path_instrumentation.{path_id}.counter_write_count",
                ),
                "event_row_write_count": _integer(
                    path_row["event_row_write_count"],
                    name=f"path_instrumentation.{path_id}.event_row_write_count",
                ),
                "rng_state_write_count": _integer(
                    path_row["rng_state_write_count"],
                    name=f"path_instrumentation.{path_id}.rng_state_write_count",
                ),
            }
        return cls(
            occurrence_id=_text(row["occurrence_id"], name="occurrence_id"),
            reproduction_path=reproduction_path,
            native_path_code=_integer(
                row["native_path_code"], name="native_path_code"
            ),
            founder_source_authority_sha256=_digest(
                row["founder_source_authority_sha256"],
                name="founder_source_authority_sha256",
            ),
            pre_dispatch_state_sha256=_digest(
                row["pre_dispatch_state_sha256"],
                name="pre_dispatch_state_sha256",
            ),
            cell_authority_sha256=_digest(
                row["cell_authority_sha256"], name="cell_authority_sha256"
            ),
            path_observation=normalized_observation,
            path_instrumentation=normalized_instrumentation,
        )


@dataclass(frozen=True, slots=True)
class InheritanceCrossPathCandidateDirectProjectionEvidenceV1:
    """Typed explicit owner evidence needed by the direct mechanic schema."""

    captured_at_utc: str
    path_code_by_id: Mapping[str, int]
    failure_code_by_value: Mapping[str, str]
    cells: tuple[CandidateDirectProjectionCellEvidenceV1, ...]
    instrumentation_owner: str

    @classmethod
    def from_payload(
        cls, value: Any, *, path_ids: tuple[str, ...]
    ) -> InheritanceCrossPathCandidateDirectProjectionEvidenceV1:
        row = _object(
            value,
            name="direct_projection_evidence",
            fields=frozenset(
                {
                    "schema_version",
                    "captured_at_utc",
                    "path_code_by_id",
                    "failure_code_by_value",
                    "cells",
                    "instrumentation_owner",
                }
            ),
        )
        if row["schema_version"] != EVIDENCE_SCHEMA_VERSION:
            raise _fail("direct_projection_evidence schema mismatch")
        captured_at = _text(row["captured_at_utc"], name="captured_at_utc")
        try:
            parsed_captured_at = datetime.fromisoformat(
                captured_at.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise _fail("captured_at_utc must be ISO-8601") from exc
        if parsed_captured_at.tzinfo is None:
            raise _fail("captured_at_utc must include a timezone")
        paths = _string_map(row["path_code_by_id"], name="path_code_by_id")
        if tuple(paths) != path_ids:
            raise _fail(
                "path_code_by_id order differs from registered path order"
            )
        path_codes = {
            path_id: _integer(paths[path_id], name=f"path_code_by_id.{path_id}")
            for path_id in path_ids
        }
        if len(set(path_codes.values())) != len(path_codes):
            raise _fail("path codes must be unique")
        failure_values = _string_map(
            row["failure_code_by_value"], name="failure_code_by_value"
        )
        failures = {
            _text(key, name="failure code value"): _text(
                label, name=f"failure_code_by_value.{key}"
            )
            for key, label in failure_values.items()
        }
        cells = tuple(
            CandidateDirectProjectionCellEvidenceV1.from_payload(
                item, path_ids=path_ids
            )
            for item in _sequence(row["cells"], name="evidence cells")
        )
        for cell in cells:
            if {
                path_id: int(cell.path_instrumentation[path_id]["native_path_code"])
                for path_id in path_ids
            } != path_codes:
                raise _fail(
                    "cell path instrumentation differs from registered path codes"
                )
        return cls(
            captured_at_utc=captured_at,
            path_code_by_id=path_codes,
            failure_code_by_value=failures,
            cells=cells,
            instrumentation_owner=_text(
                row["instrumentation_owner"], name="instrumentation_owner"
            ),
        )


def _validated_candidate_result(
    candidate_result: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(candidate_result, Mapping):
        raise _fail("candidate result must be an object")
    if "direct_projection_evidence" not in candidate_result:
        raise _fail(
            "candidate result missing upstream field: direct_projection_evidence"
        )
    expected = frozenset(
        {
            "schema_version",
            "request_sha256",
            "candidate_authority_sha256",
            "candidate_artifact_sha256s",
            "parity_dimension_sha256s",
            "plan_sha256",
            "batch_id",
            "cell_ids",
            "reporting_enabled",
            "production_authority",
            "source_owner_id",
            "terminal_owner_id",
            "fresh_owner",
            "restore_receipts",
            "segments",
            "terminal_checkpoint",
            "metrics_rows",
            "provider_result",
            "qualification_receipts",
            "direct_projection_evidence",
            "result_sha256",
        }
    )
    row = _object(candidate_result, name="candidate result", fields=expected)
    supplied = _digest(row.pop("result_sha256"), name="result_sha256")
    if canonical_sha256(row) != supplied:
        raise _fail("candidate result digest mismatch")
    if row["schema_version"] != RESULT_SCHEMA_VERSION:
        raise _fail("candidate result schema mismatch")
    if row["production_authority"] is not False:
        raise _fail("candidate result cannot grant production authority")
    return row


def _validate_segment_cell(
    artifact: Mapping[str, Any],
    *,
    segment_index: int,
    cell_id: str,
    expected_previous_chain: str | None,
) -> tuple[dict[str, Any], str]:
    row = _object(
        artifact,
        name="segment artifact",
        fields=frozenset(
            {
                "schema_version",
                "segment_index",
                "capacity_semantics",
                "cell_id",
                "cell",
                "segment_output_sha256",
            }
        ),
    )
    if (
        row["schema_version"] != SEGMENT_OUTPUT_SCHEMA_VERSION
        or row["segment_index"] != segment_index
        or row["cell_id"] != cell_id
    ):
        raise _fail("segment artifact differs from its trace placement")
    capacity = row["capacity_semantics"]
    if (
        not isinstance(capacity, Mapping)
        or capacity.get("storage_mode") != STORAGE_MODE_EXACT_FULL
    ):
        raise _fail("direct projection requires exact_full exposure artifacts")
    cell = _string_map(row["cell"], name="segment cell")
    for field in (
        "segment_sha256",
        "previous_chain_sha256",
        "chain_sha256_after_segment",
    ):
        _digest(cell.get(field), name=f"segment cell.{field}")
    unsigned_cell = dict(cell)
    segment_sha256 = unsigned_cell.pop("segment_sha256")
    previous_chain = unsigned_cell.pop("previous_chain_sha256")
    chain_after = unsigned_cell.pop("chain_sha256_after_segment")
    expected_segment = canonical_sha256(
        {
            "schema_version": SEGMENT_OUTPUT_SCHEMA_VERSION,
            "segment_index": segment_index,
            "cell": unsigned_cell,
        }
    )
    if segment_sha256 != expected_segment:
        raise _fail("segment cell digest mismatch")
    if expected_previous_chain is not None and previous_chain != expected_previous_chain:
        raise _fail("segment cell chain is discontinuous")
    if chain_after != canonical_sha256(
        {
            "schema_version": CONTINUITY_SCHEMA_VERSION,
            "segment_index": segment_index,
            "previous_chain_sha256": previous_chain,
            "segment_sha256": segment_sha256,
        }
    ):
        raise _fail("segment cell chain digest mismatch")
    return cell, chain_after


def _validate_segment_output(
    artifacts: list[dict[str, Any]], *, segment_index: int
) -> None:
    first = artifacts[0]
    if any(
        artifact["segment_output_sha256"] != first["segment_output_sha256"]
        or artifact["capacity_semantics"] != first["capacity_semantics"]
        for artifact in artifacts
    ):
        raise _fail("per-cell artifacts disagree on their segment output")
    cells = [artifact["cell"] for artifact in artifacts]
    cumulative = [cell["cumulative_counts_after_segment"] for cell in cells]
    continuity = {
        "schema_version": CONTINUITY_SCHEMA_VERSION,
        "segment_count": segment_index + 1,
        "cell_chain_sha256": [
            cell["chain_sha256_after_segment"] for cell in cells
        ],
        "cumulative_attempt_counts": [row["attempts"] for row in cumulative],
        "cumulative_success_counts": [row["successes"] for row in cumulative],
        "cumulative_failure_counts": [row["failures"] for row in cumulative],
        "cumulative_snapshot_counts": [
            row["eligible_snapshots"] for row in cumulative
        ],
    }
    core = {
        "schema_version": SEGMENT_OUTPUT_SCHEMA_VERSION,
        "segment_index": segment_index,
        "capacity_semantics": first["capacity_semantics"],
        "cells": cells,
        "continuity_after_segment": continuity,
    }
    if canonical_sha256(core) != first["segment_output_sha256"]:
        raise _fail("segment output digest mismatch")


def _direct_attempts(
    cell: Mapping[str, Any],
    *,
    path_id: str,
    path_code: int,
    failure_codes: Mapping[str, str],
    segment_index: int,
) -> list[dict[str, Any]]:
    snapshots = {
        snapshot["snapshot_index"]: snapshot
        for snapshot in _sequence(
            cell.get("eligible_parent_snapshots"),
            name="eligible_parent_snapshots",
        )
        if isinstance(snapshot, Mapping)
    }
    if len(snapshots) != cell.get("snapshot_count"):
        raise _fail("eligible snapshot count or index is invalid")
    attempts = _sequence(cell.get("attempts"), name="attempts")
    if len(attempts) != cell.get("attempt_count"):
        raise _fail("attempt_count differs from exact exposure rows")
    projected: list[dict[str, Any]] = []
    for raw in attempts:
        attempt = _string_map(raw, name="attempt")
        if attempt.get("path") != path_code:
            raise _fail("attempt path differs from registered cell path")
        refs = _sequence(
            attempt.get("eligible_snapshot_refs"), name="eligible_snapshot_refs"
        )
        counts = _sequence(
            attempt.get("eligible_parent_counts"), name="eligible_parent_counts"
        )
        if len(refs) != 2 or len(counts) != 2:
            raise _fail("attempt must reference primary and mate snapshots")
        try:
            primary, mate = (snapshots[refs[0]], snapshots[refs[1]])
        except (KeyError, TypeError) as exc:
            raise _fail("attempt references an absent eligibility snapshot") from exc
        if any(
            snapshot.get("path") != path_code
            or snapshot.get("generation") != attempt.get("generation")
            or snapshot.get("tick") != attempt.get("tick")
            for snapshot in (primary, mate)
        ):
            raise _fail("attempt and eligibility snapshot coordinates differ")
        parent_sets = [
            _sequence(
                primary.get("eligible_parent_ids"),
                name="primary eligible_parent_ids",
            ),
            _sequence(
                mate.get("eligible_parent_ids"), name="mate eligible_parent_ids"
            ),
        ]
        if counts != [len(values) for values in parent_sets]:
            raise _fail("attempt eligible counts differ from exact snapshots")
        selected = _sequence(
            attempt.get("selected_parent_ids"), name="selected_parent_ids"
        )
        if len(selected) != 2:
            raise _fail("attempt must expose exactly two selected parents")
        success = attempt.get("success")
        if type(success) is not bool:
            raise _fail("attempt success must be boolean")
        failure_value = attempt.get("failure_code")
        if success:
            failure_code = None
        else:
            key = str(failure_value)
            if key not in failure_codes:
                raise _fail(
                    f"missing upstream failure-code binding for value: {key}"
                )
            failure_code = failure_codes[key]
        commit = attempt.get("authoritative_commit")
        direct_commit = None
        if success:
            commit = _string_map(commit, name="authoritative_commit")
            if (
                commit.get("generation") != attempt.get("generation")
                or commit.get("tick") != attempt.get("tick")
                or commit.get("path_id") != path_code
                or commit.get("selected_parent_ids") != selected
            ):
                raise _fail("authoritative commit differs from its exposure row")
            source_commit = {
                "generation": commit.get("generation"),
                "tick": commit.get("tick"),
                "path_id": commit.get("path_id"),
                "selected_parent_ids": commit.get("selected_parent_ids"),
                "child_agent_id": commit.get("child_agent_id"),
                "donor_draw_ordinals": commit.get("donor_draw_ordinals"),
            }
            if commit.get("commit_authority_sha256") != canonical_sha256(
                source_commit
            ):
                raise _fail("authoritative commit source digest mismatch")
            commit_core = {
                "schema_version": DIRECT_COMMIT_SCHEMA_VERSION,
                "generation": _integer(
                    commit["generation"], name="commit generation"
                ),
                "tick": _integer(commit["tick"], name="commit tick"),
                "path_id": path_id,
                "selected_parent_ids": [str(value) for value in selected],
                "child_agent_id": str(commit["child_agent_id"]),
                "donor_draw_ordinals": [
                    _integer(value, name="donor draw ordinal")
                    for value in _sequence(
                        commit.get("donor_draw_ordinals"),
                        name="donor_draw_ordinals",
                    )
                ],
            }
            direct_commit = {
                **commit_core,
                "commit_authority_sha256": canonical_sha256(commit_core),
            }
        elif commit is not None:
            raise _fail("failed attempt carries an authoritative commit")
        transaction = {
            "segment_index": segment_index,
            "attempt": attempt,
            "primary_snapshot": primary,
            "mate_snapshot": mate,
        }
        projected.append(
            {
                "schema_version": DIRECT_ATTEMPT_SCHEMA_VERSION,
                "generation": _integer(
                    attempt.get("generation"), name="attempt generation"
                ),
                "tick": _integer(attempt.get("tick"), name="attempt tick"),
                "primary_eligible_parent_ids": [
                    str(value) for value in parent_sets[0]
                ],
                "mate_eligible_parent_ids": [
                    str(value) for value in parent_sets[1]
                ],
                "selected_parent_ids": [str(value) for value in selected],
                "success": success,
                "contributing_parent_ids": (
                    [str(value) for value in selected] if success else []
                ),
                "failure_code": failure_code,
                "authoritative_commit": direct_commit,
                "selection_transaction_sha256": canonical_sha256(transaction),
            }
        )
    return projected


def _checkpoint_rows(
    checkpoint: Any, *, cell_ids: tuple[str, ...]
) -> tuple[dict[str, Any], ...]:
    checkpoint_row = _string_map(checkpoint, name="terminal_checkpoint")
    rows = _sequence(checkpoint_row.get("cell_bundles"), name="cell_bundles")
    by_id = {
        row.get("cell_id"): dict(row)
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("cell_id"), str)
    }
    if tuple(by_id) != cell_ids or len(rows) != len(cell_ids):
        raise _fail("terminal checkpoint cell order differs")
    terminal = _string_map(
        checkpoint_row.get("terminal_cell_digests"),
        name="terminal_cell_digests",
    )
    result = []
    for cell_id in cell_ids:
        row = by_id[cell_id]
        digest = _digest(
            row.get("source_projection_sha256"),
            name="source_projection_sha256",
        )
        if terminal.get(cell_id) != digest:
            raise _fail("terminal checkpoint digest differs from its bundle")
        result.append(row)
    return tuple(result)


def project_candidate_segment_result_to_direct_mechanic_v1(
    candidate_result: Mapping[str, Any],
    *,
    runner_request: ResidentIsolationExecutionRequestV1,
    callback_authority_sha256: str,
    path_ids: Sequence[str],
) -> dict[str, Any]:
    """Reject projection of new retired-route candidate evidence."""

    raise _fail(
        "P12_RETIRED_INHERITANCE_AUTHORITY "
        "candidate direct-mechanic projection is disabled"
    )
    # Historical implementation retained temporarily for evidence reading.
    if not isinstance(runner_request, ResidentIsolationExecutionRequestV1):
        raise _fail("runner_request must be ResidentIsolationExecutionRequestV1")
    callback_digest = _digest(
        callback_authority_sha256, name="callback_authority_sha256"
    )
    paths = tuple(_text(value, name="path_id") for value in path_ids)
    if len(paths) < 2 or len(set(paths)) != len(paths):
        raise _fail("path_ids must contain at least two unique paths")
    row = _validated_candidate_result(candidate_result)
    evidence = InheritanceCrossPathCandidateDirectProjectionEvidenceV1.from_payload(
        row["direct_projection_evidence"], path_ids=paths
    )
    registered = tuple(
        sorted(runner_request.run.cells, key=lambda cell: cell.batch_position)
    )
    cell_ids = tuple(cell.occurrence_id for cell in registered)
    if (
        tuple(row["cell_ids"]) != cell_ids
        or row["candidate_authority_sha256"]
        != runner_request.run.execution_payload["candidate_authority_sha256"]
        or row["reporting_enabled"]
        is not runner_request.run.execution_payload["reporting_enabled"]
        or row["fresh_owner"] is not runner_request.run.fresh_owner
    ):
        raise _fail("candidate result differs from the registered run")
    if tuple(item.occurrence_id for item in evidence.cells) != cell_ids:
        raise _fail("direct projection evidence cell order differs")

    qualification = _string_map(
        row["qualification_receipts"], name="qualification_receipts"
    )
    fixed_control = qualification.get("fixed_action_control")
    fixed_observation = qualification.get("fixed_action_observation")
    if fixed_control is None or fixed_observation is None:
        raise _fail("candidate result lacks fixed-action qualification receipts")
    dispatches = _sequence(
        qualification.get("resident_dispatch_observations"),
        name="resident_dispatch_observations",
    )
    if not dispatches:
        raise _fail("candidate result lacks resident state/RNG observations")
    final_dispatch = _string_map(dispatches[-1], name="final resident dispatch")
    state_digests = _sequence(
        final_dispatch.get("cell_state_sha256"), name="cell_state_sha256"
    )
    rng_digests = _sequence(
        final_dispatch.get("cell_rng_sha256"), name="cell_rng_sha256"
    )
    if len(state_digests) != len(cell_ids) or len(rng_digests) != len(cell_ids):
        raise _fail("final resident dispatch cell count differs")

    restore_rows = _sequence(row["restore_receipts"], name="restore_receipts")
    restore_by_id = {
        item.get("cell_id"): dict(item)
        for item in restore_rows
        if isinstance(item, Mapping)
    }
    if tuple(restore_by_id) != cell_ids:
        raise _fail("restore receipt order differs")
    checkpoint_rows = _checkpoint_rows(
        row["terminal_checkpoint"], cell_ids=cell_ids
    )
    metrics_rows = _sequence(row["metrics_rows"], name="metrics_rows")
    overflow = runner_request.run.execution_payload["overflow_fault"]
    overflow_target = (
        overflow.get("target_occurrence_id")
        if isinstance(overflow, Mapping)
        else None
    )
    reporting_cell_ids = {
        item.get("cell_id")
        for item in metrics_rows
        if isinstance(item, Mapping) and isinstance(item.get("cell_id"), str)
    }
    if row["reporting_enabled"]:
        expected_reporting_cell_ids = set(cell_ids)
        if isinstance(overflow_target, str):
            expected_reporting_cell_ids.discard(overflow_target)
        if reporting_cell_ids != expected_reporting_cell_ids:
            raise _fail("reporting-enabled candidate lacks per-cell rows")
    elif reporting_cell_ids:
        raise _fail("reporting-disabled candidate emitted per-cell rows")

    aggregate_attempts: dict[str, list[dict[str, Any]]] = {
        cell_id: [] for cell_id in cell_ids
    }
    aggregate_snapshot_rows: dict[str, int] = {
        cell_id: 0 for cell_id in cell_ids
    }
    final_terminal: dict[str, str] = {}
    final_lifecycle: dict[str, str] = {}
    final_checkpoint_storage: dict[str, str] = {}
    final_path_instrumentation: dict[str, dict[int, dict[str, int]]] = {}
    prior_chains: dict[str, str | None] = {cell_id: None for cell_id in cell_ids}
    segments = _sequence(row["segments"], name="segments")
    if not segments:
        raise _fail("candidate result has no exposure segments")
    for expected_index, trace_value in enumerate(segments):
        trace = _object(
            trace_value,
            name="segment trace",
            fields=frozenset(
                {
                    "schema_version",
                    "owner_id",
                    "owner_segment_ordinal",
                    "segment_index",
                    "start_generation",
                    "end_generation",
                    "cell_ids",
                    "segment_artifacts",
                    "terminal_cell_digests",
                    "mechanics_checkpoint_digests",
                    "canonical_checkpoint_storage_digests",
                    "checkpoint",
                }
            ),
        )
        if (
            trace.get("schema_version") != SEGMENT_TRACE_SCHEMA_VERSION
            or trace.get("segment_index") != expected_index
        ):
            raise _fail("segment trace indices are not contiguous")
        artifacts = _sequence(
            trace.get("segment_artifacts"), name="segment_artifacts"
        )
        if len(artifacts) != len(cell_ids):
            raise _fail("segment artifact count differs from physical cells")
        validated_artifacts: list[dict[str, Any]] = []
        for cell_id, artifact in zip(cell_ids, artifacts, strict=True):
            cell, chain = _validate_segment_cell(
                artifact,
                segment_index=expected_index,
                cell_id=cell_id,
                expected_previous_chain=prior_chains[cell_id],
            )
            prior_chains[cell_id] = chain
            aggregate_snapshot_rows[cell_id] += _integer(
                cell.get("snapshot_count"), name=f"{cell_id}.snapshot_count"
            )
            path_instrumentation_rows = _sequence(
                cell.get("path_instrumentation"),
                name=f"{cell_id}.path_instrumentation",
            )
            normalized_path_instrumentation: dict[int, dict[str, int]] = {}
            for path_value in path_instrumentation_rows:
                path_row = _object(
                    path_value,
                    name=f"{cell_id}.path_instrumentation row",
                    fields=frozenset(
                        {
                            "path_code",
                            "rng_counter_state",
                            "counter_write_count",
                            "event_row_write_count",
                            "rng_state_write_count",
                        }
                    ),
                )
                path_code = _integer(
                    path_row["path_code"],
                    name=f"{cell_id}.path_instrumentation.path_code",
                )
                if path_code in normalized_path_instrumentation:
                    raise _fail("segment path instrumentation codes are duplicated")
                normalized_path_instrumentation[path_code] = {
                    name: _integer(
                        path_row[name],
                        name=f"{cell_id}.path_instrumentation.{name}",
                    )
                    for name in (
                        "rng_counter_state",
                        "counter_write_count",
                        "event_row_write_count",
                        "rng_state_write_count",
                    )
                }
            final_path_instrumentation[cell_id] = normalized_path_instrumentation
            normalized = dict(artifact)
            normalized["cell"] = cell
            validated_artifacts.append(normalized)
            path_id = registered[cell_ids.index(cell_id)].config.reproduction_path
            aggregate_attempts[cell_id].extend(
                _direct_attempts(
                    cell,
                    path_id=path_id,
                    path_code=evidence.path_code_by_id[path_id],
                    failure_codes=evidence.failure_code_by_value,
                    segment_index=expected_index,
                )
            )
        _validate_segment_output(
            validated_artifacts, segment_index=expected_index
        )
        final_terminal = _string_map(
            trace.get("terminal_cell_digests"),
            name="terminal_cell_digests",
        )
        final_lifecycle = _string_map(
            trace.get("mechanics_checkpoint_digests"),
            name="mechanics_checkpoint_digests",
        )
        final_checkpoint_storage = _string_map(
            trace.get("canonical_checkpoint_storage_digests"),
            name="canonical_checkpoint_storage_digests",
        )
        segment_checkpoint = _string_map(
            trace.get("checkpoint"), name="segment checkpoint"
        )
        if segment_checkpoint.get("generation") != trace.get("end_generation"):
            raise _fail("segment checkpoint generation differs")
        segment_checkpoint_rows = _checkpoint_rows(
            segment_checkpoint, cell_ids=cell_ids
        )
        if {
            cell_id: checkpoint_row["source_projection_sha256"]
            for cell_id, checkpoint_row in zip(
                cell_ids, segment_checkpoint_rows, strict=True
            )
        } != final_terminal:
            raise _fail("segment checkpoint projection digests differ")

    for index, (registered_cell, extra) in enumerate(
        zip(registered, evidence.cells, strict=True)
    ):
        cell_id = registered_cell.occurrence_id
        active_path = registered_cell.config.reproduction_path
        if (
            extra.reproduction_path != active_path
            or extra.native_path_code != evidence.path_code_by_id[active_path]
        ):
            raise _fail("observer path identity differs from the registered cell")
        attempts = aggregate_attempts[cell_id]
        successes = sum(bool(item["success"]) for item in attempts)
        donor_draws = sum(
            len(item["authoritative_commit"]["donor_draw_ordinals"])
            for item in attempts
            if item["authoritative_commit"] is not None
        )
        expected_observation = {
            "authoritative_attempt_rows": len(attempts),
            "successful_commit_rows": successes,
            "failed_attempt_rows": len(attempts) - successes,
            "eligibility_snapshot_rows": aggregate_snapshot_rows[cell_id],
            "donor_draw_rows": donor_draws,
            "sidecar_attempt_rows": len(attempts),
        }
        if dict(extra.path_observation) != expected_observation:
            raise _fail(
                "observer path counts differ from exact exposure artifacts"
            )
        observed_instrumentation = {
            int(extra.path_instrumentation[path_id]["native_path_code"]): {
                name: int(extra.path_instrumentation[path_id][name])
                for name in (
                    "rng_counter_state",
                    "counter_write_count",
                    "event_row_write_count",
                    "rng_state_write_count",
                )
            }
            for path_id in paths
        }
        if final_path_instrumentation.get(cell_id) != observed_instrumentation:
            raise _fail(
                "observer path instrumentation differs from checkpointed "
                "segment exposure"
            )
        _digest(
            rng_digests[index],
            name=f"{cell_id}.aggregate_resident_rng_identity",
        )

    direct_cells: list[dict[str, Any]] = []
    for index, (registered_cell, extra, checkpoint) in enumerate(
        zip(registered, evidence.cells, checkpoint_rows, strict=True)
    ):
        cell_id = registered_cell.occurrence_id
        terminal_projection = _digest(
            final_terminal.get(cell_id),
            name=f"{cell_id}.terminal_projection_sha256",
        )
        if checkpoint["source_projection_sha256"] != terminal_projection:
            raise _fail("final trace and terminal checkpoint digest differ")
        lifecycle = _digest(
            final_lifecycle.get(cell_id),
            name=f"{cell_id}.lifecycle_state_sha256",
        )
        state_identity = _digest(
            state_digests[index], name=f"{cell_id}.resident_state_sha256"
        )
        rng_identity = _digest(
            rng_digests[index], name=f"{cell_id}.resident_rng_sha256"
        )
        active_path = registered_cell.config.reproduction_path
        attempts = aggregate_attempts[cell_id]
        path_rows: list[dict[str, Any]] = []
        for path_id in paths:
            enabled = path_id == active_path
            selected_attempts = attempts if enabled else []
            successes = sum(bool(item["success"]) for item in selected_attempts)
            instrumentation = extra.path_instrumentation[path_id]
            rng_counter_state = int(instrumentation["rng_counter_state"])
            native_writes = {
                "counter": int(instrumentation["counter_write_count"]),
                "event_row": int(instrumentation["event_row_write_count"]),
                "rng_state": int(instrumentation["rng_state_write_count"]),
            }
            counters = (
                [
                    {
                        "counter_id": "native_path_rng_counter_state_v1",
                        "value": rng_counter_state,
                    }
                ]
                if enabled
                else []
            )
            writes = {
                **native_writes,
                "sidecar": len(selected_attempts),
            }
            if not enabled and (
                rng_counter_state != 0 or any(native_writes.values())
            ):
                raise _fail("disabled path evidence records RNG or writes")
            if enabled and (
                native_writes["event_row"] != len(selected_attempts)
                or rng_counter_state != successes
                or native_writes["counter"] != native_writes["rng_state"]
                or (
                    successes == 0
                    and native_writes["counter"] != 0
                )
                or (
                    successes > 0
                    and not 0 < native_writes["counter"] <= successes
                )
            ):
                raise _fail(
                    "active path instrumentation does not reconcile with "
                    "authoritative attempts and commits"
                )
            path_core = {
                "schema_version": DIRECT_PATH_SCHEMA_VERSION,
                "path_id": path_id,
                "enabled": enabled,
                "rng_counters": counters,
                "writes": writes,
                "attempts": selected_attempts,
                "authoritative_attempt_count": len(selected_attempts),
                "authoritative_success_count": successes,
                "authoritative_failure_count": len(selected_attempts) - successes,
                "sidecar_attempt_count": len(selected_attempts),
                "selection_transaction_count": len(selected_attempts),
                "eligibility_snapshot_transaction_count": len(selected_attempts),
            }
            path_rows.append(
                {
                    **path_core,
                    "path_authority_sha256": canonical_sha256(path_core),
                }
            )
        restore = restore_by_id[cell_id]
        owner_boundary = {
            "source_owner_id": row["source_owner_id"],
            "terminal_owner_id": row["terminal_owner_id"],
            "checkpoint_reloaded": restore.get("checkpoint_reloaded"),
            "checkpoint_source_sha256": restore.get(
                "checkpoint_source_sha256"
            ),
            "restored_source_sha256": restore.get("restored_source_sha256"),
        }
        if restore.get("source_owner_id") != row["source_owner_id"] or restore.get(
            "terminal_owner_id"
        ) != row["terminal_owner_id"]:
            raise _fail("restore receipt owner identity differs")
        checkpoint_sidecar = _digest(
            final_checkpoint_storage.get(cell_id),
            name=f"{cell_id}.canonical_checkpoint_storage_sha256",
        )
        fault_target = (
            isinstance(overflow, Mapping)
            and overflow.get("target_occurrence_id") == cell_id
        )
        fault_control = (
            qualification.get("capacity_fault_control") if fault_target else None
        )
        fault_observation = (
            qualification.get("capacity_fault_observation")
            if fault_target
            else None
        )
        if fault_target and (fault_control is None or fault_observation is None):
            raise _fail("overflow target lacks capacity-fault receipts")
        pre_fault_state = (
            _digest(
                fault_observation.get("mechanics_before_sha256"),
                name=f"{cell_id}.pre_fault_state_sha256",
            )
            if isinstance(fault_observation, Mapping)
            else state_identity
        )
        bound_cell_authority = (
            _digest(
                fault_control.get("target_cell_authority_sha256"),
                name=f"{cell_id}.fault_target_cell_authority_sha256",
            )
            if isinstance(fault_control, Mapping)
            else extra.cell_authority_sha256
        )
        final_state = (
            pre_fault_state
            if isinstance(fault_observation, Mapping)
            else state_identity
        )
        computed_final_authority = canonical_sha256(
            {
                "schema_version": "eiel_resident_mechanics_authority_v1",
                "final_state_sha256": final_state,
                "resident_state_sha256": state_identity,
                "resident_rng_sha256": rng_identity,
                "checkpoint_sidecar_sha256": checkpoint_sidecar,
                "lifecycle_state_sha256": lifecycle,
                "path_authority_sha256s": [
                    item["path_authority_sha256"] for item in path_rows
                ],
            }
        )
        final_authority = (
            _digest(
                fault_control.get("pre_fault_authority_sha256"),
                name=f"{cell_id}.pre_fault_authority_sha256",
            )
            if isinstance(fault_control, Mapping)
            else computed_final_authority
        )
        direct_cells.append(
            {
                "schema_version": DIRECT_CELL_SCHEMA_VERSION,
                "occurrence_id": cell_id,
                "config": registered_cell.config.payload(),
                "batch_position": registered_cell.batch_position,
                "neighbor_path_ids": list(
                    registered_cell.expected_neighbor_path_ids
                ),
                "final_state_sha256": final_state,
                "pre_fault_state_sha256": pre_fault_state,
                "checkpoint_sidecar_sha256": checkpoint_sidecar,
                "lifecycle_state_sha256": lifecycle,
                "final_authority_sha256": final_authority,
                "pre_fault_authority_sha256": (
                    final_authority
                    if isinstance(fault_control, Mapping)
                    else None
                ),
                "founder_source_authority_sha256": (
                    extra.founder_source_authority_sha256
                ),
                "ok": not fault_target,
                "fail_closed": fault_target,
                "error_code": overflow.get("error_code") if fault_target else None,
                "reporting_rows_present": cell_id in reporting_cell_ids,
                "owner_boundary": owner_boundary,
                "fixed_action_control": fixed_control,
                "fixed_action_observation": fixed_observation,
                "capacity_fault_control": fault_control,
                "capacity_fault_observation": fault_observation,
                "paths": path_rows,
                "cell_authority_sha256": bound_cell_authority,
            }
        )
    return bind_direct_mechanic_segment_result_v1(
        {
            "schema_version": DIRECT_RESULT_SCHEMA_VERSION,
            "callback_authority_sha256": callback_digest,
            "candidate_authority_sha256": row["candidate_authority_sha256"],
            "registered_run_sha256": runner_request.run.digest_sha256,
            "run_id": runner_request.run.run_id,
            "fresh_owner": runner_request.run.fresh_owner,
            "reporting_enabled": row["reporting_enabled"],
            "production_authority": False,
            "captured_at_utc": evidence.captured_at_utc,
            "cells": direct_cells,
        }
    )


__all__ = [
    "CELL_EVIDENCE_SCHEMA_VERSION",
    "CandidateDirectProjectionCellEvidenceV1",
    "EVIDENCE_SCHEMA_VERSION",
    "InheritanceCrossPathCandidateDirectProjectionError",
    "InheritanceCrossPathCandidateDirectProjectionEvidenceV1",
    "project_candidate_segment_result_to_direct_mechanic_v1",
]
