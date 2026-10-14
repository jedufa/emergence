"""Project reporting-mode parity evidence from immutable mixed-gate captures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Final

from .canonical import canonical_sha256, pretty_json
from .inheritance_cross_path_authority_publisher_v1 import (
    REPORTING_MODE_EVIDENCE_SCHEMA_VERSION,
)
from .mixed_mode_resident_isolation_runner import (
    CAPTURE_ARTIFACT_SCHEMA_VERSION,
    REGISTERED_RUN_SCHEMA_VERSION,
    RUNNER_PLAN_SCHEMA_VERSION,
    RUNNER_RESULT_SCHEMA_VERSION,
)
from .mixed_mode_resident_isolation_gate import GATE_SCHEMA_VERSION


PROJECTION_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_reporting_parity_projection_v1"
)
PROVENANCE_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_reporting_parity_provenance_v1"
)
ENABLED_FILENAME: Final[str] = "reporting-enabled-evidence-v1.json"
DISABLED_FILENAME: Final[str] = "reporting-disabled-evidence-v1.json"
PROVENANCE_FILENAME: Final[str] = "reporting-parity-provenance-v1.json"


class InheritanceCrossPathReportingParityProjectionError(ValueError):
    """Reporting-mode captures are stale, substituted, or not exactly equal."""


def _fail(message: str) -> InheritanceCrossPathReportingParityProjectionError:
    return InheritanceCrossPathReportingParityProjectionError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_file(path: Path, *, name: str) -> Path:
    raw = Path(path)
    if not raw.is_absolute() or raw.is_symlink():
        raise _fail(f"{name} must be an absolute non-symlink file")
    resolved = raw.resolve()
    if not resolved.is_file():
        raise _fail(f"{name} is not a regular file")
    return resolved


def _json(path: Path, *, name: str) -> dict[str, Any]:
    source = _absolute_file(path, name=name)
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token: {token}")
            ),
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise _fail(f"{name} is not valid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise _fail(f"{name} must contain one JSON object")
    return dict(value)


def _object(
    value: Any, *, name: str, keys: frozenset[str] | None = None
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(f"{name} must be an object")
    row = dict(value)
    if keys is not None and set(row) != keys:
        raise _fail(f"{name} fields are invalid")
    return row


def _sequence(value: Any, *, name: str) -> list[Any]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
    ):
        raise _fail(f"{name} must be a sequence")
    return list(value)


def _digest(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _fail(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class ReportingParityProjectionRequestV1:
    plan_path: Path
    result_path: Path
    enabled_capture_path: Path
    disabled_capture_path: Path
    enabled_run_id: str
    disabled_run_id: str
    output_dir: Path

    def __post_init__(self) -> None:
        for name in ("enabled_run_id", "disabled_run_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise _fail(f"{name} must be non-empty text")
        if self.enabled_run_id == self.disabled_run_id:
            raise _fail("reporting run IDs must be distinct")


@dataclass(frozen=True, slots=True)
class ReportingParityProjectionReceiptV1:
    output_dir: Path
    enabled_evidence_path: Path
    disabled_evidence_path: Path
    provenance_path: Path
    authoritative_state_sha256: str
    native_reproduction_exposure_sha256: str


def _registered_runs(
    plan_path: Path,
) -> tuple[dict[str, Any], str, list[dict[str, Any]], dict[str, Any]]:
    plan = _json(plan_path, name="runner plan")
    if plan.get("schema_version") != RUNNER_PLAN_SCHEMA_VERSION:
        raise _fail("runner plan schema_version is invalid")
    runner_digest = _digest(
        plan.get("runner_plan_sha256"), name="runner_plan_sha256"
    )
    unsigned = dict(plan)
    unsigned.pop("runner_plan_sha256", None)
    if canonical_sha256(unsigned) != runner_digest:
        raise _fail("runner plan digest is stale")
    gate_plan = _object(plan.get("gate_plan"), name="gate plan")
    gate_plan_digest = _digest(
        plan.get("gate_plan_sha256"), name="gate_plan_sha256"
    )
    if canonical_sha256(gate_plan) != gate_plan_digest:
        raise _fail("gate plan digest is stale")
    execution_authority = _digest(
        plan.get("execution_authority_sha256"),
        name="execution_authority_sha256",
    )
    runs = [
        _object(
            value,
            name=f"runner plan run {index}",
            keys=frozenset(
                {
                    "schema_version",
                    "run_id",
                    "run_kind",
                    "fresh_owner",
                    "cells",
                    "execution_payload",
                }
            ),
        )
        for index, value in enumerate(_sequence(plan.get("runs"), name="runs"))
    ]
    if not runs:
        raise _fail("runner plan contains no runs")
    by_id: dict[str, Any] = {}
    for run in runs:
        run_id = run.get("run_id")
        if (
            run.get("schema_version") != REGISTERED_RUN_SCHEMA_VERSION
            or not isinstance(run_id, str)
            or not run_id
            or run_id in by_id
        ):
            raise _fail("runner plan contains an invalid or duplicate run")
        by_id[run_id] = run
    return by_id, runner_digest, runs, {
        "gate_plan_sha256": gate_plan_digest,
        "execution_callback_id": plan.get("execution_callback_id"),
        "execution_authority_sha256": execution_authority,
    }


def _result_bindings(
    result_path: Path,
    *,
    runner_digest: str,
    registered_run_ids: Sequence[str],
    plan_authority: Mapping[str, Any],
) -> dict[str, str]:
    result = _object(
        _json(result_path, name="runner result"),
        name="runner result",
        keys=frozenset(
            {
                "schema_version",
                "runner_plan_sha256",
                "gate_plan_sha256",
                "execution_callback_id",
                "execution_authority_sha256",
                "capture_artifact_sha256s",
                "gate_result",
            }
        ),
    )
    gate_result = _object(result.get("gate_result"), name="gate result")
    if (
        result.get("schema_version") != RUNNER_RESULT_SCHEMA_VERSION
        or result.get("runner_plan_sha256") != runner_digest
        or result.get("gate_plan_sha256")
        != plan_authority["gate_plan_sha256"]
        or result.get("execution_callback_id")
        != plan_authority["execution_callback_id"]
        or result.get("execution_authority_sha256")
        != plan_authority["execution_authority_sha256"]
        or gate_result.get("schema_version") != GATE_SCHEMA_VERSION
        or gate_result.get("ok") is not True
        or gate_result.get("plan_sha256")
        != plan_authority["gate_plan_sha256"]
    ):
        raise _fail("runner result is not one successful bound gate")
    rows = _sequence(
        result.get("capture_artifact_sha256s"),
        name="capture_artifact_sha256s",
    )
    if len(rows) != len(registered_run_ids):
        raise _fail("runner result capture count differs from the plan")
    bindings: dict[str, str] = {}
    observed_order: list[str] = []
    for index, value in enumerate(rows):
        row = _object(
            value,
            name=f"capture binding {index}",
            keys=frozenset({"run_id", "sha256"}),
        )
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or run_id in bindings:
            raise _fail("runner result has an invalid or duplicate run binding")
        bindings[run_id] = _digest(
            row.get("sha256"), name=f"capture binding {run_id}"
        )
        observed_order.append(run_id)
    if observed_order != list(registered_run_ids):
        raise _fail("runner result capture order differs from the plan")
    return bindings


_CAPTURE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "runner_plan_sha256",
        "execution_authority_sha256",
        "registered_run_sha256",
        "observation_sha256",
        "execution_evidence_sha256",
        "execution_evidence",
        "capture_sha256",
        "capture",
    }
)


def _capture(
    path: Path,
    *,
    name: str,
    expected_file_sha256: str,
    expected_runner_digest: str,
    expected_gate_digest: str,
    expected_execution_authority: str,
    registered_run: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    source = _absolute_file(path, name=name)
    if _sha256(source) != expected_file_sha256:
        raise _fail(f"{name} file hash differs from the runner result")
    artifact = _object(_json(source, name=name), name=name, keys=_CAPTURE_KEYS)
    capture = _object(artifact.get("capture"), name=f"{name}.capture")
    if (
        artifact.get("schema_version") != CAPTURE_ARTIFACT_SCHEMA_VERSION
        or artifact.get("runner_plan_sha256") != expected_runner_digest
        or artifact.get("execution_authority_sha256")
        != expected_execution_authority
        or artifact.get("registered_run_sha256")
        != canonical_sha256(registered_run)
        or artifact.get("capture_sha256") != canonical_sha256(capture)
        or capture.get("plan_sha256") != expected_gate_digest
    ):
        raise _fail(f"{name} is not an immutable capture of its registered run")
    _digest(artifact.get("observation_sha256"), name=f"{name}.observation_sha256")
    if artifact.get("execution_evidence_sha256") != canonical_sha256(
        artifact.get("execution_evidence")
    ):
        raise _fail(f"{name} execution evidence digest is stale")
    if (
        capture.get("run_id") != registered_run.get("run_id")
        or capture.get("run_kind") != registered_run.get("run_kind")
        or capture.get("fresh_owner") is not registered_run.get("fresh_owner")
    ):
        raise _fail(f"{name} run identity differs from preregistration")
    return capture, _sha256(source)


_RUN_NORMALIZED_FIELDS: Final[frozenset[str]] = frozenset({"run_id"})
_EXECUTION_NORMALIZED_FIELDS: Final[frozenset[str]] = frozenset(
    {"reporting_enabled"}
)
_SEGMENT_NORMALIZED_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "run_id",
        "reporting_enabled",
        "run_root",
        "owner_root",
        "checkpoint_root",
        "segment_request_sha256",
    }
)


def _normalized_registration(run: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only reporting mode and its explicitly run-local identities."""

    normalized = {
        name: value for name, value in run.items() if name not in _RUN_NORMALIZED_FIELDS
    }
    execution = _object(
        normalized.get("execution_payload"), name="execution payload"
    )
    normalized_execution = {
        name: value
        for name, value in execution.items()
        if name not in _EXECUTION_NORMALIZED_FIELDS
    }
    segment = normalized_execution.get("segment_request")
    if segment is not None:
        segment_row = _object(segment, name="segment request")
        normalized_execution["segment_request"] = {
            name: value
            for name, value in segment_row.items()
            if name not in _SEGMENT_NORMALIZED_FIELDS
        }
    normalized["execution_payload"] = normalized_execution
    return normalized


def _reporting_rows_by_occurrence(
    artifact_path: Path,
    *,
    name: str,
) -> dict[str, bool]:
    artifact = _json(artifact_path, name=name)
    evidence = _object(
        artifact.get("execution_evidence"),
        name=f"{name}.execution_evidence",
    )
    direct_cells = _sequence(
        evidence.get("direct_cells"),
        name=f"{name}.execution_evidence.direct_cells",
    )
    result: dict[str, bool] = {}
    for index, value in enumerate(direct_cells):
        cell = _object(value, name=f"{name}.direct_cells[{index}]")
        occurrence_id = cell.get("occurrence_id")
        rows_present = cell.get("reporting_rows_present")
        if (
            not isinstance(occurrence_id, str)
            or not occurrence_id
            or occurrence_id in result
            or type(rows_present) is not bool
        ):
            raise _fail(f"{name} contains invalid reporting-row evidence")
        result[occurrence_id] = rows_present
    if not result:
        raise _fail(f"{name} contains no reporting-row evidence")
    return result


def _cells(
    capture: Mapping[str, Any],
    registered_run: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, dict[str, Any]]:
    registered_cells = _sequence(
        registered_run.get("cells"), name=f"{name}.registered cells"
    )
    captured_cells = _sequence(capture.get("cells"), name=f"{name}.captured cells")
    if len(registered_cells) != len(captured_cells):
        raise _fail(f"{name} cell count differs from preregistration")
    result: dict[str, dict[str, Any]] = {}
    for registered, captured in zip(
        registered_cells, captured_cells, strict=True
    ):
        expected = _object(registered, name=f"{name}.registered cell")
        actual = _object(captured, name=f"{name}.captured cell")
        occurrence_id = actual.get("occurrence_id")
        if (
            not isinstance(occurrence_id, str)
            or not occurrence_id
            or occurrence_id in result
            or occurrence_id != expected.get("occurrence_id")
            or actual.get("config") != expected.get("config")
            or actual.get("batch_position") != expected.get("batch_position")
            or actual.get("neighbor_path_ids")
            != expected.get("expected_neighbor_path_ids")
        ):
            raise _fail(f"{name} cell topology differs from preregistration")
        _digest(
            actual.get("final_authority_sha256"),
            name=f"{name}.{occurrence_id}.final_authority_sha256",
        )
        receipts = _sequence(
            actual.get("path_receipts"),
            name=f"{name}.{occurrence_id}.path_receipts",
        )
        path_ids: list[str] = []
        for receipt in receipts:
            row = _object(receipt, name=f"{name}.{occurrence_id}.path receipt")
            path_id = row.get("path_id")
            if not isinstance(path_id, str) or not path_id:
                raise _fail(f"{name} contains an invalid path receipt")
            path_ids.append(path_id)
        if len(path_ids) != len(set(path_ids)):
            raise _fail(f"{name} contains duplicate path receipts")
        result[occurrence_id] = actual
    return result


def _evidence(
    *,
    mode: str,
    reporting_rows_present: bool,
    authoritative_state_sha256: str,
    native_reproduction_exposure_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": REPORTING_MODE_EVIDENCE_SCHEMA_VERSION,
        "mode": mode,
        "authoritative_state_sha256": authoritative_state_sha256,
        "reporting_rows_present": reporting_rows_present,
        "native_reproduction_exposure_sha256": (
            native_reproduction_exposure_sha256
        ),
    }


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    payload = pretty_json(value).encode("ascii")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def project_reporting_parity_evidence_v1(
    request: ReportingParityProjectionRequestV1,
) -> ReportingParityProjectionReceiptV1:
    """Reject publication of new retired-route reporting parity evidence."""

    raise _fail(
        "P12_RETIRED_INHERITANCE_AUTHORITY "
        "cross-path reporting-parity projection is disabled"
    )
    # Historical implementation retained temporarily for evidence reading.
    if type(request) is not ReportingParityProjectionRequestV1:
        raise _fail("projection request must be typed v1")
    destination = Path(request.output_dir)
    if not destination.is_absolute() or destination.exists() or destination.is_symlink():
        raise _fail("output_dir must be a new absolute path")
    for parent in destination.parents:
        if parent.exists() and parent.is_symlink():
            raise _fail("output_dir must not traverse a symlink")

    by_id, runner_digest, runs, plan_authority = _registered_runs(
        request.plan_path
    )
    if request.enabled_run_id not in by_id or request.disabled_run_id not in by_id:
        raise _fail("reporting run ID is absent from the runner plan")
    enabled_run = by_id[request.enabled_run_id]
    disabled_run = by_id[request.disabled_run_id]
    enabled_flag = _object(
        enabled_run.get("execution_payload"), name="enabled execution payload"
    ).get("reporting_enabled")
    disabled_flag = _object(
        disabled_run.get("execution_payload"), name="disabled execution payload"
    ).get("reporting_enabled")
    if enabled_flag is not True or disabled_flag is not False:
        raise _fail("reporting run modes are not enabled/disabled")
    if _normalized_registration(enabled_run) != _normalized_registration(
        disabled_run
    ):
        raise _fail(
            "reporting registrations differ outside reporting mode and "
            "run-local IDs/roots"
        )
    bindings = _result_bindings(
        request.result_path,
        runner_digest=runner_digest,
        registered_run_ids=[run["run_id"] for run in runs],
        plan_authority=plan_authority,
    )
    enabled_capture, enabled_file_hash = _capture(
        request.enabled_capture_path,
        name="reporting enabled capture",
        expected_file_sha256=bindings[request.enabled_run_id],
        expected_runner_digest=runner_digest,
        expected_gate_digest=plan_authority["gate_plan_sha256"],
        expected_execution_authority=plan_authority[
            "execution_authority_sha256"
        ],
        registered_run=enabled_run,
    )
    disabled_capture, disabled_file_hash = _capture(
        request.disabled_capture_path,
        name="reporting disabled capture",
        expected_file_sha256=bindings[request.disabled_run_id],
        expected_runner_digest=runner_digest,
        expected_gate_digest=plan_authority["gate_plan_sha256"],
        expected_execution_authority=plan_authority[
            "execution_authority_sha256"
        ],
        registered_run=disabled_run,
    )
    enabled_cells = _cells(
        enabled_capture, enabled_run, name="reporting enabled capture"
    )
    disabled_cells = _cells(
        disabled_capture, disabled_run, name="reporting disabled capture"
    )
    if set(enabled_cells) != set(disabled_cells):
        raise _fail("reporting captures contain different occurrence IDs")
    enabled_reporting_rows = _reporting_rows_by_occurrence(
        request.enabled_capture_path,
        name="reporting enabled capture",
    )
    disabled_reporting_rows = _reporting_rows_by_occurrence(
        request.disabled_capture_path,
        name="reporting disabled capture",
    )
    if (
        set(enabled_reporting_rows) != set(enabled_cells)
        or set(disabled_reporting_rows) != set(disabled_cells)
    ):
        raise _fail("reporting-row evidence does not cover captured occurrences")
    if not all(enabled_reporting_rows.values()):
        raise _fail("reporting enabled capture has absent reporting rows")
    if any(disabled_reporting_rows.values()):
        raise _fail("reporting disabled capture has present reporting rows")

    authority_rows: list[dict[str, Any]] = []
    exposure_rows: list[dict[str, Any]] = []
    for occurrence_id in sorted(enabled_cells):
        enabled_cell = enabled_cells[occurrence_id]
        disabled_cell = disabled_cells[occurrence_id]
        if (
            enabled_cell["final_authority_sha256"]
            != disabled_cell["final_authority_sha256"]
        ):
            raise _fail(
                f"reporting changed final authority for {occurrence_id!r}"
            )
        if enabled_cell["path_receipts"] != disabled_cell["path_receipts"]:
            raise _fail(
                f"reporting changed path/exposure authority for {occurrence_id!r}"
            )
        authority_rows.append(
            {
                "occurrence_id": occurrence_id,
                "final_authority_sha256": enabled_cell[
                    "final_authority_sha256"
                ],
            }
        )
        exposure_rows.append(
            {
                "occurrence_id": occurrence_id,
                "path_receipts": enabled_cell["path_receipts"],
            }
        )
    authoritative_state_sha256 = canonical_sha256(
        {
            "schema_version": PROJECTION_SCHEMA_VERSION,
            "runner_plan_sha256": runner_digest,
            "cells": authority_rows,
        }
    )
    native_reproduction_exposure_sha256 = canonical_sha256(
        {
            "schema_version": PROJECTION_SCHEMA_VERSION,
            "runner_plan_sha256": runner_digest,
            "cells": exposure_rows,
        }
    )
    enabled_evidence = _evidence(
        mode="enabled",
        reporting_rows_present=next(iter(enabled_reporting_rows.values())),
        authoritative_state_sha256=authoritative_state_sha256,
        native_reproduction_exposure_sha256=(
            native_reproduction_exposure_sha256
        ),
    )
    disabled_evidence = _evidence(
        mode="disabled",
        reporting_rows_present=next(iter(disabled_reporting_rows.values())),
        authoritative_state_sha256=authoritative_state_sha256,
        native_reproduction_exposure_sha256=(
            native_reproduction_exposure_sha256
        ),
    )
    provenance_core = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "runner_plan": {
            "path": str(_absolute_file(request.plan_path, name="runner plan")),
            "sha256": _sha256(Path(request.plan_path)),
            "runner_plan_sha256": runner_digest,
        },
        "runner_result": {
            "path": str(_absolute_file(request.result_path, name="runner result")),
            "sha256": _sha256(Path(request.result_path)),
        },
        "captures": {
            "enabled": {
                "run_id": request.enabled_run_id,
                "path": str(Path(request.enabled_capture_path).resolve()),
                "sha256": enabled_file_hash,
            },
            "disabled": {
                "run_id": request.disabled_run_id,
                "path": str(Path(request.disabled_capture_path).resolve()),
                "sha256": disabled_file_hash,
            },
        },
        "occurrence_ids": sorted(enabled_cells),
        "authoritative_state_sha256": authoritative_state_sha256,
        "native_reproduction_exposure_sha256": (
            native_reproduction_exposure_sha256
        ),
        "exactness": {
            "capture_hashes_and_order": True,
            "registered_run_bindings": True,
            "occurrence_ids": True,
            "final_authority": True,
            "path_and_exposure_authority": True,
        },
    }
    provenance = {
        **provenance_core,
        "provenance_sha256": canonical_sha256(provenance_core),
    }

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    moved = False
    try:
        _write_new(temporary / ENABLED_FILENAME, enabled_evidence)
        _write_new(temporary / DISABLED_FILENAME, disabled_evidence)
        _write_new(temporary / PROVENANCE_FILENAME, provenance)
        os.rename(temporary, destination)
        moved = True
    finally:
        if not moved and temporary.is_dir() and not temporary.is_symlink():
            shutil.rmtree(temporary)
    return ReportingParityProjectionReceiptV1(
        output_dir=destination,
        enabled_evidence_path=destination / ENABLED_FILENAME,
        disabled_evidence_path=destination / DISABLED_FILENAME,
        provenance_path=destination / PROVENANCE_FILENAME,
        authoritative_state_sha256=authoritative_state_sha256,
        native_reproduction_exposure_sha256=(
            native_reproduction_exposure_sha256
        ),
    )


__all__ = [
    "DISABLED_FILENAME",
    "ENABLED_FILENAME",
    "InheritanceCrossPathReportingParityProjectionError",
    "PROVENANCE_FILENAME",
    "ReportingParityProjectionReceiptV1",
    "ReportingParityProjectionRequestV1",
    "project_reporting_parity_evidence_v1",
]
