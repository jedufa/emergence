#!/usr/bin/env python3
from __future__ import annotations

"""Replay Patch P2 g2 sticky-failure receipts without executing CUDA."""

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT_PATH: Final[Path] = (
    ROOT / "docs" / "40_eiel_design" / "patch_resident_failure_contract.json"
)

ANALYZER_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_stage3_patch_p2_sticky_failure_analyzer_v1"
)
REPLAY_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_stage3_patch_p2_sticky_failure_replay_v1"
)
CONTRACT_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_stage3_patch_resident_failure_contract_v1"
)
PROFILE_ID: Final[str] = "patch_p2_resident_g2_sticky_failure_v1"
FIXTURE_ID: Final[str] = "patch_p2_g2_generation1_tick1_event_capacity"

FAILURE_FIELDS: Final[tuple[str, ...]] = (
    "status",
    "failure_code",
    "status_name",
    "stage",
    "cell",
    "generation",
    "tick",
    "attempted_index",
    "required_capacity",
    "available_capacity",
    "sequence",
)
PROGRESS_FIELDS: Final[tuple[str, ...]] = (
    "completed_generations",
    "ticks_started",
    "ticks_completed",
    "last_started_generation",
    "last_started_tick",
    "event_capacity",
    "event_cursor",
    "event_rows_committed",
    "rng_values_consumed",
    "agent_id_allocations",
    "mutation_commits",
    "turnover_commits",
)
STATE_DIGEST_FIELDS: Final[tuple[str, ...]] = (
    "patch_state",
    "agent_state",
    "rng_state_and_cursors",
    "event_history_active_prefix",
    "identity_maps_and_free_queue",
    "mutation_tape",
    "turnover_state",
)
SUMMARY_FIELDS: Final[tuple[str, ...]] = (
    "requested_generations",
    "cells",
    "completed_row_count",
    "validity",
    "status_codes",
    "integer_values",
    "float_values",
    "digest_values",
    "patch_optionality_mask",
)
REQUIRED_NEGATIVE_CONTROLS: Final[tuple[str, ...]] = (
    "flat_g2_same_process_repeat",
    "patch_p2_g2_same_process_repeat",
)
SHA256_LENGTH: Final[int] = 64


class ReplayValidationError(ValueError):
    """A contract or stored replay receipt is missing exact evidence."""


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReplayValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    """Load strict JSON and reject duplicate keys and non-finite values."""

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ReplayValidationError(f"non-finite JSON constant: {value}")
            ),
        )
    except ReplayValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReplayValidationError(f"cannot load strict JSON {path}: {exc}") from exc


def _fail(path: str, message: str) -> None:
    raise ReplayValidationError(f"{path}: {message}")


def _mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: Sequence[str], *, path: str) -> None:
    missing = sorted(set(expected) - set(value))
    extra = sorted(set(value) - set(expected))
    if missing:
        _fail(path, f"missing required field(s): {missing}")
    if extra:
        _fail(path, f"additional field(s) forbidden: {extra}")


def _integer(value: Any, *, path: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(path, f"must be an integer >= {minimum}")
    return value


def _string(value: Any, *, path: str) -> str:
    if type(value) is not str or not value:
        _fail(path, "must be a non-empty string")
    return value


def _string_sequence(value: Any, *, path: str) -> tuple[str, ...]:
    if type(value) is not list:
        _fail(path, "must be an array")
    result: list[str] = []
    for index, item in enumerate(value):
        result.append(_string(item, path=f"{path}[{index}]"))
    return tuple(result)


def _sha256_text(value: Any, *, path: str) -> str:
    text = _string(value, path=path)
    if len(text) != SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in text
    ):
        _fail(path, "must be a lowercase SHA-256 digest")
    return text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ReplayValidationError(f"cannot hash source artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReplayValidationError(f"receipt is not strict JSON: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def validate_contract(contract: Any) -> dict[str, Any]:
    """Validate the exact normative contract understood by this analyzer."""

    contract = _mapping(contract, path="$contract")
    _exact_keys(
        contract,
        (
            "schema_version",
            "profile_id",
            "status",
            "failure_fixture",
            "first_failure_latch",
            "after_failure_must_not",
            "after_failure_may",
            "completed_generations_after_fixture",
            "compact_summary_rows_after_failure",
            "required_negative_controls",
            "qualification",
        ),
        path="$contract",
    )
    expected_scalars = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "profile_id": PROFILE_ID,
        "status": "normative_contract_live_negative_control_pending",
        "completed_generations_after_fixture": 0,
        "compact_summary_rows_after_failure": "invalid_noop",
        "qualification": "separate_sticky_failure_integration_gate_required",
    }
    for field, expected in expected_scalars.items():
        if contract[field] != expected:
            _fail(f"$contract.{field}", f"expected {expected!r}")

    fixture = _mapping(contract["failure_fixture"], path="$contract.failure_fixture")
    expected_fixture = {
        "fixture_id": FIXTURE_ID,
        "generation_index_zero_based": 0,
        "generation_ordinal_one_based": 1,
        "tick_index_zero_based": 0,
        "tick_ordinal_one_based": 1,
        "stage": "birth_event_append",
        "status_code": 1,
        "status_name": "event_capacity_exceeded",
        "injection": "initialize_event_cursor_at_declared_capacity_before_dispatch",
    }
    _exact_keys(fixture, tuple(expected_fixture), path="$contract.failure_fixture")
    for field, expected in expected_fixture.items():
        if fixture[field] != expected:
            _fail(f"$contract.failure_fixture.{field}", f"expected {expected!r}")

    expected_latch = (
        "failure_code",
        "stage",
        "cell",
        "generation",
        "tick",
        "attempted_index",
        "required_capacity",
        "available_capacity",
        "sequence",
    )
    if _string_sequence(
        contract["first_failure_latch"], path="$contract.first_failure_latch"
    ) != expected_latch:
        _fail("$contract.first_failure_latch", "unsupported metadata surface")
    expected_must_not = (
        "execute_later_ticks",
        "consume_rng",
        "append_trajectory_events",
        "allocate_agent_ids",
        "mutate_patch_or_agent_state",
        "commit_failed_or_later_turnover",
        "mark_later_summary_rows_valid",
        "increment_completed_generations_past_reality",
    )
    if _string_sequence(
        contract["after_failure_must_not"], path="$contract.after_failure_must_not"
    ) != expected_must_not:
        _fail("$contract.after_failure_must_not", "unsupported fail-stop surface")
    expected_may = (
        "publish_the_immutable_failure_record",
        "perform_one_final_synchronization",
        "export_frozen_diagnostics",
    )
    if _string_sequence(
        contract["after_failure_may"], path="$contract.after_failure_may"
    ) != expected_may:
        _fail("$contract.after_failure_may", "unsupported post-failure allowance")
    if _string_sequence(
        contract["required_negative_controls"],
        path="$contract.required_negative_controls",
    ) != REQUIRED_NEGATIVE_CONTROLS:
        _fail("$contract.required_negative_controls", "unsupported controls")
    return dict(fixture)


def _validate_failure_status(
    value: Any, *, path: str, fixture: Mapping[str, Any]
) -> dict[str, Any]:
    status = _mapping(value, path=path)
    _exact_keys(status, FAILURE_FIELDS, path=path)
    expected = {
        "status": "failed",
        "failure_code": fixture["status_code"],
        "status_name": fixture["status_name"],
        "stage": fixture["stage"],
        "cell": 0,
        "generation": fixture["generation_index_zero_based"],
        "tick": fixture["tick_index_zero_based"],
        "sequence": 1,
    }
    for field, expected_value in expected.items():
        if status[field] != expected_value:
            _fail(f"{path}.{field}", f"expected {expected_value!r}")
    for field in (
        "failure_code",
        "cell",
        "generation",
        "tick",
        "attempted_index",
        "required_capacity",
        "available_capacity",
        "sequence",
    ):
        _integer(status[field], path=f"{path}.{field}")
    available = status["available_capacity"]
    attempted = status["attempted_index"]
    required = status["required_capacity"]
    if attempted != available:
        _fail(
            f"{path}.attempted_index",
            "must equal declared capacity for the injected append",
        )
    if required != attempted + 1:
        _fail(
            f"{path}.required_capacity",
            "must be exactly one row beyond the attempted append index",
        )
    return dict(status)


def _validate_progress(
    value: Any,
    *,
    path: str,
    fixture: Mapping[str, Any],
    failure_status: Mapping[str, Any],
    completed_generations: int,
) -> dict[str, int]:
    progress = _mapping(value, path=path)
    _exact_keys(progress, PROGRESS_FIELDS, path=path)
    result = {
        field: _integer(progress[field], path=f"{path}.{field}")
        for field in PROGRESS_FIELDS
    }
    expected = {
        "completed_generations": completed_generations,
        "ticks_started": 1,
        "ticks_completed": 0,
        "last_started_generation": fixture["generation_index_zero_based"],
        "last_started_tick": fixture["tick_index_zero_based"],
        "mutation_commits": 0,
        "turnover_commits": 0,
    }
    for field, expected_value in expected.items():
        if result[field] != expected_value:
            _fail(f"{path}.{field}", f"expected {expected_value}")
    if result["event_capacity"] != failure_status["available_capacity"]:
        _fail(f"{path}.event_capacity", "does not match failure metadata")
    if result["event_cursor"] != result["event_capacity"]:
        _fail(f"{path}.event_cursor", "fixture must begin at declared capacity")
    if result["event_rows_committed"] != result["event_cursor"]:
        _fail(f"{path}.event_rows_committed", "must match the frozen cursor")
    return result


def _validate_state_digests(value: Any, *, path: str) -> dict[str, str]:
    digests = _mapping(value, path=path)
    _exact_keys(digests, STATE_DIGEST_FIELDS, path=path)
    return {
        field: _sha256_text(digests[field], path=f"{path}.{field}")
        for field in STATE_DIGEST_FIELDS
    }


def _zero_scalar(value: Any, *, path: str, kind: str) -> None:
    if kind == "float":
        if type(value) is not float or value != 0.0:
            _fail(path, "must be the float64 no-op value 0.0")
        return
    if type(value) is not int or value != 0:
        _fail(path, "must be the integer no-op value 0")


def _validate_nonempty_zero_payload(value: Any, *, path: str, kind: str) -> None:
    if type(value) is list:
        if not value:
            _fail(path, "must contain the preallocated plane payload")
        for index, item in enumerate(value):
            _validate_nonempty_zero_payload(item, path=f"{path}[{index}]", kind=kind)
        return
    _zero_scalar(value, path=path, kind=kind)


def _validate_zero_plane(
    value: Any,
    *,
    path: str,
    generations: int,
    cells: int,
    kind: str,
    trailing_payload: bool,
) -> None:
    if type(value) is not list or len(value) != generations:
        _fail(path, f"must have exactly {generations} generation rows")
    for generation, row in enumerate(value):
        row_path = f"{path}[{generation}]"
        if type(row) is not list or len(row) != cells:
            _fail(row_path, f"must have exactly {cells} cell slots")
        for cell, payload in enumerate(row):
            payload_path = f"{row_path}[{cell}]"
            if trailing_payload:
                if type(payload) is not list:
                    _fail(payload_path, "must include a trailing plane dimension")
                _validate_nonempty_zero_payload(payload, path=payload_path, kind=kind)
            else:
                _zero_scalar(payload, path=payload_path, kind=kind)


def _validate_compact_summary(
    value: Any, *, path: str, completed_generations: int
) -> dict[str, Any]:
    summary = _mapping(value, path=path)
    _exact_keys(summary, SUMMARY_FIELDS, path=path)
    generations = _integer(
        summary["requested_generations"], path=f"{path}.requested_generations", minimum=1
    )
    cells = _integer(summary["cells"], path=f"{path}.cells", minimum=1)
    completed = _integer(
        summary["completed_row_count"], path=f"{path}.completed_row_count"
    )
    if generations != 2:
        _fail(f"{path}.requested_generations", "Patch P2 g2 requires exactly 2")
    if cells != 1:
        _fail(f"{path}.cells", "Patch P2 g2 requires exactly 1")
    if completed != completed_generations:
        _fail(
            f"{path}.completed_row_count",
            "must equal the contract completed-generations count",
        )
    _validate_zero_plane(
        summary["validity"],
        path=f"{path}.validity",
        generations=generations,
        cells=cells,
        kind="integer",
        trailing_payload=False,
    )
    _validate_zero_plane(
        summary["patch_optionality_mask"],
        path=f"{path}.patch_optionality_mask",
        generations=generations,
        cells=cells,
        kind="integer",
        trailing_payload=False,
    )
    for field, kind in (
        ("status_codes", "integer"),
        ("integer_values", "integer"),
        ("float_values", "float"),
        ("digest_values", "integer"),
    ):
        _validate_zero_plane(
            summary[field],
            path=f"{path}.{field}",
            generations=generations,
            cells=cells,
            kind=kind,
            trailing_payload=True,
        )
    return dict(summary)


def _validate_snapshot(
    value: Any,
    *,
    path: str,
    fixture: Mapping[str, Any],
    completed_generations: int,
) -> dict[str, Any]:
    snapshot = _mapping(value, path=path)
    _exact_keys(
        snapshot,
        ("failure_status", "progress", "state_digests", "compact_summary"),
        path=path,
    )
    failure_status = _validate_failure_status(
        snapshot["failure_status"], path=f"{path}.failure_status", fixture=fixture
    )
    return {
        "failure_status": failure_status,
        "progress": _validate_progress(
            snapshot["progress"],
            path=f"{path}.progress",
            fixture=fixture,
            failure_status=failure_status,
            completed_generations=completed_generations,
        ),
        "state_digests": _validate_state_digests(
            snapshot["state_digests"], path=f"{path}.state_digests"
        ),
        "compact_summary": _validate_compact_summary(
            snapshot["compact_summary"],
            path=f"{path}.compact_summary",
            completed_generations=completed_generations,
        ),
    }


def _validate_source_artifacts(
    value: Any, *, evidence_kind: str, base_directory: Path, path: str
) -> list[dict[str, str]]:
    if type(value) is not list:
        _fail(path, "must be an array")
    if evidence_kind == "live" and not value:
        _fail(path, "live evidence requires at least one digest-bound source artifact")
    validated: list[dict[str, str]] = []
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        binding = _mapping(item, path=item_path)
        _exact_keys(binding, ("path", "sha256"), path=item_path)
        supplied_path = Path(_string(binding["path"], path=f"{item_path}.path"))
        artifact_path = (
            supplied_path
            if supplied_path.is_absolute()
            else base_directory / supplied_path
        ).resolve()
        expected = _sha256_text(binding["sha256"], path=f"{item_path}.sha256")
        if not artifact_path.is_file():
            _fail(f"{item_path}.path", f"source artifact is missing: {artifact_path}")
        actual = _sha256_file(artifact_path)
        if actual != expected:
            _fail(f"{item_path}.sha256", f"digest mismatch for {artifact_path}")
        validated.append({"path": str(artifact_path), "sha256": actual})
    return validated


def _validate_negative_control(value: Any, *, path: str) -> dict[str, Any]:
    control = _mapping(value, path=path)
    _exact_keys(control, ("process_identity", "artifact_sha256", "runs"), path=path)
    process_identity = _string(control["process_identity"], path=f"{path}.process_identity")
    artifact_sha256 = _sha256_text(
        control["artifact_sha256"], path=f"{path}.artifact_sha256"
    )
    runs = control["runs"]
    if type(runs) is not list or len(runs) != 2:
        _fail(f"{path}.runs", "must contain exactly two same-process dispatch receipts")
    validated_runs: list[dict[str, Any]] = []
    for index, run_value in enumerate(runs):
        run_path = f"{path}.runs[{index}]"
        run = _mapping(run_value, path=run_path)
        _exact_keys(
            run,
            ("dispatch_ordinal", "status", "completed_generations", "terminal_digest"),
            path=run_path,
        )
        ordinal = _integer(
            run["dispatch_ordinal"], path=f"{run_path}.dispatch_ordinal", minimum=1
        )
        if ordinal != index + 1:
            _fail(f"{run_path}.dispatch_ordinal", f"expected {index + 1}")
        if run["status"] != "completed":
            _fail(f"{run_path}.status", "expected 'completed'")
        generations = _integer(
            run["completed_generations"], path=f"{run_path}.completed_generations"
        )
        if generations != 2:
            _fail(f"{run_path}.completed_generations", "expected 2")
        terminal_digest = _sha256_text(
            run["terminal_digest"], path=f"{run_path}.terminal_digest"
        )
        validated_runs.append(
            {
                "dispatch_ordinal": ordinal,
                "status": "completed",
                "completed_generations": generations,
                "terminal_digest": terminal_digest,
            }
        )
    if validated_runs[0]["terminal_digest"] != validated_runs[1]["terminal_digest"]:
        _fail(f"{path}.runs[1].terminal_digest", "same-process repeat is not exact")
    return {
        "process_identity": process_identity,
        "artifact_sha256": artifact_sha256,
        "runs": validated_runs,
        "exact": True,
    }


def validate_replay(
    replay: Any,
    contract: Any,
    *,
    base_directory: Path | None = None,
) -> dict[str, Any]:
    """Validate one stored sticky-failure replay and return a bounded verdict."""

    fixture = validate_contract(contract)
    replay = _mapping(replay, path="$replay")
    _exact_keys(
        replay,
        (
            "schema_version",
            "evidence_kind",
            "fixture_id",
            "source_artifacts",
            "snapshots",
            "negative_controls",
        ),
        path="$replay",
    )
    if replay["schema_version"] != REPLAY_SCHEMA_VERSION:
        _fail("$replay.schema_version", f"expected {REPLAY_SCHEMA_VERSION!r}")
    evidence_kind = _string(replay["evidence_kind"], path="$replay.evidence_kind")
    if evidence_kind not in {"synthetic", "live"}:
        _fail("$replay.evidence_kind", "must be 'synthetic' or 'live'")
    if replay["fixture_id"] != fixture["fixture_id"]:
        _fail("$replay.fixture_id", "does not match the normative fixture")

    source_artifacts = _validate_source_artifacts(
        replay["source_artifacts"],
        evidence_kind=evidence_kind,
        base_directory=(base_directory or Path.cwd()).resolve(),
        path="$replay.source_artifacts",
    )
    snapshots = _mapping(replay["snapshots"], path="$replay.snapshots")
    _exact_keys(snapshots, ("at_latch", "final"), path="$replay.snapshots")
    completed_generations = 0
    at_latch = _validate_snapshot(
        snapshots["at_latch"],
        path="$replay.snapshots.at_latch",
        fixture=fixture,
        completed_generations=completed_generations,
    )
    final = _validate_snapshot(
        snapshots["final"],
        path="$replay.snapshots.final",
        fixture=fixture,
        completed_generations=completed_generations,
    )
    for field in ("failure_status", "progress", "state_digests", "compact_summary"):
        if at_latch[field] != final[field]:
            _fail(
                f"$replay.snapshots.final.{field}",
                "changed after the first failure latch",
            )

    controls = _mapping(replay["negative_controls"], path="$replay.negative_controls")
    _exact_keys(controls, REQUIRED_NEGATIVE_CONTROLS, path="$replay.negative_controls")
    validated_controls = {
        name: _validate_negative_control(
            controls[name], path=f"$replay.negative_controls.{name}"
        )
        for name in REQUIRED_NEGATIVE_CONTROLS
    }
    artifact_digests = {
        control["artifact_sha256"] for control in validated_controls.values()
    }
    if len(artifact_digests) != 1:
        _fail(
            "$replay.negative_controls",
            "flat and Patch controls must use the same artifact",
        )
    if evidence_kind == "live" and not artifact_digests.issubset(
        {binding["sha256"] for binding in source_artifacts}
    ):
        _fail(
            "$replay.source_artifacts",
            "must include the artifact used by both live negative controls",
        )

    live_fixture_status = "validated" if evidence_kind == "live" else "pending"
    qualification = (
        "qualified_by_supplied_live_artifacts"
        if evidence_kind == "live"
        else "analyzer_valid_live_fixture_pending"
    )
    return {
        "ok": True,
        "analyzer_valid": True,
        "analyzer_schema_version": ANALYZER_SCHEMA_VERSION,
        "replay_schema_version": REPLAY_SCHEMA_VERSION,
        "artifact_replay_performed": True,
        "replay_contract_satisfied": True,
        "evidence_kind": evidence_kind,
        "live_fixture_status": live_fixture_status,
        "qualification": qualification,
        "fixture_id": fixture["fixture_id"],
        "first_failure_metadata": at_latch["failure_status"],
        "completed_generations": completed_generations,
        "checks": {
            "first_failure_metadata_exact_and_immutable": True,
            "completed_generations_exact": True,
            "no_later_tick_progress": True,
            "no_rng_progress_after_latch": True,
            "no_event_progress_after_latch": True,
            "no_agent_id_allocation_after_latch": True,
            "no_patch_or_agent_mutation_after_latch": True,
            "no_mutation_commit_after_latch": True,
            "no_turnover_commit_after_latch": True,
            "compact_summary_tail_invalid_noop_and_immutable": True,
            "required_same_process_negative_controls_exact": True,
        },
        "negative_controls": {
            name: {
                "exact": True,
                "process_identity": control["process_identity"],
                "artifact_sha256": control["artifact_sha256"],
            }
            for name, control in validated_controls.items()
        },
        "source_artifacts": source_artifacts,
        "replay_sha256": _canonical_sha256(replay),
        "simulation_runs_started": 0,
        "cuda_builds_started": 0,
        "cuda_dispatches_started": 0,
    }


def validate_analyzer_contract(contract: Any) -> dict[str, Any]:
    """Validate analyzer/contract compatibility without claiming a live fixture."""

    fixture = validate_contract(contract)
    return {
        "ok": True,
        "analyzer_valid": True,
        "analyzer_schema_version": ANALYZER_SCHEMA_VERSION,
        "replay_schema_version": REPLAY_SCHEMA_VERSION,
        "artifact_replay_performed": False,
        "replay_contract_satisfied": None,
        "fixture_id": fixture["fixture_id"],
        "live_fixture_status": "pending",
        "qualification": "analyzer_valid_live_fixture_pending",
        "simulation_runs_started": 0,
        "cuda_builds_started": 0,
        "cuda_dispatches_started": 0,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument(
        "--artifact",
        type=Path,
        help="stored synthetic or live replay receipt; CUDA is never invoked",
    )
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    analyzer_valid = False
    try:
        contract = load_json(args.contract)
        validate_contract(contract)
        analyzer_valid = True
        if args.artifact is None:
            result = validate_analyzer_contract(contract)
        else:
            replay_path = args.artifact.expanduser().resolve()
            result = validate_replay(
                load_json(replay_path),
                contract,
                base_directory=replay_path.parent,
            )
    except ReplayValidationError as exc:
        result = {
            "ok": False,
            "analyzer_valid": analyzer_valid,
            "analyzer_schema_version": ANALYZER_SCHEMA_VERSION,
            "artifact_replay_performed": args.artifact is not None,
            "replay_contract_satisfied": False if analyzer_valid else None,
            "live_fixture_status": "pending",
            "qualification": "rejected_fail_closed",
            "error": str(exc),
            "simulation_runs_started": 0,
            "cuda_builds_started": 0,
            "cuda_dispatches_started": 0,
        }
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
