"""Materialize long-horizon qualification evidence from two canonical runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from .canonical import canonical_sha256
from .long_horizon_qualification import (
    BOUNDED_ROUTE_QUALIFIED,
    LONG_HORIZON_EVIDENCE_SCHEMA_VERSION,
    LONG_HORIZON_RUNABILITY_QUALIFIED,
    LONG_HORIZON_RUNABILITY_ROUTE_SCOPE,
    LongHorizonQualificationError,
    validate_long_horizon_evidence_v1,
)
from .orchestration import atomic_write_json
from .tick_native_control import (
    FORBIDDEN_ACTIVE_GENERATION_FIELDS_V1,
    TickNativeControlError,
    assert_tick_native_plan_payload_v1,
)
from .physical_b136_acs_parity_v1 import (
    EventSinkError,
    event_history_digest_from_references_v1,
    validate_event_sink_prefix_reference_v1,
)


class LongHorizonEvidenceMaterializationError(RuntimeError):
    """Canonical run evidence is absent, inconsistent, or incomplete."""


_METADATA_STATUS_PREFIXES = (
    "_artifacts/",
    "docs/900_lifecycle/940_incoming/",
    "tools/evidence/librarian/catalog_source/00-catalog/",
)
_TRACKED_METADATA_STATUS_PREFIXES = (
    "docs/200_strategy-science-programme/200_current-strategy/"
    "EIEL-DOC-002526--unregistered-eiel-canonical-science-log.md",
    "tools/evidence/librarian/catalog_source/00-catalog/",
)
_B136_CELL_COUNT = 136
_B136_WIDTH = 32
_B136_POPULATION_CAPACITY = 256
_B136_CONTROLLER_CAPACITY = 512
_B136_TOTAL_TICKS = 5000
_B136_RESTORE_TICK = 2500
_B136_OWNER_TRANSITIONS = (1120, 2240, 3360, 4480)
_B136_EVENT_RING_CAPACITY = 41_804 * _B136_CELL_COUNT
_B136_RESIDENT_EVENT_BYTES = 801_633_504


def _porcelain_status_row(line: str) -> tuple[str, str]:
    if len(line) >= 4 and line[2] == " " and line[:2] in {"??", " M", "M "}:
        return line[:2], line[3:]
    # ExecutionRequestV2 historically stripped the leading whitespace from the
    # first porcelain row. Preserve that authenticated representation while
    # comparing it to the live, correctly spaced row.
    if len(line) >= 3 and line.startswith("M ") and line[2] != " ":
        return " M", line[2:]
    raise LongHorizonEvidenceMaterializationError(
        "formal qualification source contains an unsupported status row"
    )


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LongHorizonEvidenceMaterializationError(f"{name} must be an object")
    return dict(value)


def _read_object(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise LongHorizonEvidenceMaterializationError(
            f"{name} must be a regular file: {path}"
        )
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), name=name)
    except (OSError, json.JSONDecodeError) as exc:
        raise LongHorizonEvidenceMaterializationError(
            f"{name} is unreadable: {exc}"
        ) from exc


def _file_sha256(path: Path, *, name: str) -> str:
    if not path.is_file() or path.is_symlink():
        raise LongHorizonEvidenceMaterializationError(
            f"{name} must be a regular non-symlink file: {path}"
        )
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise LongHorizonEvidenceMaterializationError(
            f"{name} is unreadable: {exc}"
        ) from exc


def _sha256_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise LongHorizonEvidenceMaterializationError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _created_at(value: Any, *, name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise LongHorizonEvidenceMaterializationError(
            f"{name} must be an ISO-8601 UTC timestamp"
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LongHorizonEvidenceMaterializationError(
            f"{name} is not parseable as an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise LongHorizonEvidenceMaterializationError(
            f"{name} must carry an explicit UTC offset"
        )
    return parsed.astimezone(timezone.utc)


def _git_bytes(repository_root: Path, *arguments: str, name: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LongHorizonEvidenceMaterializationError(
            f"{name} is not available from the committed Git source"
        ) from exc
    return completed.stdout


def _assert_no_generation_authority(value: Any, *, name: str) -> None:
    found: list[str] = []

    def walk(item: Any, path: tuple[str, ...]) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                key_text = str(key)
                if key_text == "legacy_compatibility":
                    continue
                child_path = (*path, key_text)
                if key_text in FORBIDDEN_ACTIVE_GENERATION_FIELDS_V1:
                    found.append(".".join(child_path))
                walk(child, child_path)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            for index, child in enumerate(item):
                walk(child, (*path, f"[{index}]"))

    walk(value, ())
    if found:
        raise LongHorizonEvidenceMaterializationError(
            f"{name} contains active generation authority: {sorted(found)}"
        )


def _validated_result(root: Path) -> dict[str, Any]:
    result = _read_object(root / "execution_result.json", name="execution result")
    supplied = result.get("result_sha256")
    core = dict(result)
    core.pop("result_sha256", None)
    if supplied != canonical_sha256(core):
        raise LongHorizonEvidenceMaterializationError(
            "execution result SHA-256 differs"
        )
    if (
        result.get("status") != "completed"
        or result.get("result_class") != "SUCCESS"
        or result.get("ok") is not True
        or result.get("execution_status") != "completed"
        or result.get("purpose") != "qualification"
        or result.get("invocation_mode") != "execute"
    ):
        raise LongHorizonEvidenceMaterializationError(
            f"canonical qualification execution is not a verified success: {root}"
        )
    _sha256_text(result.get("request_sha256"), name="request_sha256")
    return result


def _normalized_twin_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(plan))
    value.pop("plan_sha256", None)
    value.pop("campaign_id", None)
    value.pop("description", None)
    value.pop("no_claims", None)
    output = value.get("output")
    if isinstance(output, dict):
        output.pop("runtime_root", None)
    execution = value.get("execution")
    if isinstance(execution, dict):
        execution.pop("deliberate_restore_tick", None)
        execution.pop("deliberate_restore_ticks", None)
    return value


def _validated_git_identity(source_identity: Mapping[str, Any]) -> dict[str, Any]:
    git = _object(source_identity.get("git"), name="Git source identity")
    commit = git.get("commit")
    repository_root = git.get("repository_root")
    clean = git.get("working_tree_clean")
    status_lines = git.get("working_tree_status_lines")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise LongHorizonEvidenceMaterializationError(
            "Git source identity has an invalid committed source"
        )
    if (
        not isinstance(repository_root, str)
        or not repository_root
        or not Path(repository_root).is_absolute()
    ):
        raise LongHorizonEvidenceMaterializationError(
            "Git source identity lacks an absolute repository_root"
        )
    repository_path = Path(repository_root)
    if repository_path.is_symlink() or not repository_path.is_dir():
        raise LongHorizonEvidenceMaterializationError(
            "Git source identity repository_root is not an existing directory"
        )
    repository_path = repository_path.resolve()
    try:
        top_level = _git_bytes(
            repository_path,
            "rev-parse",
            "--show-toplevel",
            name="Git worktree root",
        ).decode("utf-8", errors="strict").strip()
        inside = _git_bytes(
            repository_path,
            "rev-parse",
            "--is-inside-work-tree",
            name="Git worktree marker",
        ).decode("ascii", errors="strict").strip()
    except UnicodeError as exc:
        raise LongHorizonEvidenceMaterializationError(
            "Git worktree identity output is malformed"
        ) from exc
    if inside != "true" or not top_level or Path(top_level).resolve() != repository_path:
        raise LongHorizonEvidenceMaterializationError(
            "Git source identity repository_root is not an existing Git worktree"
        )
    _git_bytes(
        repository_path,
        "cat-file",
        "-e",
        f"{commit}^{{commit}}",
        name="Git commit",
    )
    try:
        live_head = _git_bytes(
            repository_path,
            "rev-parse",
            "HEAD",
            name="live Git HEAD",
        ).decode("ascii", errors="strict").strip()
        live_status_lines = _git_bytes(
            repository_path,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            name="live Git working-tree status",
        ).decode("utf-8", errors="strict").splitlines()
    except UnicodeError as exc:
        raise LongHorizonEvidenceMaterializationError(
            "live Git identity output is malformed"
        ) from exc
    if live_head != commit:
        ancestry = subprocess.run(
            [
                "git",
                "-C",
                str(repository_path),
                "merge-base",
                "--is-ancestor",
                commit,
                live_head,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if ancestry.returncode != 0:
            raise LongHorizonEvidenceMaterializationError(
                "declared Git commit is not an ancestor of live HEAD"
            )
    if type(clean) is not bool or not isinstance(status_lines, list):
        raise LongHorizonEvidenceMaterializationError(
            "Git source identity lacks an exact working-tree status"
        )
    if any(not isinstance(line, str) for line in status_lines):
        raise LongHorizonEvidenceMaterializationError(
            "Git working-tree status rows are malformed"
        )
    parsed_status_lines = []
    for line in status_lines:
        status, path = _porcelain_status_row(line)
        parsed_status_lines.append((status, path))
        if (
            not path
            or path.startswith("/")
            or path.startswith("./")
            or path.startswith("../")
            or "/../" in path
        ):
            raise LongHorizonEvidenceMaterializationError(
                "source status path is malformed"
            )
        if status == "??":
            allowed_prefixes = _METADATA_STATUS_PREFIXES
        elif status in {" M", "M "}:
            allowed_prefixes = _TRACKED_METADATA_STATUS_PREFIXES
        if not any(path.startswith(prefix) for prefix in allowed_prefixes):
            raise LongHorizonEvidenceMaterializationError(
                "source state is outside the metadata allowlist"
            )
    if clean != (not status_lines):
        raise LongHorizonEvidenceMaterializationError(
            "Git working_tree_clean disagrees with ordered working-tree status"
        )
    status_sha256 = _sha256_text(
        git.get("working_tree_status_sha256"),
        name="working_tree_status_sha256",
    )
    if status_sha256 != canonical_sha256(status_lines):
        raise LongHorizonEvidenceMaterializationError(
            "Git working-tree status SHA-256 differs from ordered status rows"
        )
    parsed_live_status_lines = [_porcelain_status_row(line) for line in live_status_lines]
    if parsed_status_lines != parsed_live_status_lines or clean != (not live_status_lines):
        raise LongHorizonEvidenceMaterializationError(
            "Git declared commit/status does not match exact committed repository/status"
        )
    return {
        "commit": commit,
        "repository_root": str(repository_path),
        "working_tree_clean": clean,
        "working_tree_status_lines": list(status_lines),
        "working_tree_status_sha256": status_sha256,
    }


def _validated_campaign_spec_binding(
    *, source: Mapping[str, Any]
) -> tuple[str, str]:
    git = _object(source.get("git"), name="Git source identity")
    repository_root = Path(str(git.get("repository_root"))).resolve()
    spec_raw = source.get("campaign_spec_path")
    if not isinstance(spec_raw, str) or not spec_raw:
        raise LongHorizonEvidenceMaterializationError(
            "formal evidence must bind a campaign specification"
        )
    spec_path = Path(spec_raw)
    if not spec_path.is_absolute() or spec_path.is_symlink() or not spec_path.is_file():
        raise LongHorizonEvidenceMaterializationError(
            "campaign specification must be an existing regular file"
        )
    spec_path = spec_path.resolve()
    try:
        relative = spec_path.relative_to(repository_root)
    except ValueError as exc:
        raise LongHorizonEvidenceMaterializationError(
            "campaign specification escapes the committed Git worktree"
        ) from exc
    if not relative.parts:
        raise LongHorizonEvidenceMaterializationError(
            "campaign specification path is empty"
        )
    relative_text = relative.as_posix()
    committed_bytes = _git_bytes(
        repository_root,
        "show",
        f"{git['commit']}:{relative_text}",
        name="campaign specification at committed path",
    )
    current_bytes = spec_path.read_bytes()
    declared_sha256 = _sha256_text(
        source.get("campaign_spec_sha256"), name="campaign_spec_sha256"
    )
    if (
        current_bytes != committed_bytes
        or hashlib.sha256(current_bytes).hexdigest() != declared_sha256
        or hashlib.sha256(committed_bytes).hexdigest() != declared_sha256
    ):
        raise LongHorizonEvidenceMaterializationError(
            "campaign specification bytes differ from the bound Git commit path"
        )
    return str(spec_path), declared_sha256


def _validated_formal_source(
    *, plan: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    lifecycle = _object(
        plan.get("artifact_lifecycle"), name="artifact lifecycle contract"
    )
    if lifecycle.get("lifecycle_profile") != "QUALIFICATION":
        raise LongHorizonEvidenceMaterializationError(
            "formal evidence requires the QUALIFICATION artifact lifecycle"
        )
    baseline = _object(result.get("baseline"), name="Production Baseline V2 binding")
    baseline_authority = {
        "profile_id": baseline.get("profile_id"),
        "profile_version": baseline.get("profile_version"),
        "overall_status": baseline.get("overall_status"),
        "execution_admission_status": baseline.get("execution_admission_status"),
        "admitted": baseline.get("admitted"),
        "configuration_eligible": baseline.get("configuration_eligible"),
        "configuration_status": baseline.get("configuration_status"),
        "operational_status": baseline.get("operational_status"),
        "deviations": baseline.get("deviations"),
    }
    if baseline_authority != {
        "profile_id": "EIEL_PRODUCTION_BASELINE_V1",
        "profile_version": 2,
        "overall_status": "BASELINE_CONFORMANT",
        "execution_admission_status": "ADMITTED",
        "admitted": True,
        "configuration_eligible": True,
        "configuration_status": "CONFORMANT",
        "operational_status": "CONFORMANT",
        "deviations": [],
    }:
        raise LongHorizonEvidenceMaterializationError(
            "formal evidence requires an admitted, conformant Production Baseline V2"
        )
    if baseline.get("qualification_status") not in {"UNQUALIFIED", "PARTIAL"}:
        raise LongHorizonEvidenceMaterializationError(
            "formal evidence requires an unqualified or partially qualified "
            "Production Baseline V2"
        )
    source_identity = _object(result.get("source_identity"), name="source identity")
    if source_identity.get("bound_plan_sha256") != plan.get("plan_sha256"):
        raise LongHorizonEvidenceMaterializationError(
            "source identity does not bind the persisted execution plan"
        )
    git = _validated_git_identity(source_identity)
    source_with_git = dict(source_identity)
    source_with_git["git"] = git
    spec_path, spec_sha256 = _validated_campaign_spec_binding(
        source=source_with_git
    )
    return {
        "git": git,
        "campaign_spec_path": spec_path,
        "campaign_spec_sha256": spec_sha256,
    }


def _validate_b136_plan(plan: Mapping[str, Any]) -> None:
    cells = plan.get("cells")
    physical = plan.get("physical_batches")
    if (
        not isinstance(cells, list)
        or len(cells) != _B136_CELL_COUNT
        or any(not isinstance(row, Mapping) for row in cells)
        or len({row.get("cell_id") for row in cells}) != _B136_CELL_COUNT
        or any(not isinstance(row.get("cell_id"), str) for row in cells)
        or not isinstance(physical, list)
        or len(physical) != 1
    ):
        raise LongHorizonEvidenceMaterializationError(
            "formal long-horizon evidence requires one physical B136 batch"
        )
    batch = _object(physical[0], name="physical B136 batch")
    cell_ids = [row["cell_id"] for row in cells]
    if (
        batch.get("batch_id") != "batch-0000"
        or batch.get("cell_ids") != cell_ids
        or len(batch.get("cell_ids", [])) != _B136_CELL_COUNT
    ):
        raise LongHorizonEvidenceMaterializationError(
            "physical batch must contain exactly the ordered B136 cell set"
        )
    execution = _object(plan.get("execution"), name="execution")
    if (
        execution.get("execution_route_identity") != "tick_native_execution_route_v1"
        or execution.get("total_ticks") != _B136_TOTAL_TICKS
        or execution.get("start_next_tick") != 0
        or execution.get("checkpoint_ticks") != [_B136_RESTORE_TICK, _B136_TOTAL_TICKS]
        or execution.get("acs_checkpoint_slots") != 2
        or execution.get("acs_capture_instrumentation") is not True
        or execution.get("native_streamed_checkpoint") is not True
        or execution.get("checkpoint_storage_mode") != "disk"
        or execution.get("time_authority") != "absolute_tick_v1"
    ):
        raise LongHorizonEvidenceMaterializationError(
            "formal execution must bind the exact ACS2 absolute-tick contract"
        )
    control = _object(execution.get("tick_native_control"), name="tick-native control")
    if (
        control.get("total_ticks") != _B136_TOTAL_TICKS
        or control.get("checkpoint_ticks") != [_B136_RESTORE_TICK, _B136_TOTAL_TICKS]
        or control.get("resident_owner_span_ticks") != 1120
        or control.get("native_dispatch_window_ticks") != 1120
        or control.get("source_window_ticks") != 160
        or control.get("event_window_capacity") != 41_804
        or control.get("time_authority") != "absolute_tick_v1"
    ):
        raise LongHorizonEvidenceMaterializationError(
            "formal tick-native control is not the frozen Q6 contract"
        )
    estimate = _object(plan.get("resource_estimate"), name="resource estimate")
    if (
        estimate.get("compiled_batch_count") != 1
        or estimate.get("compiled_logical_cells") != _B136_CELL_COUNT
        or estimate.get("compiled_partition") is not False
        or estimate.get("compiled_physical_batch_cells_max") != _B136_CELL_COUNT
        or estimate.get("compiled_physical_batch_cells_min") != _B136_CELL_COUNT
        or estimate.get("compiled_physical_batch_cells_total") != _B136_CELL_COUNT
        or estimate.get("physical_batch_cells") != _B136_CELL_COUNT
        or estimate.get("requested_total_ticks") != _B136_TOTAL_TICKS
        or estimate.get("resident_event_capacity") != 41_804
    ):
        raise LongHorizonEvidenceMaterializationError(
            "compiled resource estimate is not one physical B136 batch"
        )
    for row in cells:
        parameters = _object(row.get("parameters"), name="cell parameters")
        if {
            "patch.grid_size": parameters.get("patch.grid_size"),
            "patch.initial_population": parameters.get("patch.initial_population"),
            "patch.capacity": parameters.get("patch.capacity"),
            "cuda_architecture_b.cpu_request_staging_enable": parameters.get(
                "cuda_architecture_b.cpu_request_staging_enable"
            ),
            "tw_rngv2.semantics_version": parameters.get("tw_rngv2.semantics_version"),
        } != {
            "patch.grid_size": _B136_WIDTH,
            "patch.initial_population": _B136_POPULATION_CAPACITY,
            "patch.capacity": _B136_CONTROLLER_CAPACITY,
            "cuda_architecture_b.cpu_request_staging_enable": False,
            "tw_rngv2.semantics_version": 2,
        }:
            raise LongHorizonEvidenceMaterializationError(
                "formal cell parameters are not W32/P256/C512 with native RNG/no staging"
            )


def _validated_launchpad_artifacts(
    *,
    root: Path,
    plan: Mapping[str, Any],
    result: Mapping[str, Any],
    source: Mapping[str, Any],
    receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the canonical Launchpad request/result/passport bindings."""

    request_receipt = _read_object(
        root / "execution_request.json", name="execution request receipt"
    )
    if request_receipt.get("schema_version") != "eiel_execution_request_receipt_v2":
        raise LongHorizonEvidenceMaterializationError(
            "execution request receipt schema is not the canonical Launchpad V2 schema"
        )
    request_payload = _object(
        request_receipt.get("request"), name="execution request payload"
    )
    if request_payload.get("schema_version") != "eiel_execution_request_v2":
        raise LongHorizonEvidenceMaterializationError(
            "execution request payload schema is not the canonical V2 schema"
        )
    request_sha256 = _sha256_text(
        request_receipt.get("request_sha256"), name="execution request SHA-256"
    )
    request_core = dict(request_payload)
    declared_request_sha256 = request_core.pop("request_sha256", None)
    if declared_request_sha256 != request_sha256 or request_sha256 != canonical_sha256(
        request_core
    ):
        raise LongHorizonEvidenceMaterializationError(
            "execution request SHA-256 differs from its canonical payload"
        )
    request_created_at = _created_at(
        request_receipt.get("created_at_utc"),
        name="execution request created_at_utc",
    )
    result_created_at = _created_at(
        result.get("created_at_utc"),
        name="execution result created_at_utc",
    )
    if result_created_at < request_created_at:
        raise LongHorizonEvidenceMaterializationError(
            "execution result created_at_utc precedes the execution request"
        )
    request_baseline = _object(
        request_payload.get("baseline"), name="execution request baseline"
    )
    if (
        request_payload.get("purpose") != "qualification"
        or request_baseline.get("schema_version") != "eiel_baseline_declaration_v1"
        or request_baseline.get("intent") != "qualification"
        or request_baseline.get("profile_id") != "EIEL_PRODUCTION_BASELINE_V1"
        or request_baseline.get("deviations") != []
        or request_payload.get("campaign_spec") != source.get("campaign_spec_path")
        or request_payload.get("output_root") != str(root)
    ):
        raise LongHorizonEvidenceMaterializationError(
            "execution request is not the fresh qualification request for this run root"
        )
    if (
        request_receipt.get("bound_plan_sha256") != plan.get("plan_sha256")
        or request_receipt.get("run_root") != str(root)
        or request_receipt.get("request_sha256") != request_sha256
    ):
        raise LongHorizonEvidenceMaterializationError(
            "execution request receipt does not bind its plan or run root"
        )
    request_source = _object(
        request_receipt.get("source_identity"), name="execution request source identity"
    )
    if (
        request_source.get("bound_plan_sha256") != plan.get("plan_sha256")
        or request_source.get("campaign_spec_path") != source.get("campaign_spec_path")
        or request_source.get("campaign_spec_sha256")
        != source.get("campaign_spec_sha256")
        or _validated_git_identity(request_source) != source.get("git")
    ):
        raise LongHorizonEvidenceMaterializationError(
            "execution request source identity differs from the qualification result"
        )
    if (
        result.get("schema_version") != "eiel_execution_result_v2"
        or result.get("run_root") != str(root)
        or result.get("request_sha256") != request_sha256
    ):
        raise LongHorizonEvidenceMaterializationError(
            "execution result does not bind the canonical request and run root"
        )
    result_source = _object(
        result.get("source_identity"), name="execution result source identity"
    )
    if (
        result_source.get("bound_plan_sha256") != plan.get("plan_sha256")
        or result_source.get("campaign_spec_path") != source.get("campaign_spec_path")
        or result_source.get("campaign_spec_sha256")
        != source.get("campaign_spec_sha256")
        or _validated_git_identity(result_source) != source.get("git")
    ):
        raise LongHorizonEvidenceMaterializationError(
            "execution result source identity differs from the qualification source"
        )
    spec_path = Path(str(source.get("campaign_spec_path", "")))
    if not spec_path.is_absolute():
        raise LongHorizonEvidenceMaterializationError(
            "campaign specification path is not absolute"
        )
    if _file_sha256(spec_path, name="campaign specification") != source.get(
        "campaign_spec_sha256"
    ):
        raise LongHorizonEvidenceMaterializationError(
            "campaign specification SHA-256 differs from the frozen source identity"
        )

    passport_path = root / "run_passport.json"
    if receipt.get("run_passport_path") != str(passport_path):
        raise LongHorizonEvidenceMaterializationError(
            "execution receipt does not bind the canonical run passport path"
        )
    passport = _read_object(passport_path, name="run passport")
    if passport.get("schema_version") != "eiel_run_passport_v1":
        raise LongHorizonEvidenceMaterializationError(
            "run passport schema is not the canonical Launchpad schema"
        )
    passport_sha256 = _sha256_text(
        passport.get("passport_sha256"), name="run passport SHA-256"
    )
    passport_core = dict(passport)
    supplied_passport_sha256 = passport_core.pop("passport_sha256", None)
    if (
        supplied_passport_sha256 != passport_sha256
        or passport_sha256 != canonical_sha256(passport_core)
    ):
        raise LongHorizonEvidenceMaterializationError(
            "run passport SHA-256 differs from its canonical payload"
        )
    provider_identity = _object(
        passport.get("provider_identity"), name="run passport provider identity"
    )
    lifecycle = _object(plan.get("artifact_lifecycle"), name="artifact lifecycle")
    lifecycle_path = root / "artifact_lifecycle_contract.json"
    lifecycle_file = _read_object(lifecycle_path, name="artifact lifecycle contract")
    lifecycle_core = dict(lifecycle_file)
    lifecycle_sha256 = _sha256_text(
        lifecycle_core.pop("contract_sha256", None),
        name="artifact lifecycle contract SHA-256",
    )
    disposition_path = root / "artifact_disposition_manifest.json"
    disposition_receipt_path = root / "storage_disposition_receipt.json"
    if (
        passport.get("plan_sha256") != plan.get("plan_sha256")
        or passport.get("terminal_status") != "completed"
        or passport.get("lifecycle_profile") != lifecycle.get("lifecycle_profile")
        or lifecycle_file != lifecycle
        or lifecycle_sha256 != canonical_sha256(lifecycle_core)
        or passport.get("lifecycle_contract_sha256") != lifecycle_sha256
        or passport.get("disposition_manifest_path") != str(disposition_path)
        or passport.get("disposition_receipt_path") != str(disposition_receipt_path)
        or provider_identity.get("execution_route_sha256")
        != manifest.get("execution_route_sha256")
        or provider_identity.get("profile_id") != manifest.get("profile_id")
        or provider_identity.get("provider_id") != manifest.get("provider_id")
    ):
        raise LongHorizonEvidenceMaterializationError(
            "run passport is not an exact completed QUALIFICATION route binding"
        )
    disposition = _read_object(disposition_path, name="artifact disposition manifest")
    disposition_core = dict(disposition)
    disposition_sha256 = _sha256_text(
        disposition_core.pop("manifest_sha256", None),
        name="artifact disposition manifest SHA-256",
    )
    storage_receipt = _read_object(
        disposition_receipt_path, name="storage disposition receipt"
    )
    storage_core = dict(storage_receipt)
    storage_sha256 = _sha256_text(
        storage_core.pop("receipt_sha256", None),
        name="storage disposition receipt SHA-256",
    )
    if (
        disposition.get("schema_version") != "eiel_artifact_disposition_manifest_v1"
        or disposition_sha256 != canonical_sha256(disposition_core)
        or disposition.get("plan_sha256") != plan.get("plan_sha256")
        or disposition.get("lifecycle_contract_sha256") != lifecycle_sha256
        or disposition.get("lifecycle_profile") != "QUALIFICATION"
        or disposition.get("terminal_status") != "completed"
        or disposition.get("runtime_root") != str(root)
        or storage_receipt.get("schema_version")
        != "eiel_storage_disposition_receipt_v1"
        or storage_sha256 != canonical_sha256(storage_core)
        or storage_receipt.get("plan_sha256") != plan.get("plan_sha256")
        or storage_receipt.get("lifecycle_contract_sha256") != lifecycle_sha256
        or storage_receipt.get("manifest_sha256") != disposition_sha256
    ):
        raise LongHorizonEvidenceMaterializationError(
            "QUALIFICATION lifecycle closeout artifacts are not exact"
        )
    process_isolation = request_payload.get("restore_process_isolation", "inline")
    if process_isolation not in {"inline", "fresh_invocation_per_restore"}:
        raise LongHorizonEvidenceMaterializationError(
            "execution request process isolation mode is unsupported"
        )
    process_instance_ids: list[str] = []
    if process_isolation == "fresh_invocation_per_restore":
        transitions = receipt.get("restore_process_transitions_completed")
        if not isinstance(transitions, list) or len(transitions) != 1:
            raise LongHorizonEvidenceMaterializationError(
                "fresh restore execution lacks one exact process transition"
            )
        transition = _object(transitions[0], name="restore process transition")
        event = _object(
            transition.get("event"), name="restore process transition event"
        )
        producer = _object(
            transition.get("producer_process"), name="restore producer process"
        )
        restorer = _object(
            transition.get("restore_process"), name="restore consumer process"
        )
        process_instance_ids = [
            str(producer.get("process_instance_id") or ""),
            str(restorer.get("process_instance_id") or ""),
        ]
        if (
            transition.get("schema_version")
            != "eiel_execution_restore_process_transition_v1"
            or event.get("restore_tick") != _B136_RESTORE_TICK
            or producer.get("schema_version") != "eiel_execution_process_identity_v1"
            or restorer.get("schema_version") != "eiel_execution_process_identity_v1"
            or any(
                re.fullmatch(r"[0-9a-f]{32}", item) is None
                for item in process_instance_ids
            )
            or process_instance_ids[0] == process_instance_ids[1]
        ):
            raise LongHorizonEvidenceMaterializationError(
                "fresh restore did not use distinct canonical process identities"
            )
    process_binding = {
        "restore_process_isolation": process_isolation,
        "execution_request_sha256": request_sha256,
        "execution_result_sha256": _sha256_text(
            result.get("result_sha256"), name="execution result SHA-256"
        ),
        "run_passport_sha256": passport_sha256,
        "process_instance_ids": process_instance_ids,
    }
    return {
        "request": request_receipt,
        "request_payload": request_payload,
        "passport": passport,
        "request_sha256": request_sha256,
        "passport_sha256": passport_sha256,
        "source_spec_sha256": source.get("campaign_spec_sha256"),
        "source_spec_path": str(spec_path),
        "lifecycle_contract_sha256": lifecycle_sha256,
        "disposition_manifest_sha256": disposition_sha256,
        "storage_disposition_receipt_sha256": storage_sha256,
        "process_isolation": process_isolation,
        "process_identity_sha256": canonical_sha256(process_binding),
        "process_instance_ids": process_instance_ids,
        "request_created_at_utc": request_created_at.isoformat(),
        "result_created_at_utc": result_created_at.isoformat(),
        "lifecycle_contract": lifecycle_file,
        "disposition": disposition,
    }


def _inventory_path(
    raw_path: Any, *, root: Path, name: str, require_exists: bool = True
) -> tuple[Path, str]:
    if not isinstance(raw_path, str) or not raw_path:
        raise LongHorizonEvidenceMaterializationError(f"{name} path is malformed")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    if candidate.is_symlink():
        raise LongHorizonEvidenceMaterializationError(
            f"{name} path is a symlink"
        )
    try:
        resolved = candidate.resolve(strict=require_exists)
        relative = resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise LongHorizonEvidenceMaterializationError(
            f"{name} path is not an existing in-root object"
        ) from exc
    return resolved, relative.as_posix()


def _stat_binding(path: Path, *, row: Mapping[str, Any], name: str) -> None:
    sealed = _object(row.get("producer_sealed_stat"), name=f"{name} sealed stat")
    required = ("device", "inode", "bytes", "mtime_ns", "ctime_ns")
    if any(type(sealed.get(field)) is not int or sealed[field] < 0 for field in required):
        raise LongHorizonEvidenceMaterializationError(
            f"{name} producer-sealed stat is incomplete"
        )
    try:
        observed = path.stat()
    except OSError as exc:
        raise LongHorizonEvidenceMaterializationError(
            f"{name} path cannot be stat'ed"
        ) from exc
    actual = {
        "device": int(observed.st_dev),
        "inode": int(observed.st_ino),
        "bytes": int(observed.st_size),
        "mtime_ns": int(observed.st_mtime_ns),
        "ctime_ns": int(observed.st_ctime_ns),
    }
    if sealed != actual:
        raise LongHorizonEvidenceMaterializationError(
            f"{name} producer-sealed stat differs from the live object"
        )


def _validate_lifecycle_checkpoint_inventory(
    *,
    root: Path,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    batches: Sequence[Mapping[str, Any]],
    launchpad: Mapping[str, Any],
) -> None:
    disposition = _object(
        launchpad.get("disposition"), name="artifact disposition manifest"
    )
    inventory = _object(
        disposition.get("inventory"), name="lifecycle inventory disposition"
    )
    if inventory.get("mode") != "producer_declared" or inventory.get("complete") is not True:
        raise LongHorizonEvidenceMaterializationError(
            "qualification lifecycle inventory is not complete producer_declared"
        )
    lifecycle_rows = disposition.get("artifacts")
    if not isinstance(lifecycle_rows, list):
        raise LongHorizonEvidenceMaterializationError(
            "qualification lifecycle manifest lacks artifact rows"
        )
    lifecycle_by_path: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(lifecycle_rows):
        row = _object(raw, name=f"lifecycle artifact row {index}")
        if not row.get("path"):
            continue
        _, relative = _inventory_path(
            row.get("path"),
            root=root,
            name=f"lifecycle artifact row {index}",
            require_exists=False,
        )
        if relative in lifecycle_by_path:
            raise LongHorizonEvidenceMaterializationError(
                "qualification lifecycle artifact paths are duplicated"
            )
        lifecycle_by_path[relative] = row

    retention = _object(
        disposition.get("retention"), name="lifecycle retention disposition"
    )
    hold_fields = ("audit_hold_status", "qualification_hold_status")
    expected_holds = {
        field: retention.get(field)
        for field in hold_fields
    }
    if any(not isinstance(value, str) or not value for value in expected_holds.values()):
        raise LongHorizonEvidenceMaterializationError(
            "qualification lifecycle retention/hold semantics are incomplete"
        )

    contract = _object(plan.get("artifact_lifecycle"), name="artifact lifecycle")
    checkpoints = _object(contract.get("checkpoints"), name="lifecycle checkpoints")
    if checkpoints.get("authority") != "absolute_tick":
        raise LongHorizonEvidenceMaterializationError(
            "qualification lifecycle checkpoint authority is not absolute-tick"
        )
    roles = checkpoints.get("roles")
    if not isinstance(roles, list):
        raise LongHorizonEvidenceMaterializationError(
            "qualification lifecycle checkpoint roles are absent"
        )
    roles_by_tick: dict[int, dict[str, Any]] = {}
    for raw in roles:
        role = _object(raw, name="lifecycle checkpoint role")
        tick = role.get("checkpoint_tick")
        if type(tick) is not int or tick in roles_by_tick:
            raise LongHorizonEvidenceMaterializationError(
                "qualification lifecycle checkpoint roles are malformed"
            )
        roles_by_tick[tick] = role

    checkpoint_rows: dict[int, list[tuple[dict[str, Any], dict[str, Any], str]]] = {}
    expected_batch_ids = {
        str(batch.get("batch_id")) for batch in batches
    }
    for batch in batches:
        batch_id = batch.get("batch_id")
        if not isinstance(batch_id, str) or not batch_id:
            raise LongHorizonEvidenceMaterializationError(
                "checkpoint inventory batch identity is malformed"
            )
        inventory_complete = batch.get("artifact_inventory_complete")
        if inventory_complete is not None and inventory_complete is not True:
            raise LongHorizonEvidenceMaterializationError(
                "checkpoint artifact inventory is not complete producer_declared"
            )
        raw_inventory = batch.get("artifact_inventory")
        if not isinstance(raw_inventory, list):
            raise LongHorizonEvidenceMaterializationError(
                "batch manifest lacks producer-declared artifact inventory"
            )
        result = _object(batch.get("result"), name="provider batch result")
        catalog = _object(result.get("checkpoint_catalog"), name="checkpoint catalog")
        for raw in raw_inventory:
            row = _object(raw, name="producer artifact inventory row")
            association = row.get("checkpoint_run_association")
            if not isinstance(association, Mapping) or "checkpoint_tick" not in association:
                continue
            tick = association.get("checkpoint_tick")
            if type(tick) is not int:
                raise LongHorizonEvidenceMaterializationError(
                    "checkpoint artifact inventory tick is not absolute"
                )
            checkpoint = _object(
                catalog.get(str(tick)), name=f"checkpoint catalog tick {tick}"
            )
            identity = _sha256_text(
                checkpoint.get("checkpoint_identity_sha256"),
                name=f"checkpoint identity SHA-256 at tick {tick}",
            )
            if row.get("producer_checkpoint_identity_sha256") != identity:
                raise LongHorizonEvidenceMaterializationError(
                    "checkpoint artifact inventory identity differs from checkpoint catalog"
                )
            path, relative = _inventory_path(
                row.get("path"), root=root, name=f"checkpoint artifact tick {tick}"
            )
            artifact_type = row.get("artifact_type")
            if artifact_type not in {"file", "sealed_tree"}:
                raise LongHorizonEvidenceMaterializationError(
                    "checkpoint artifact inventory type is not sealed"
                )
            if artifact_type == "file" and not path.is_file():
                raise LongHorizonEvidenceMaterializationError(
                    "checkpoint artifact inventory file is absent"
                )
            if artifact_type == "sealed_tree" and not path.is_dir():
                raise LongHorizonEvidenceMaterializationError(
                    "checkpoint artifact inventory tree is absent"
                )
            _stat_binding(path, row=row, name=f"checkpoint artifact tick {tick}")
            hash_field = "sha256" if artifact_type == "file" else "tree_sha256"
            _sha256_text(
                row.get(hash_field),
                name=f"checkpoint artifact tick {tick} {hash_field}",
            )
            if type(row.get("bytes")) is not int or row["bytes"] < 0:
                raise LongHorizonEvidenceMaterializationError(
                    "checkpoint artifact inventory bytes are malformed"
                )
            if artifact_type == "file" and row["bytes"] != path.stat().st_size:
                raise LongHorizonEvidenceMaterializationError(
                    "checkpoint artifact inventory bytes differ from the file stat"
                )
            if row.get("exact_object_identity") not in {None, True}:
                raise LongHorizonEvidenceMaterializationError(
                    "checkpoint artifact inventory exact_object_identity is false"
                )
            role = roles_by_tick.get(tick)
            if role is None:
                raise LongHorizonEvidenceMaterializationError(
                    f"checkpoint lifecycle role is absent at tick {tick}"
                )
            if (
                association.get("campaign_id") != plan.get("campaign_id")
                or association.get("batch_id") != batch_id
                or row.get("checkpoint_identity") != role.get("identity")
                or row.get("checkpoint_role") != role.get("role")
                or row.get("retention_class") != role.get("retention_class")
            ):
                raise LongHorizonEvidenceMaterializationError(
                    "checkpoint artifact inventory lifecycle binding is not exact"
                )
            lifecycle_row = lifecycle_by_path.get(relative)
            if lifecycle_row is None:
                raise LongHorizonEvidenceMaterializationError(
                    "checkpoint artifact is absent from the self-hashed lifecycle manifest"
                )
            for field in (
                "artifact_type",
                "bytes",
                "retention_class",
                "checkpoint_role",
                "checkpoint_identity",
                "checkpoint_run_association",
                "exact_object_identity",
            ):
                expected = True if field == "exact_object_identity" else row.get(field)
                if lifecycle_row.get(field) != expected:
                    raise LongHorizonEvidenceMaterializationError(
                        "lifecycle manifest and batch checkpoint inventory disagree"
                    )
            if lifecycle_row.get(hash_field) != row.get(hash_field):
                raise LongHorizonEvidenceMaterializationError(
                    "lifecycle manifest and provider checkpoint seal disagree"
                )
            for field, expected in expected_holds.items():
                if lifecycle_row.get(field) != expected:
                    raise LongHorizonEvidenceMaterializationError(
                        "lifecycle manifest checkpoint retention/hold semantics differ"
                    )
                if field in row and row.get(field) != expected:
                    raise LongHorizonEvidenceMaterializationError(
                        "batch checkpoint retention/hold semantics differ"
                    )
            checkpoint_rows.setdefault(tick, []).append((row, lifecycle_row, batch_id))

    for tick in (_B136_RESTORE_TICK, _B136_TOTAL_TICKS):
        rows = checkpoint_rows.get(tick, [])
        if len(rows) != len(expected_batch_ids):
            raise LongHorizonEvidenceMaterializationError(
                f"checkpoint artifact inventory does not contain exactly one row at tick {tick}"
            )


def _validated_run(root: Path) -> dict[str, Any]:
    result = _validated_result(root)
    plan = _read_object(root / "execution_plan.json", name="execution plan")
    plan_core = dict(plan)
    supplied_plan_sha256 = plan_core.pop("plan_sha256", None)
    if supplied_plan_sha256 != canonical_sha256(plan_core):
        raise LongHorizonEvidenceMaterializationError(
            "persisted execution plan SHA-256 differs from its contents"
        )
    if plan.get("plan_sha256") != result.get("plan_sha256"):
        raise LongHorizonEvidenceMaterializationError(
            "execution result does not bind its persisted plan"
        )
    try:
        assert_tick_native_plan_payload_v1(plan)
    except TickNativeControlError as exc:
        raise LongHorizonEvidenceMaterializationError(
            f"persisted plan violates absolute tick authority: {exc}"
        ) from exc
    source = _validated_formal_source(plan=plan, result=result)
    _validate_b136_plan(plan)
    receipt = _object(result.get("execution_receipt"), name="execution receipt")
    manifest_path = Path(str(receipt.get("campaign_manifest_path", ""))).resolve()
    try:
        manifest_path.relative_to(root)
    except ValueError as exc:
        raise LongHorizonEvidenceMaterializationError(
            "campaign manifest escapes the canonical run root"
        ) from exc
    manifest = _read_object(manifest_path, name="campaign manifest")
    if (
        manifest.get("status") != "completed"
        or manifest.get("plan_sha256") != plan.get("plan_sha256")
        or manifest.get("execution_route_sha256")
        != receipt.get("execution_route_sha256")
    ):
        raise LongHorizonEvidenceMaterializationError(
            "campaign manifest is not an exact completed-plan binding"
        )
    batches: list[dict[str, Any]] = []
    for summary in manifest.get("batches", []):
        row = _object(summary, name="campaign batch summary")
        path = Path(str(row.get("manifest_path", ""))).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise LongHorizonEvidenceMaterializationError(
                "batch manifest escapes the canonical run root"
            ) from exc
        batch = _read_object(path, name="batch manifest")
        if (
            row.get("status") != "completed"
            or batch.get("status") != "completed"
            or batch.get("plan_sha256") != plan.get("plan_sha256")
            or batch.get("execution_route_sha256")
            != manifest.get("execution_route_sha256")
        ):
            raise LongHorizonEvidenceMaterializationError(
                "batch manifest is not an exact completed-route binding"
            )
        batches.append(batch)
    launchpad = _validated_launchpad_artifacts(
        root=root,
        plan=plan,
        result=result,
        source=source,
        receipt=receipt,
        manifest=manifest,
    )
    physical = plan.get("physical_batches")
    cells = plan.get("cells")
    if (
        not isinstance(physical, list)
        or not isinstance(cells, list)
        or len(batches) != len(physical)
        or len(cells) != _B136_CELL_COUNT
        or len(batches) != 1
        or any(len(batch.get("cell_ids", [])) != _B136_CELL_COUNT for batch in batches)
    ):
        raise LongHorizonEvidenceMaterializationError(
            "long-horizon evidence requires one exact physical B136 batch"
        )
    expected_batches = {
        str(row.get("batch_id")): list(row.get("cell_ids", ()))
        for row in physical
        if isinstance(row, Mapping)
    }
    actual_batches = {
        str(row.get("batch_id")): list(row.get("cell_ids", ())) for row in batches
    }
    if expected_batches != actual_batches:
        raise LongHorizonEvidenceMaterializationError(
            "batch identities/cell membership differ from the bound plan"
        )
    _validate_lifecycle_checkpoint_inventory(
        root=root,
        plan=plan,
        manifest=manifest,
        batches=batches,
        launchpad=launchpad,
    )
    return {
        "root": root,
        "result": result,
        "source": source,
        "receipt": receipt,
        "plan": plan,
        "manifest": manifest,
        "batches": batches,
        "launchpad": launchpad,
    }


def _receipts(batch: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = _object(batch.get("result"), name="provider batch result")
    raw = result.get("native_execution_receipts")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise LongHorizonEvidenceMaterializationError(
            "batch lacks native execution receipts"
        )
    rows = [_object(row, name="native execution receipt") for row in raw]
    for row in rows:
        _assert_no_generation_authority(row, name="native execution receipt")
    return rows


def _resource_windows(
    batches: Sequence[Mapping[str, Any]], *, start_tick: int, total_ticks: int
) -> tuple[list[dict[str, int]], int, int, int]:
    by_span: dict[tuple[int, int], list[dict[str, int]]] = {}
    capacities: list[int] = []
    terminal_occupancies: list[int] = []
    external_counts: list[int] = []
    for batch in batches:
        expected_start = start_tick
        previous_external_count: int | None = None
        rows = _receipts(batch)
        for receipt in rows:
            measurement = _object(
                receipt.get("resource_window_measurement"),
                name="resource window measurement",
            )
            start = measurement.get("start_tick")
            end = measurement.get("end_tick")
            settlement_value = receipt.get("compact_settlement")
            if settlement_value is None:
                settlement = {}
                checkpoint_settlement = (
                    receipt.get("checkpoint_capture_mode")
                    == "detached_physical_batch_host_settlement_v1"
                    and end in {_B136_RESTORE_TICK, _B136_TOTAL_TICKS}
                    and isinstance(
                        receipt.get("event_history_prefix_references_by_cell"), list
                    )
                    and len(receipt["event_history_prefix_references_by_cell"])
                    == _B136_CELL_COUNT
                    and re.fullmatch(
                        r"[0-9a-f]{64}",
                        str(receipt.get("external_event_history_digest") or ""),
                    )
                    is not None
                )
            else:
                settlement = _object(
                    settlement_value,
                    name="compact native settlement",
                )
                checkpoint_settlement = False
            compact_settlement = (
                settlement.get("schema_version")
                == "eiel_compact_segment_settlement_v1"
                and settlement.get("start_next_tick") == start
                and settlement.get("end_next_tick") == end
                and settlement.get("owner_identity")
                == receipt.get("owner_identity")
            )
            if (
                type(start) is not int
                or type(end) is not int
                or start != expected_start
                or end <= start
                or receipt.get("time_authority") != "absolute_tick_v1"
                or receipt.get("start_next_tick") != start
                or receipt.get("end_next_tick") != end
                or receipt.get("execution_device") != "cuda"
                or receipt.get("native_owner") != "native_extension"
                or receipt.get("explicit_native_no_fallback") is not True
                or receipt.get("hidden_fallback") is not False
                or receipt.get("runtime_jit") is not False
                or receipt.get("native_host_gaussian_segment_refill_used") is not False
                or receipt.get("gaussian_segment_refill_calls_inside_tick_loop") != 0
                or receipt.get("gaussian_segment_refill_receipts") != []
                or receipt.get("intermediate_device_to_host_operations") != 0
                or receipt.get("intermediate_host_to_device_operations") != 0
                or receipt.get("gaussian_segment_device_to_host_operations_lower_bound")
                != 0
                or receipt.get("gaussian_segment_host_to_device_operations_lower_bound")
                != 0
                or type(receipt.get("owner_span_start_tick")) is not int
                or type(receipt.get("owner_span_end_tick")) is not int
                or receipt.get("owner_span_start_tick") > start
                or receipt.get("owner_span_end_tick") < end
                or re.fullmatch(
                    r"[0-9a-f]{64}", str(receipt.get("owner_identity") or "")
                )
                is None
                or not (compact_settlement or checkpoint_settlement)
            ):
                raise LongHorizonEvidenceMaterializationError(
                    "native resource windows are not exact, resident, or fallback-free"
                )
            expected_start = end
            fields = {
                name: measurement.get(name)
                for name in (
                    "peak_gpu_allocated_bytes",
                    "peak_gpu_reserved_bytes",
                    "minimum_free_gpu_bytes",
                    "host_peak_rss_bytes",
                    "host_terminal_rss_bytes",
                    "resident_event_bytes",
                    "resident_source_bytes",
                    "event_ring_peak_occupancy",
                    "terminal_event_ring_occupancy",
                    "events_emitted",
                    "events_drained",
                )
            }
            external_history_count = measurement.get("external_event_history_count")
            if any(type(value) is not int or value < 0 for value in fields.values()) or (
                type(external_history_count) is not int or external_history_count < 0
            ) or (
                type(measurement.get("event_ring_capacity")) is not int
                or measurement["event_ring_capacity"] < 0
            ):
                raise LongHorizonEvidenceMaterializationError(
                    "resource window contains an invalid measurement"
                )
            if (
                fields["peak_gpu_allocated_bytes"] > fields["peak_gpu_reserved_bytes"]
                or fields["event_ring_peak_occupancy"]
                > measurement.get("event_ring_capacity")
                or fields["terminal_event_ring_occupancy"]
                > measurement.get("event_ring_capacity")
                or fields["events_emitted"]
                != fields["events_drained"]
                + fields["terminal_event_ring_occupancy"]
            ):
                raise LongHorizonEvidenceMaterializationError(
                    "resource window violates allocator, terminal occupancy, or event accounting bounds"
                )
            if (
                measurement.get("event_ring_capacity") != _B136_EVENT_RING_CAPACITY
                or measurement.get("resident_event_bytes") != _B136_RESIDENT_EVENT_BYTES
            ):
                raise LongHorizonEvidenceMaterializationError(
                    "physical B136 resident event capacity/bytes are not exact"
                )
            if previous_external_count is None:
                if start_tick == 0 and external_history_count != fields["events_drained"]:
                    raise LongHorizonEvidenceMaterializationError(
                        "uninterrupted external event history does not start from zero"
                    )
                if start_tick > 0 and external_history_count < fields["events_drained"]:
                    raise LongHorizonEvidenceMaterializationError(
                        "restored external event history lacks its allowed initial prefix"
                    )
            elif external_history_count != previous_external_count + fields["events_drained"]:
                raise LongHorizonEvidenceMaterializationError(
                    "external event history does not advance by events drained"
                )
            previous_external_count = external_history_count
            by_span.setdefault((start, end), []).append(fields)  # type: ignore[arg-type]
        if expected_start != total_ticks:
            raise LongHorizonEvidenceMaterializationError(
                "native resource windows do not reach the exact terminal tick"
            )
        final_measurement = _object(
            rows[-1].get("resource_window_measurement"),
            name="terminal resource window measurement",
        )
        for values, field in (
            (capacities, "event_ring_capacity"),
            (terminal_occupancies, "terminal_event_ring_occupancy"),
            (external_counts, "external_event_history_count"),
        ):
            value = final_measurement.get(field)
            if type(value) is not int or value < 0:
                raise LongHorizonEvidenceMaterializationError(
                    f"terminal resource measurement lacks {field}"
                )
            values.append(value)
    windows: list[dict[str, int]] = []
    expected_start = start_tick
    for (start, end), rows in sorted(by_span.items()):
        if start != expected_start or len(rows) != len(batches):
            raise LongHorizonEvidenceMaterializationError(
                "physical batches do not share one exact owner-window schedule"
            )
        windows.append(
            {
                "start_tick": start,
                "end_tick": end,
                "peak_gpu_allocated_bytes": max(
                    row["peak_gpu_allocated_bytes"] for row in rows
                ),
                "peak_gpu_reserved_bytes": max(
                    row["peak_gpu_reserved_bytes"] for row in rows
                ),
                "minimum_free_gpu_bytes": min(
                    row["minimum_free_gpu_bytes"] for row in rows
                ),
                "host_peak_rss_bytes": max(row["host_peak_rss_bytes"] for row in rows),
                "host_terminal_rss_bytes": max(
                    row["host_terminal_rss_bytes"] for row in rows
                ),
                "resident_event_bytes": max(
                    row["resident_event_bytes"] for row in rows
                ),
                "resident_source_bytes": max(
                    row["resident_source_bytes"] for row in rows
                ),
                # Event accounting is logical across all sequential physical batches.
                "event_ring_peak_occupancy": sum(
                    row["event_ring_peak_occupancy"] for row in rows
                ),
                "events_emitted": sum(row["events_emitted"] for row in rows),
                "events_drained": sum(row["events_drained"] for row in rows),
            }
        )
        expected_start = end
    if len(windows) < 2:
        raise LongHorizonEvidenceMaterializationError(
            "resource evidence must cross more than one measured owner window"
        )
    return (
        windows,
        sum(capacities),
        sum(terminal_occupancies),
        sum(external_counts),
    )


def _validated_checkpoint_identity(
    checkpoint: Mapping[str, Any], *, checkpoint_tick: int
) -> str:
    identity = _object(
        checkpoint.get("checkpoint_identity_contract"),
        name=f"checkpoint identity contract at tick {checkpoint_tick}",
    )
    # The public checkpoint identity is absolute-tick authoritative. Detached
    # staging may carry a private legacy-ABI payload used only to serialize the
    # physical aggregate; it is not part of checkpoint identity or restore
    # cursor authority.
    _assert_no_generation_authority(
        identity, name=f"checkpoint identity at tick {checkpoint_tick}"
    )
    canonical_identity = dict(identity)
    canonical_identity.pop("legacy_compatibility", None)
    supplied = _sha256_text(
        checkpoint.get("checkpoint_identity_sha256"),
        name=f"checkpoint identity SHA-256 at tick {checkpoint_tick}",
    )
    digest_maps = (
        "source_import_projection_digest_sha256s",
        "parent_checkpoint_projection_digest_sha256s",
        "terminal_projection_digest_sha256s",
        "canonical_terminal_state_digest_sha256s",
        "canonical_checkpoint_storage_digest_sha256s",
    )
    if (
        checkpoint.get("schema_version") != "eiel_production_engine_checkpoint_set_v1"
        or checkpoint.get("time_authority") != "absolute_tick_v1"
        or checkpoint.get("absolute_tick") != checkpoint_tick
        or checkpoint.get("next_absolute_tick") != checkpoint_tick
        or identity.get("time_authority") != "absolute_tick_v1"
        or identity.get("absolute_tick") != checkpoint_tick
        or identity.get("next_absolute_tick") != checkpoint_tick
        or any(identity.get(name) != checkpoint.get(name) for name in digest_maps)
        or checkpoint.get("mechanics_checkpoint_digests")
        != checkpoint.get("canonical_terminal_state_digest_sha256s")
        or checkpoint.get("canonical_checkpoint_storage_digests")
        != checkpoint.get("canonical_checkpoint_storage_digest_sha256s")
        or supplied != canonical_sha256(canonical_identity)
    ):
        raise LongHorizonEvidenceMaterializationError(
            f"tick-{checkpoint_tick} checkpoint identity contract is not exact"
        )
    return supplied


def _aggregate_terminal_digests(
    batches: Sequence[Mapping[str, Any]], *, expected_cell_ids: Sequence[str]
) -> tuple[str, str, str]:
    mechanics: dict[str, str] = {}
    storage: dict[str, str] = {}
    identities: list[str] = []
    expected = set(expected_cell_ids)
    for batch in batches:
        checkpoint = _object(batch.get("checkpoint"), name="terminal checkpoint")
        if (
            checkpoint.get("time_authority") != "absolute_tick_v1"
            or checkpoint.get("absolute_tick") != _B136_TOTAL_TICKS
            or checkpoint.get("next_absolute_tick") != _B136_TOTAL_TICKS
        ):
            raise LongHorizonEvidenceMaterializationError(
                "terminal checkpoint cursor is not exact absolute tick 5000"
            )
        declared_cells = batch.get("cell_ids")
        if not isinstance(declared_cells, list) or set(declared_cells) != expected:
            raise LongHorizonEvidenceMaterializationError(
                "terminal checkpoint batch cell coverage differs from B136"
            )
        for field, target, label in (
            (
                "mechanics_checkpoint_digests",
                mechanics,
                "canonical mechanics checkpoint digests",
            ),
            (
                "canonical_checkpoint_storage_digests",
                storage,
                "canonical checkpoint storage digests",
            ),
        ):
            digests = _object(checkpoint.get(field), name=label)
            for cell_id, digest in digests.items():
                if (
                    not isinstance(cell_id, str)
                    or not isinstance(digest, str)
                    or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                ):
                    raise LongHorizonEvidenceMaterializationError(
                        f"{label} authority is malformed"
                    )
                if cell_id in target and target[cell_id] != digest:
                    raise LongHorizonEvidenceMaterializationError(
                        f"{label} contains conflicting cell authority"
                    )
                target[cell_id] = digest
        identities.append(
            _validated_checkpoint_identity(
                checkpoint,
                checkpoint_tick=_B136_TOTAL_TICKS,
            )
        )
    if set(mechanics) != expected or set(storage) != expected:
        raise LongHorizonEvidenceMaterializationError(
            "canonical terminal digest authorities do not cover B136 exactly"
        )
    return (
        canonical_sha256(mechanics),
        canonical_sha256(storage),
        canonical_sha256(identities),
    )


def _aggregate_checkpoint_digests_at_tick(
    batches: Sequence[Mapping[str, Any]],
    *,
    checkpoint_tick: int,
    expected_cell_ids: Sequence[str],
) -> tuple[str, str, str, dict[str, str]]:
    mechanics: dict[str, str] = {}
    storage: dict[str, str] = {}
    terminal_projection: dict[str, str] = {}
    identities: list[str] = []
    expected = set(expected_cell_ids)
    for batch in batches:
        result = _object(batch.get("result"), name="provider batch result")
        catalog = _object(result.get("checkpoint_catalog"), name="checkpoint catalog")
        checkpoint = _object(
            catalog.get(str(checkpoint_tick)),
            name=f"checkpoint catalog tick {checkpoint_tick}",
        )
        if (
            checkpoint.get("time_authority") != "absolute_tick_v1"
            or checkpoint.get("absolute_tick") != checkpoint_tick
            or checkpoint.get("next_absolute_tick") != checkpoint_tick
        ):
            raise LongHorizonEvidenceMaterializationError(
                f"checkpoint catalog cursor is not exact at tick {checkpoint_tick}"
            )
        for field, target, label in (
            (
                "mechanics_checkpoint_digests",
                mechanics,
                "checkpoint mechanics digests",
            ),
            (
                "canonical_checkpoint_storage_digests",
                storage,
                "checkpoint storage digests",
            ),
            (
                "terminal_projection_digest_sha256s",
                terminal_projection,
                "checkpoint terminal projection digests",
            ),
        ):
            digests = _object(checkpoint.get(field), name=label)
            for cell_id, digest in digests.items():
                if (
                    not isinstance(cell_id, str)
                    or not isinstance(digest, str)
                    or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                    or cell_id in target
                ):
                    raise LongHorizonEvidenceMaterializationError(
                        f"{label} are malformed or duplicated"
                    )
                target[cell_id] = digest
        identities.append(
            _validated_checkpoint_identity(
                checkpoint,
                checkpoint_tick=checkpoint_tick,
            )
        )
    if (
        set(mechanics) != expected
        or set(storage) != expected
        or set(terminal_projection) != expected
    ):
        raise LongHorizonEvidenceMaterializationError(
            f"checkpoint catalog tick {checkpoint_tick} does not cover B136 exactly"
        )
    return (
        canonical_sha256(mechanics),
        canonical_sha256(storage),
        canonical_sha256(identities),
        terminal_projection,
    )


def _assert_resource_windows_within_limits(
    windows: Sequence[Mapping[str, int]], *, limits: Mapping[str, Any]
) -> None:
    def slope(field: str) -> float:
        return float(windows[-1][field] - windows[0][field]) / float(len(windows) - 1)

    if (
        max(row["peak_gpu_reserved_bytes"] for row in windows)
        > limits["gpu_budget_bytes"]
        or min(row["minimum_free_gpu_bytes"] for row in windows)
        < limits["minimum_free_gpu_bytes_required"]
        or max(row["host_peak_rss_bytes"] for row in windows)
        > limits["host_budget_bytes"]
        or slope("peak_gpu_reserved_bytes")
        > limits["maximum_gpu_memory_slope_bytes_per_window"]
        or slope("peak_gpu_allocated_bytes")
        > limits["maximum_gpu_memory_slope_bytes_per_window"]
        or slope("host_terminal_rss_bytes")
        > limits["maximum_host_memory_slope_bytes_per_window"]
        or slope("resident_event_bytes") != 0.0
        or slope("resident_source_bytes") != 0.0
        or slope("event_ring_peak_occupancy")
        > limits["maximum_event_ring_occupancy_slope_per_window"]
    ):
        raise LongHorizonEvidenceMaterializationError(
            "paired run resource windows exceed the frozen qualification limits"
        )


def _aggregate_event_digest(batches: Sequence[Mapping[str, Any]]) -> str:
    values: dict[str, str] = {}
    for batch in batches:
        rows = _receipts(batch)
        digest = rows[-1].get("external_event_history_digest")
        batch_id = batch.get("batch_id")
        if not isinstance(batch_id, str) or not isinstance(digest, str):
            raise LongHorizonEvidenceMaterializationError(
                "terminal external event-history authority is incomplete"
            )
        values[batch_id] = digest
    return canonical_sha256(values)


def _aggregate_event_frontier(
    batches: Sequence[Mapping[str, Any]],
    *,
    expected_cell_ids: Sequence[str],
    run_root: Path,
) -> tuple[str, str]:
    frontier: dict[str, dict[str, Any]] = {}
    event_digests: dict[str, str] = {}
    expected = set(expected_cell_ids)
    for batch in batches:
        cell_ids = batch.get("cell_ids")
        if not isinstance(cell_ids, list) or set(cell_ids) != expected:
            raise LongHorizonEvidenceMaterializationError(
                "event frontier batch coverage differs from B136"
            )
        rows = _receipts(batch)
        terminal = rows[-1]
        raw = terminal.get("event_history_prefix_references_by_cell")
        if not isinstance(raw, list) or len(raw) != _B136_CELL_COUNT:
            raise LongHorizonEvidenceMaterializationError(
                "terminal event-history frontier does not cover B136"
            )
        authenticated_rows: list[dict[str, Any]] = []
        for index, item in enumerate(raw):
            row = _object(item, name="event-history frontier row")
            if row.get("cell_index") != index:
                raise LongHorizonEvidenceMaterializationError(
                    "terminal event-history frontier cell order is not exact"
                )
            for field in (
                "path",
                "prefix_identity_schema",
                "prefix_sha256",
                "row_bytes",
                "row_count",
                "schema_version",
            ):
                if field not in row:
                    raise LongHorizonEvidenceMaterializationError(
                        "terminal event-history frontier row is incomplete"
                    )
            _sha256_text(
                row["prefix_sha256"], name="event-history frontier prefix_sha256"
            )
            path_value = row["path"]
            if not isinstance(path_value, str):
                raise LongHorizonEvidenceMaterializationError(
                    "terminal event-history frontier path is malformed"
                )
            path = Path(path_value)
            if not path.is_absolute() or path.is_symlink():
                raise LongHorizonEvidenceMaterializationError(
                    "terminal event-history frontier path is not an absolute non-symlink file"
                )
            try:
                path = path.resolve(strict=True)
            except OSError as exc:
                raise LongHorizonEvidenceMaterializationError(
                    "terminal event-history frontier is unreadable"
                ) from exc
            try:
                path.relative_to(run_root)
            except ValueError as exc:
                raise LongHorizonEvidenceMaterializationError(
                    "terminal event-history frontier escapes its run root"
                ) from exc
            if not path.is_file() or path.is_symlink():
                raise LongHorizonEvidenceMaterializationError(
                    "terminal event-history frontier path is not a regular file"
                )
            if (
                not isinstance(row["prefix_identity_schema"], str)
                or type(row["row_bytes"]) is not int
                or row["row_bytes"] < 0
                or type(row["row_count"]) is not int
                or row["row_count"] < 0
                or not isinstance(row["schema_version"], str)
            ):
                raise LongHorizonEvidenceMaterializationError(
                    "terminal event-history frontier row is malformed"
                )
            prefix_length = row["row_count"] * row["row_bytes"]
            try:
                if path.stat().st_size < prefix_length:
                    raise LongHorizonEvidenceMaterializationError(
                        "terminal event-history frontier prefix is truncated"
                    )
            except OSError as exc:
                raise LongHorizonEvidenceMaterializationError(
                    "terminal event-history frontier is unreadable"
                ) from exc
            try:
                authenticated = validate_event_sink_prefix_reference_v1(row)
            except EventSinkError as exc:
                raise LongHorizonEvidenceMaterializationError(
                    "terminal event-history frontier prefix authentication differs"
                ) from exc
            if authenticated != row:
                raise LongHorizonEvidenceMaterializationError(
                    "terminal event-history frontier prefix authentication differs"
                )
            authenticated_rows.append(authenticated)
            frontier[cell_ids[index]] = {
                "prefix_identity_schema": row["prefix_identity_schema"],
                "prefix_sha256": row["prefix_sha256"],
                "row_bytes": row["row_bytes"],
                "row_count": row["row_count"],
                "schema_version": row["schema_version"],
            }
        batch_id = batch.get("batch_id")
        event_digest = terminal.get("external_event_history_digest")
        measurement = _object(
            terminal.get("resource_window_measurement"),
            name="terminal resource window measurement",
        )
        if not isinstance(batch_id, str) or batch_id in event_digests:
            raise LongHorizonEvidenceMaterializationError(
                "terminal event-history frontier batch identity is malformed"
            )
        if (
            not isinstance(event_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", event_digest) is None
        ):
            raise LongHorizonEvidenceMaterializationError(
                "terminal external event-history digest is malformed"
            )
        row_total = sum(row["row_count"] for row in raw)
        if measurement.get("external_event_history_count") != row_total:
            raise LongHorizonEvidenceMaterializationError(
                "terminal event-history frontier count differs from external history"
            )
        try:
            authenticated_history_digest = event_history_digest_from_references_v1(
                authenticated_rows
            )
        except EventSinkError as exc:
            raise LongHorizonEvidenceMaterializationError(
                "terminal event-history digest cannot be derived from authenticated prefixes"
            ) from exc
        if authenticated_history_digest != event_digest:
            raise LongHorizonEvidenceMaterializationError(
                "terminal event-history digest differs from authenticated prefixes"
            )
        event_digests[batch_id] = event_digest
    if set(frontier) != expected:
        raise LongHorizonEvidenceMaterializationError(
            "terminal event-history frontier does not cover B136 exactly"
        )
    return canonical_sha256(frontier), canonical_sha256(event_digests)


def _lifecycle_qacct_authority(batches: Sequence[Mapping[str, Any]]) -> list[Any]:
    values: list[Any] | None = None
    for batch in batches:
        result = _object(batch.get("result"), name="provider batch result")
        current = result.get("lifecycle_qacct_receipts")
        if not isinstance(current, list):
            raise LongHorizonEvidenceMaterializationError(
                "provider lifecycle/QACCT receipts are absent"
            )
        if values is None:
            values = current
        elif values != current:
            raise LongHorizonEvidenceMaterializationError(
                "physical batch lifecycle/QACCT receipts disagree"
            )
    if values is None:
        raise LongHorizonEvidenceMaterializationError(
            "provider lifecycle/QACCT receipts are absent"
        )
    return values


def _route_receipt_authority(run: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _object(run.get("manifest"), name="campaign manifest")
    receipt = _object(run.get("receipt"), name="execution receipt")
    route = manifest.get("execution_route_sha256")
    if not isinstance(route, str) or re.fullmatch(r"[0-9a-f]{64}", route) is None:
        raise LongHorizonEvidenceMaterializationError(
            "campaign route receipt lacks an exact route SHA-256"
        )
    if receipt.get("execution_route_sha256") != route:
        raise LongHorizonEvidenceMaterializationError(
            "execution receipt and campaign route receipts disagree"
        )
    for batch in run.get("batches", []):
        if _object(batch, name="batch manifest").get("execution_route_sha256") != route:
            raise LongHorizonEvidenceMaterializationError(
                "batch route receipt differs from campaign route"
            )
    result = _object(run.get("result"), name="execution result")
    preflight = _object(result.get("preflight"), name="qualification preflight")
    binding = _object(
        preflight.get("execution_route"), name="qualified execution route receipt"
    )
    if (
        binding.get("authority_state") != "route_qualified"
        or binding.get("execution_route_sha256") != route
        or binding.get("missing_obligations") != []
        or binding.get("qualification_dimensions") != [BOUNDED_ROUTE_QUALIFIED]
    ):
        raise LongHorizonEvidenceMaterializationError(
            "formal qualification requires a qualified exact execution-route receipt"
        )
    covered = binding.get("covered_obligations")
    required = binding.get("required_obligations")
    if (
        not isinstance(covered, list)
        or not isinstance(required, list)
        or len(required) != 24
        or any(not isinstance(value, str) or not value for value in covered)
        or any(not isinstance(value, str) or not value for value in required)
        or covered != required
        or len(set(covered)) != 24
    ):
        raise LongHorizonEvidenceMaterializationError(
            "qualified execution-route receipt must cover exactly 24 obligations matching required_obligations"
        )
    qualification_path = binding.get("qualification_receipt_path")
    if not isinstance(qualification_path, str) or not qualification_path:
        raise LongHorizonEvidenceMaterializationError(
            "qualified execution-route receipt lacks a qualification sidecar"
        )
    sidecar = Path(qualification_path)
    if not sidecar.is_file() or sidecar.is_symlink():
        raise LongHorizonEvidenceMaterializationError(
            "qualification sidecar is not a regular file"
        )
    qualification_sha = _sha256_text(
        binding.get("qualification_receipt_sha256"),
        name="qualification_receipt_sha256",
    )
    try:
        actual_sha = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    except OSError as exc:
        raise LongHorizonEvidenceMaterializationError(
            "qualification sidecar is unreadable"
        ) from exc
    if actual_sha != qualification_sha:
        promoted = _read_object(sidecar, name="promoted qualification sidecar")
        scope = promoted.get("long_horizon_scope")
        promoted_evidence = (
            scope.get("validated_evidence") if isinstance(scope, Mapping) else None
        )
        try:
            promoted_validation = (
                validate_long_horizon_evidence_v1(promoted_evidence)
                if isinstance(promoted_evidence, Mapping)
                else None
            )
        except LongHorizonQualificationError as exc:
            raise LongHorizonEvidenceMaterializationError(
                "qualification sidecar SHA-256 differs"
            ) from exc
        if (
            promoted.get("schema_version")
            != "eiel_execution_route_qualification_v1"
            or promoted.get("authority_state") != "route_qualified"
            or promoted.get("execution_route_sha256") != route
            or promoted.get("promotion_source_receipt_sha256")
            != qualification_sha
            or set(promoted.get("qualification_dimensions", []))
            != {BOUNDED_ROUTE_QUALIFIED, LONG_HORIZON_RUNABILITY_QUALIFIED}
            or not isinstance(promoted_validation, Mapping)
            or promoted_validation.get("qualification_dimensions")
            != [LONG_HORIZON_RUNABILITY_QUALIFIED]
        ):
            raise LongHorizonEvidenceMaterializationError(
                "qualification sidecar SHA-256 differs"
            )
    return binding


def _owner_telemetry(batches: Sequence[Mapping[str, Any]], *, restored: bool) -> None:
    expected = list(
        _B136_OWNER_TRANSITIONS[2:] if restored else _B136_OWNER_TRANSITIONS
    )
    for batch in batches:
        result = _object(batch.get("result"), name="provider batch result")
        native_rows = _receipts(batch)
        owner_by_start = {
            row.get("owner_span_start_tick"): row.get("owner_identity")
            for row in native_rows
        }
        transfers = result.get("resident_owner_epoch_transfer_receipts")
        releases = result.get("resident_owner_release_receipts")
        if (
            result.get("expected_resident_owner_transition_ticks") != expected
            or result.get("observed_resident_owner_transition_ticks") != expected
            or result.get("resident_owner_live_count_terminal") != 0
            or result.get("resident_owner_max_simultaneous") != 1
            or result.get("terminal_next_absolute_tick") != _B136_TOTAL_TICKS
            or result.get("time_authority") != "absolute_tick_v1"
            or not isinstance(transfers, list)
            or len(transfers) != len(expected)
            or not isinstance(releases, list)
            or len(releases) != 1
        ):
            raise LongHorizonEvidenceMaterializationError(
                "resident-owner telemetry is not the exact Q6 transition/terminal contract"
            )
        for boundary, raw in zip(expected, transfers, strict=True):
            row = _object(raw, name="resident owner epoch transfer")
            prior_start = boundary - 1120
            if restored and boundary == 3360:
                prior_start = 2240
            if (
                row.get("schema_version") != "eiel_resident_owner_epoch_transfer_v1"
                or row.get("boundary_next_absolute_tick") != boundary
                or row.get("prior_owner_identity") != owner_by_start.get(prior_start)
                or row.get("successor_owner_identity") != owner_by_start.get(boundary)
                or row.get("successor_span_end_tick")
                != min(boundary + 1120, _B136_TOTAL_TICKS)
                or row.get("simultaneous_physical_owner_count") != 1
                or row.get("cuda_allocation_recreated") is not False
                or row.get("physical_owner_reused") is not True
            ):
                raise LongHorizonEvidenceMaterializationError(
                    "resident-owner transfer receipts contain overlap or broken lineage"
                )
        release = _object(releases[0], name="terminal owner release receipt")
        if (
            release.get("schema_version") != "eiel_owner_release_receipt_v1"
            or release.get("final_next_absolute_tick") != _B136_TOTAL_TICKS
            or release.get("owner_identity") != owner_by_start.get(4480)
            or release.get("live_alias_count") != 0
            or release.get("release_synchronization_complete") is not True
        ):
            raise LongHorizonEvidenceMaterializationError(
                "terminal resident owner release is not exact"
            )


def _aggregate_restore_checkpoint(
    batches: Sequence[Mapping[str, Any]],
    *,
    restore_tick: int,
    expected_cell_ids: Sequence[str],
    expected_terminal_projection: Mapping[str, str],
) -> tuple[str, str]:
    source: dict[str, str] = {}
    loaded: dict[str, str] = {}
    expected = set(expected_cell_ids)
    for batch in batches:
        result = _object(batch.get("result"), name="restored provider batch result")
        receipt = _object(
            result.get("fresh_owner_restore_receipt"),
            name="fresh-owner restore receipt",
        )
        if (
            receipt.get("time_authority") != "absolute_tick_v1"
            or receipt.get("source_checkpoint_absolute_tick") != restore_tick
            or receipt.get("source_checkpoint_next_absolute_tick") != restore_tick
            or receipt.get("restore_next_tick") != restore_tick
            or receipt.get("first_executed_absolute_tick") != restore_tick
            or receipt.get("first_committed_next_tick") != restore_tick + 1
            or receipt.get("off_by_one_check") != "PASS"
        ):
            raise LongHorizonEvidenceMaterializationError(
                "fresh-owner restore frontier is not exact"
            )
        source_digests = _object(
            receipt.get("source_checkpoint_cell_digests"),
            name="source checkpoint cell digests",
        )
        if source_digests != dict(expected_terminal_projection):
            raise LongHorizonEvidenceMaterializationError(
                "fresh-owner source checkpoint digests differ from terminal projection authority"
            )
        for cell_id, digest in source_digests.items():
            if (
                not isinstance(cell_id, str)
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or cell_id in source
            ):
                raise LongHorizonEvidenceMaterializationError(
                    "source checkpoint cell digests are malformed or duplicated"
                )
            source[cell_id] = digest
        reloads = receipt.get("reload_receipts")
        if not isinstance(reloads, list):
            raise LongHorizonEvidenceMaterializationError(
                "fresh-owner reload receipts are absent"
            )
        for raw in reloads:
            row = _object(raw, name="fresh-owner reload receipt")
            cell_id = row.get("cell_id")
            declared = row.get("source_projection_sha256")
            restored = row.get("loaded_projection_sha256")
            if (
                not isinstance(cell_id, str)
                or not isinstance(declared, str)
                or declared != expected_terminal_projection.get(cell_id)
                or restored != declared
            ):
                raise LongHorizonEvidenceMaterializationError(
                    "fresh-owner checkpoint reload digest differs"
                )
            if cell_id in loaded:
                raise LongHorizonEvidenceMaterializationError(
                    "fresh-owner checkpoint reload digests are duplicated"
                )
            loaded[cell_id] = restored
    if set(source) != expected or set(loaded) != expected or source != loaded:
        raise LongHorizonEvidenceMaterializationError(
            "fresh-owner checkpoint digest authority does not cover B136 exactly"
        )
    digest = canonical_sha256(source)
    return digest, canonical_sha256(loaded)


def materialize_long_horizon_evidence_v1(
    *,
    uninterrupted_run_root: str | Path,
    restored_run_root: str | Path,
    output_path: str | Path | None,
) -> dict[str, Any]:
    """Create one validated immutable evidence object from paired Launchpad runs."""

    uninterrupted_root = Path(uninterrupted_run_root).resolve()
    restored_root = Path(restored_run_root).resolve()
    if uninterrupted_root == restored_root:
        raise LongHorizonEvidenceMaterializationError(
            "formal qualification requires two distinct fresh run roots"
        )
    uninterrupted = _validated_run(uninterrupted_root)
    restored = _validated_run(restored_root)
    if uninterrupted["launchpad"].get("request_sha256") == restored["launchpad"].get(
        "request_sha256"
    ):
        raise LongHorizonEvidenceMaterializationError(
            "paired runs require distinct execution request identities"
        )
    if canonical_sha256(
        _normalized_twin_plan(uninterrupted["plan"])
    ) != canonical_sha256(_normalized_twin_plan(restored["plan"])):
        raise LongHorizonEvidenceMaterializationError(
            "uninterrupted and restored plans are not exact execution twins"
        )
    uninterrupted_route_receipt = _route_receipt_authority(uninterrupted)
    restored_route_receipt = _route_receipt_authority(restored)
    route = uninterrupted["manifest"].get("execution_route_sha256")
    if uninterrupted_route_receipt != restored_route_receipt:
        raise LongHorizonEvidenceMaterializationError(
            "paired runs use different qualified execution-route receipts"
        )
    if uninterrupted["manifest"].get("provider_id") != restored["manifest"].get(
        "provider_id"
    ) or uninterrupted["manifest"].get("profile_id") != restored["manifest"].get(
        "profile_id"
    ):
        raise LongHorizonEvidenceMaterializationError(
            "paired runs use different provider/profile route receipts"
        )
    if uninterrupted["source"]["git"] != restored["source"]["git"]:
        raise LongHorizonEvidenceMaterializationError(
            "paired runs do not share the exact committed repository/status source"
        )
    uninterrupted_output = _object(
        uninterrupted["plan"].get("output"), name="uninterrupted output"
    ).get("runtime_root")
    restored_output = _object(
        restored["plan"].get("output"), name="restored output"
    ).get("runtime_root")
    if (
        not isinstance(uninterrupted_output, str)
        or not isinstance(restored_output, str)
        or not uninterrupted_output
        or not restored_output
        or Path(uninterrupted_output).resolve() == Path(restored_output).resolve()
    ):
        raise LongHorizonEvidenceMaterializationError(
            "formal qualification twins require distinct output roots"
        )
    execution = _object(uninterrupted["plan"].get("execution"), name="execution")
    total_ticks = execution.get("total_ticks")
    if total_ticks != _B136_TOTAL_TICKS:
        raise LongHorizonEvidenceMaterializationError(
            "canonical B136 evidence must request exactly 5000 absolute ticks"
        )
    uninterrupted_restore_ticks = execution.get("deliberate_restore_ticks")
    if (
        uninterrupted_restore_ticks != []
        or execution.get("deliberate_restore_tick") is not None
    ):
        raise LongHorizonEvidenceMaterializationError(
            "uninterrupted qualification twin must not carry a restore selector"
        )
    if uninterrupted["launchpad"].get("process_isolation") != "inline":
        raise LongHorizonEvidenceMaterializationError(
            "uninterrupted qualification twin must use one inline invocation"
        )
    restore_execution = _object(
        restored["plan"].get("execution"), name="restore execution"
    )
    restore_ticks = restore_execution.get("deliberate_restore_ticks")
    if (
        restore_ticks != [_B136_RESTORE_TICK]
        or restore_execution.get("deliberate_restore_tick") != _B136_RESTORE_TICK
    ):
        raise LongHorizonEvidenceMaterializationError(
            "restored twin must declare exactly deliberate restore tick 2500"
        )
    if restored["launchpad"].get("process_isolation") != "fresh_invocation_per_restore":
        raise LongHorizonEvidenceMaterializationError(
            "restored qualification twin must use fresh invocation per restore"
        )
    restore_tick = _B136_RESTORE_TICK
    expected_cell_ids = [row["cell_id"] for row in uninterrupted["plan"]["cells"]]
    _owner_telemetry(uninterrupted["batches"], restored=False)
    _owner_telemetry(restored["batches"], restored=True)
    windows, capacity, terminal_occupancy, external_count = _resource_windows(
        uninterrupted["batches"], start_tick=0, total_ticks=total_ticks
    )
    restored_windows, *_ = _resource_windows(
        restored["batches"], start_tick=restore_tick, total_ticks=total_ticks
    )
    (
        uninterrupted_checkpoint,
        uninterrupted_checkpoint_storage,
        uninterrupted_checkpoint_identity,
        uninterrupted_checkpoint_projection,
    ) = _aggregate_checkpoint_digests_at_tick(
        uninterrupted["batches"],
        checkpoint_tick=restore_tick,
        expected_cell_ids=expected_cell_ids,
    )
    (
        restored_checkpoint,
        restored_checkpoint_storage,
        restored_checkpoint_identity,
        restored_checkpoint_projection,
    ) = _aggregate_checkpoint_digests_at_tick(
        restored["batches"],
        checkpoint_tick=restore_tick,
        expected_cell_ids=expected_cell_ids,
    )
    if (
        uninterrupted_checkpoint != restored_checkpoint
        or uninterrupted_checkpoint_storage != restored_checkpoint_storage
    ):
        raise LongHorizonEvidenceMaterializationError(
            "uninterrupted and restored tick-2500 mechanics/storage authorities differ"
        )
    uninterrupted_terminal, uninterrupted_storage, uninterrupted_terminal_identity = (
        _aggregate_terminal_digests(
            uninterrupted["batches"], expected_cell_ids=expected_cell_ids
        )
    )
    restored_terminal, restored_storage, restored_terminal_identity = (
        _aggregate_terminal_digests(
            restored["batches"], expected_cell_ids=expected_cell_ids
        )
    )
    uninterrupted_event = _aggregate_event_digest(uninterrupted["batches"])
    restored_event = _aggregate_event_digest(restored["batches"])
    uninterrupted_frontier, uninterrupted_frontier_event = _aggregate_event_frontier(
        uninterrupted["batches"],
        expected_cell_ids=expected_cell_ids,
        run_root=uninterrupted["root"],
    )
    restored_frontier, restored_frontier_event = _aggregate_event_frontier(
        restored["batches"],
        expected_cell_ids=expected_cell_ids,
        run_root=restored["root"],
    )
    uninterrupted_qacct = _lifecycle_qacct_authority(uninterrupted["batches"])
    restored_qacct = _lifecycle_qacct_authority(restored["batches"])
    if (
        uninterrupted_terminal != restored_terminal
        or uninterrupted_storage != restored_storage
        or uninterrupted_event != restored_event
        or uninterrupted_frontier_event != uninterrupted_event
        or restored_frontier_event != restored_event
        or uninterrupted_frontier != restored_frontier
        or uninterrupted_qacct != restored_qacct
    ):
        raise LongHorizonEvidenceMaterializationError(
            "uninterrupted and restored canonical terminal/event/lifecycle/QACCT authorities differ"
        )
    checkpoint_digest, loaded_checkpoint_digest = _aggregate_restore_checkpoint(
        restored["batches"],
        restore_tick=restore_tick,
        expected_cell_ids=expected_cell_ids,
        expected_terminal_projection=restored_checkpoint_projection,
    )
    artifacts = uninterrupted["plan"].get("artifacts")
    artifact_sha256s = {
        str(name): str(row.get("sha256"))
        for name, row in (artifacts.items() if isinstance(artifacts, Mapping) else ())
        if isinstance(row, Mapping) and isinstance(row.get("sha256"), str)
    }
    git_identity = uninterrupted["source"]["git"]
    budgets = _object(execution.get("budgets"), name="execution budgets")
    limits = {
        "gpu_budget_bytes": budgets.get("vram_bytes"),
        "host_budget_bytes": budgets.get("host_bytes"),
        "minimum_free_gpu_bytes_required": budgets.get("minimum_free_gpu_bytes"),
        "maximum_gpu_memory_slope_bytes_per_window": budgets.get(
            "maximum_gpu_memory_slope_bytes_per_window"
        ),
        "maximum_host_memory_slope_bytes_per_window": budgets.get(
            "maximum_host_memory_slope_bytes_per_window"
        ),
        "maximum_event_ring_occupancy_slope_per_window": budgets.get(
            "maximum_event_ring_occupancy_slope_per_window"
        ),
    }
    if any(type(value) is not int or value < 1 for value in limits.values()):
        raise LongHorizonEvidenceMaterializationError(
            "plan lacks frozen long-horizon resource limits"
        )
    _assert_resource_windows_within_limits(windows, limits=limits)
    _assert_resource_windows_within_limits(restored_windows, limits=limits)
    vectors = {
        name: [row[name] for row in windows]
        for name in (
            "peak_gpu_allocated_bytes",
            "peak_gpu_reserved_bytes",
            "host_peak_rss_bytes",
            "host_terminal_rss_bytes",
            "resident_event_bytes",
            "resident_source_bytes",
            "event_ring_peak_occupancy",
        )
    }

    def slope(values: Sequence[int]) -> float:
        return float(values[-1] - values[0]) / float(len(values) - 1)

    def lineage_binding(
        run: Mapping[str, Any],
        route_receipt: Mapping[str, Any],
        *,
        lifecycle_qacct: Sequence[Any],
        event_frontier: str,
        terminal_mechanics: str,
        terminal_storage: str,
    ) -> dict[str, Any]:
        source = _object(run.get("source"), name="run source lineage")
        result = _object(run.get("result"), name="run result lineage")
        plan = _object(run.get("plan"), name="run plan lineage")
        launchpad = _object(run.get("launchpad"), name="run Launchpad lineage")
        return {
            "run_root": str(run["root"]),
            "plan_sha256": _sha256_text(
                plan.get("plan_sha256"), name="lineage plan_sha256"
            ),
            "result_sha256": _sha256_text(
                result.get("result_sha256"), name="lineage result_sha256"
            ),
            "request_sha256": _sha256_text(
                result.get("request_sha256"), name="lineage request_sha256"
            ),
            "run_passport_sha256": _sha256_text(
                launchpad.get("passport_sha256"),
                name="lineage run_passport_sha256",
            ),
            "lifecycle_contract_sha256": _sha256_text(
                launchpad.get("lifecycle_contract_sha256"),
                name="lineage lifecycle_contract_sha256",
            ),
            "disposition_manifest_sha256": _sha256_text(
                launchpad.get("disposition_manifest_sha256"),
                name="lineage disposition_manifest_sha256",
            ),
            "storage_disposition_receipt_sha256": _sha256_text(
                launchpad.get("storage_disposition_receipt_sha256"),
                name="lineage storage_disposition_receipt_sha256",
            ),
            "invocation_identity_sha256": _sha256_text(
                launchpad.get("process_identity_sha256"),
                name="lineage invocation_identity_sha256",
            ),
            "process_instance_ids": list(launchpad.get("process_instance_ids", ())),
            "campaign_spec_path": source.get("campaign_spec_path"),
            "campaign_spec_sha256": _sha256_text(
                source.get("campaign_spec_sha256"),
                name="lineage campaign_spec_sha256",
            ),
            "git_status_sha256": git_identity["working_tree_status_sha256"],
            "execution_route_sha256": route_receipt.get("execution_route_sha256"),
            "qualification_receipt_sha256": _sha256_text(
                route_receipt.get("qualification_receipt_sha256"),
                name="lineage qualification_receipt_sha256",
            ),
            "lifecycle_qacct_digest": canonical_sha256(list(lifecycle_qacct)),
            "event_frontier_digest": event_frontier,
            "terminal_mechanics_digest": terminal_mechanics,
            "terminal_storage_digest": terminal_storage,
        }

    body: dict[str, Any] = {
        "schema_version": LONG_HORIZON_EVIDENCE_SCHEMA_VERSION,
        "materialization_profile": "eiel_b136_5000tick_qualification_pair_v1",
        "evidence_identity": {
            "execution_route_sha256": route,
            "provider_id": uninterrupted["manifest"].get("provider_id"),
            "profile_id": uninterrupted["manifest"].get("profile_id"),
            "source_sha256": canonical_sha256({"git_commit": git_identity["commit"]}),
            "artifact_sha256s": artifact_sha256s,
            "plan_sha256": uninterrupted["plan"].get("plan_sha256"),
            "run_id": uninterrupted["plan"].get("campaign_id"),
            "run_receipt_sha256": uninterrupted["result"].get("result_sha256"),
            "route_scope": LONG_HORIZON_RUNABILITY_ROUTE_SCOPE,
            "geometry": {
                "batch_cells": 136,
                "world_width": 32,
                "population_capacity": 256,
                "controller_capacity": 512,
            },
        },
        "paired_run_bindings": {
            "git_identity": git_identity,
            "uninterrupted": lineage_binding(
                uninterrupted,
                uninterrupted_route_receipt,
                lifecycle_qacct=uninterrupted_qacct,
                event_frontier=uninterrupted_frontier,
                terminal_mechanics=uninterrupted_terminal,
                terminal_storage=uninterrupted_storage,
            ),
            "restored": lineage_binding(
                restored,
                restored_route_receipt,
                lifecycle_qacct=restored_qacct,
                event_frontier=restored_frontier,
                terminal_mechanics=restored_terminal,
                terminal_storage=restored_storage,
            ),
        },
        "resource_limits": limits,
        "ticks_requested": total_ticks,
        "ticks_committed": total_ticks,
        "owner_boundaries_crossed": len(_B136_OWNER_TRANSITIONS),
        "peak_gpu_allocated_bytes": max(vectors["peak_gpu_allocated_bytes"]),
        "peak_gpu_reserved_bytes": max(vectors["peak_gpu_reserved_bytes"]),
        "minimum_free_gpu_bytes": min(row["minimum_free_gpu_bytes"] for row in windows),
        "gpu_memory_slope_bytes_per_window": slope(vectors["peak_gpu_reserved_bytes"]),
        "gpu_allocated_memory_slope_bytes_per_window": slope(
            vectors["peak_gpu_allocated_bytes"]
        ),
        "host_peak_rss_bytes": max(vectors["host_peak_rss_bytes"]),
        "host_terminal_rss_bytes": max(vectors["host_terminal_rss_bytes"]),
        "host_memory_slope_bytes_per_window": slope(vectors["host_terminal_rss_bytes"]),
        "resident_event_bytes": max(vectors["resident_event_bytes"]),
        "resident_event_bytes_slope_per_window": slope(vectors["resident_event_bytes"]),
        "resident_source_bytes": max(vectors["resident_source_bytes"]),
        "resident_source_bytes_slope_per_window": slope(
            vectors["resident_source_bytes"]
        ),
        "event_ring_capacity": capacity,
        "event_ring_peak_occupancy": max(vectors["event_ring_peak_occupancy"]),
        "event_ring_occupancy_slope_per_window": slope(
            vectors["event_ring_peak_occupancy"]
        ),
        "events_emitted": sum(row["events_emitted"] for row in windows),
        "events_drained": sum(row["events_drained"] for row in windows),
        "terminal_event_ring_occupancy": terminal_occupancy,
        "external_event_history_count": external_count,
        "external_event_history_digest": uninterrupted_event,
        "checkpoint_frontier": {
            "next_absolute_tick": total_ticks,
            "terminal_digest": uninterrupted_terminal,
            "uninterrupted_identity_digest": uninterrupted_terminal_identity,
            "restored_identity_digest": restored_terminal_identity,
        },
        "checkpoint_restore": {
            "checkpoint_tick": restore_tick,
            "checkpoint_digest": checkpoint_digest,
            "restored_checkpoint_digest": loaded_checkpoint_digest,
            "paired_checkpoint": {
                "uninterrupted_mechanics_digest": uninterrupted_checkpoint,
                "restored_mechanics_digest": restored_checkpoint,
                "uninterrupted_storage_digest": uninterrupted_checkpoint_storage,
                "restored_storage_digest": restored_checkpoint_storage,
                "uninterrupted_identity_digest": uninterrupted_checkpoint_identity,
                "restored_identity_digest": restored_checkpoint_identity,
                "uninterrupted_terminal_projection_digest_sha256s": dict(
                    uninterrupted_checkpoint_projection
                ),
                "restored_terminal_projection_digest_sha256s": dict(
                    restored_checkpoint_projection
                ),
            },
            "fresh_owner_restore_tick": restore_tick,
            "fresh_owner_uninterrupted_terminal_digest": uninterrupted_terminal,
            "fresh_owner_restored_terminal_digest": restored_terminal,
            "fresh_owner_uninterrupted_event_digest": uninterrupted_event,
            "fresh_owner_restored_event_digest": restored_event,
        },
        "terminal_digest": uninterrupted_terminal,
        "event_digest": uninterrupted_event,
        "windows": windows,
        "restored_windows": restored_windows,
        "errors": [],
        "ooms": [],
    }
    body["paired_run_bindings_sha256"] = canonical_sha256(body["paired_run_bindings"])
    evidence = {**body, "evidence_sha256": canonical_sha256(body)}
    validate_long_horizon_evidence_v1(evidence)
    if output_path is not None:
        atomic_write_json(Path(output_path).resolve(), evidence)
    return evidence


def validate_b136_5000tick_qualification_pair_evidence_v1(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-materialize both canonical roots and reject self-authored attestations."""

    bindings = _object(evidence.get("paired_run_bindings"), name="paired run bindings")
    uninterrupted = _object(
        bindings.get("uninterrupted"), name="uninterrupted run binding"
    )
    restored = _object(bindings.get("restored"), name="restored run binding")
    regenerated = materialize_long_horizon_evidence_v1(
        uninterrupted_run_root=str(uninterrupted.get("run_root") or ""),
        restored_run_root=str(restored.get("run_root") or ""),
        output_path=None,
    )
    if regenerated != dict(evidence):
        raise LongHorizonEvidenceMaterializationError(
            "qualification-pair evidence differs from canonical re-materialization"
        )
    return regenerated


__all__ = [
    "LongHorizonEvidenceMaterializationError",
    "materialize_long_horizon_evidence_v1",
    "validate_b136_5000tick_qualification_pair_evidence_v1",
]
