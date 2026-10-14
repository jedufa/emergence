"""Single-manifest, crash-safe Adaptive Expedition operator entrypoint.

The operator is coordination above the existing Science Request front door. It
does not schedule providers, grant execution authority, infer evidence claims,
or create a mutable service. One immutable manifest resolves its content-bound
founder request, budgets, policy and lifecycle intent. The append-only
Expedition ledger remains the recovery boundary.

The internal runner now realizes both founder-basis and exact selected-world
actions.  The user-facing contract remains ``eiel expedition run MANIFEST``;
the operator does not expose child execution or continuation plumbing.
"""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
from typing import Any, Final
from urllib.parse import unquote, urlsplit

from emergence_contracts import canonical_json_bytes
from emergence_contracts.adaptive_science_expedition_v1 import (
    EvolutionaryAtlasV1,
    ExpeditionManifestV1,
    ObservationFrameV1,
    RetainedParentBankV1,
    StateClassV1,
    StateRefV1,
)
from emergence_contracts.artifacts import RunArtifactRefV1
from emergence_contracts.canonical import canonical_sha256
from emergence_contracts.science_request_v1 import (
    EielScienceRequestV1,
    validate_eiel_science_request_v1,
)

from .general_closed_loop import (
    EXACT_OBSERVATION_FRAME_ARTIFACT_KIND_V1,
    GeneralClosedLoopDependenciesV1,
    GeneralClosedLoopResultV1,
    _direct_exact_observation_projection_v1,
    run_general_closed_loop_v1,
)
from .continuation_adapter import (
    SelectedWorldBranchError,
    resolve_workhorse_selected_world_continuation_v1,
    workhorse_selected_world_authority_ref_v1,
)
from .ledger import project_ledger_to_public_journal_v1
from .ledger import reconstruct_expedition_frontier_v1
from .genealogy import build_experiment_dag_v1, validate_experiment_dag_v1
from .lifecycle import (
    ExpeditionLifecycleContextV1,
    ExpeditionLifecycleError,
    close_expedition_lifecycle_v1,
    enforce_expedition_storage_v1,
    open_expedition_lifecycle_v1,
    _preflight_expedition_outer_object_budget_v1,
    release_active_parent_packages_v1,
    release_expedition_storage_v1,
    resolve_expedition_lifecycle_policy_v1,
)
from .policy_protocol import ExpeditionPolicyV1, canonical_plain_value_v1
from .closeout import build_expedition_closeout_v1
from .review_projection import (
    build_human_review_projection_v1,
    validate_policy_review_presentation_v1,
)
from engine.perf.cuda_full_evolution_v2.workhorse.indexed_checkpoint import (
    WORKHORSE_SELECTED_WORLD_AUTHORITY_FILE_V1,
    WORKHORSE_SELECTED_WORLD_CHECKPOINT_FILE_V1,
    WORKHORSE_SELECTED_WORLD_STATE_FILE_V1,
    inspect_workhorse_selected_world_checkpoint_v1,
)

OPERATOR_START_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_adaptive_expedition_operator_start_v1"
)
OPERATOR_CLOSEOUT_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_adaptive_expedition_operator_closeout_v1"
)
DEFAULT_EXPEDITION_RUNTIME_PARENT_V1: Final[Path] = Path(
    public_runtime_path('adaptive-expedition-v1/expeditions')
)
FRONTIER_TIMING_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_adaptive_expedition_frontier_dispatch_timing_internal_v1"
)
SAFE_FRONTIER_RESERVE_BASIS_V1: Final[str] = (
    "cold_start_five_percent_capped_300_plus_two_times_max_observed_frontier_v1"
)
SAFE_FRONTIER_COLD_RESERVE_MAX_SECONDS_V1: Final[int] = 300
EXPEDITION_OPERATOR_EXECUTION_MODE_V1: Final[str] = "ADAPTIVE_EXPEDITION_OPERATOR_V1"
EXPEDITION_OPERATOR_PROGRAMME_STAGE_V1: Final[str] = "ADAPTIVE_EXPEDITION_OPERATOR_V1"
_SAFE_COMPONENT_RE: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9_.-]+")
_FRONTIER_TIMING_FILE_RE_V1: Final[re.Pattern[str]] = re.compile(r"([0-9]{8})\.json")
_FRONTIER_TIMING_TEMP_FILE_RE_V1: Final[re.Pattern[str]] = re.compile(
    r"\.([0-9]{8})\.json\.([a-z0-9_]{8})"
)
_MAX_FRONTIER_TIMING_TEMP_BYTES_V1: Final[int] = 4096


class ExpeditionOperatorError(ValueError):
    """A manifest cannot be launched without weakening a frozen boundary."""


ClockV1 = Callable[[], datetime]
ClosedLoopRunnerV1 = Callable[..., GeneralClosedLoopResultV1]


def _founder_nominal_batch_worlds_v1(
    founder_request: EielScienceRequestV1,
) -> int:
    """Return the request's nominal batch projection for stable replay.

    This value is deliberately not a physical-capacity claim.  The Science
    Request front door may partition it to the selected sealed carrier B.
    Keeping the projection in the operator start record makes child request
    generation restart-stable without freezing a hardware limit into the
    Expedition mechanism.
    """

    try:
        return founder_request.batch_plan().worlds
    except (TypeError, ValueError) as error:
        raise ExpeditionOperatorError(
            "founder Science Request has no valid nominal batch projection"
        ) from error


def _utc_now_v1() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class ExpeditionOperatorDependenciesV1:
    """Narrow injection seam for clocks and non-executing focused tests."""

    clock: ClockV1 = _utc_now_v1
    closed_loop_runner: ClosedLoopRunnerV1 = run_general_closed_loop_v1
    closed_loop_dependencies: object = GeneralClosedLoopDependenciesV1()
    # The user-facing operator is a generic mechanism.  Engineering gates may
    # inject an explicit G4 identity, but a plain manifest launch must not
    # silently claim a programme stage or historical continuation authority.
    execution_mode: str = EXPEDITION_OPERATOR_EXECUTION_MODE_V1
    programme_stage: str = EXPEDITION_OPERATOR_PROGRAMME_STAGE_V1
    historical_continuation_authority: bool = False


@dataclass(frozen=True, slots=True)
class ExpeditionOperatorResultV1:
    runtime_root: Path
    start_record: Mapping[str, object]
    loop_result: GeneralClosedLoopResultV1
    atlas_path: Path
    retained_parent_bank_path: Path
    journal_path: Path
    human_review_path: Path
    closeout_path: Path
    lifecycle_closeout: Mapping[str, object]
    experiment_dag_path: Path | None = None

    def receipt(self) -> dict[str, object]:
        terminal = self.loop_result.ledger["events"][-1]["payload"]
        return {
            "status": "EXPEDITION_TERMINAL",
            "authority_effect": "NO_NEW_EXECUTION_AUTHORITY",
            "execution_route": "existing_eiel_science_request_front_door",
            "qualification_status": "NOT_REQUESTED",
            "no_claims": True,
            "expedition_id": self.start_record["expedition_id"],
            "manifest_sha256": self.start_record["manifest_sha256"],
            "runtime_root": str(self.runtime_root),
            "terminal_status": terminal["terminal_status"],
            "frontier_count": len(self.loop_result.decisions),
            "child_count": len(self.loop_result.children),
            "atlas_sha256": self.loop_result.atlas.atlas_sha256().hex,
            "retained_parent_bank_sha256": (
                self.loop_result.retained_parent_bank.bank_sha256().hex
            ),
            "head_event_sha256": self.loop_result.ledger["head_event_sha256"],
            "atlas_path": str(self.atlas_path),
            "retained_parent_bank_path": str(self.retained_parent_bank_path),
            "journal_path": str(self.journal_path),
            "human_review_path": str(self.human_review_path),
            "closeout_path": str(self.closeout_path),
            "experiment_dag_path": (
                None if self.experiment_dag_path is None else str(self.experiment_dag_path)
            ),
            "lifecycle_closeout": dict(self.lifecycle_closeout),
        }


def _safe_component_v1(value: str) -> str:
    projected = _SAFE_COMPONENT_RE.sub("-", value).strip("-.")
    if not projected:
        raise ExpeditionOperatorError("expedition_id has no safe path projection")
    return projected[:96]


def _source_commit_v1(repo_root: str | Path | None) -> str:
    """Return the coordinator source identity bound to a new operator run."""

    root = (
        Path(repo_root).expanduser().absolute()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        # Focused injected tests may use a temporary non-Git repo_root.  The
        # coordinator source itself remains bound to this module's repository;
        # a real operator launch therefore still fails closed if that fallback
        # cannot be authenticated.
        fallback = Path(__file__).resolve().parents[2]
        try:
            value = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=fallback,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            raise ExpeditionOperatorError(
                "Expedition coordinator source commit cannot be authenticated"
            ) from None
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ExpeditionOperatorError("Expedition coordinator source commit is invalid")
    return value


def _canonical_mapping_v1(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ExpeditionOperatorError(f"{field} must be a JSON object")
    try:
        return json.loads(canonical_json_bytes(dict(value)))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ExpeditionOperatorError(f"{field} is not canonical JSON") from error


def _read_json_mapping_v1(path: Path, *, field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ExpeditionOperatorError(
            f"{field} is not a regular non-symlink file: {path}"
        )
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExpeditionOperatorError(f"{field} is unreadable JSON: {path}") from error
    return _canonical_mapping_v1(value, field=field)


def _local_file_locator_v1(locator: str) -> Path:
    parsed = urlsplit(locator)
    if (
        parsed.scheme != "file"
        or parsed.netloc not in {"", "localhost"}
        or parsed.query
        or parsed.fragment
    ):
        raise ExpeditionOperatorError(
            "founder request locator must be an absolute local file URI"
        )
    path = Path(unquote(parsed.path))
    if not path.is_absolute():
        raise ExpeditionOperatorError("founder request file URI must be absolute")
    return path


def _founder_request_ref_v1(manifest: ExpeditionManifestV1) -> RunArtifactRefV1:
    refs = tuple(
        ref
        for ref in manifest.founder_basis_refs
        if ref.artifact_kind == "eiel_science_request_v1"
    )
    if len(refs) != 1:
        raise ExpeditionOperatorError(
            "V1 operator requires exactly one founder eiel_science_request_v1 reference"
        )
    ref = refs[0]
    if str(ref.artifact_schema_version) != "eiel_science_request_v1":
        raise ExpeditionOperatorError("founder request schema identity is unsupported")
    return ref


def resolve_founder_request_v1(
    manifest: ExpeditionManifestV1,
) -> tuple[EielScienceRequestV1, Path]:
    """Resolve and authenticate the founder only for an initial launch."""

    ref = _founder_request_ref_v1(manifest)
    source = _local_file_locator_v1(ref.locator)
    request = validate_eiel_science_request_v1(
        _read_json_mapping_v1(source, field="founder Science Request")
    )
    canonical_bytes = canonical_json_bytes(request.to_dict())
    if ref.content_digest != request.request_sha256():
        raise ExpeditionOperatorError(
            "founder Science Request digest differs from the immutable manifest"
        )
    if ref.size_bytes is None or ref.size_bytes != len(canonical_bytes):
        raise ExpeditionOperatorError(
            "founder Science Request canonical size differs from the immutable manifest"
        )
    return request, source.resolve()


def _authenticate_initial_exact_observation_frame_v1(
    *,
    ref: RunArtifactRefV1,
    source: Path,
    manifest: ExpeditionManifestV1,
) -> ObservationFrameV1:
    raw = _read_json_mapping_v1(source, field="initial exact ObservationFrame")
    try:
        frame = ObservationFrameV1.from_dict(raw)
    except (KeyError, TypeError, ValueError) as error:
        raise ExpeditionOperatorError(
            "initial exact ObservationFrame contract is invalid"
        ) from error
    payload = canonical_json_bytes(frame.to_dict())
    if (
        str(ref.artifact_schema_version) != EXACT_OBSERVATION_FRAME_ARTIFACT_KIND_V1
        or ref.content_digest != frame.observation_frame_sha256()
        or ref.size_bytes != len(payload)
        or source.read_bytes() != payload
        or frame.expedition_id != manifest.expedition_id
        or frame.observation_profile_id != manifest.observation_profile_id
        or frame.observation_descriptor_set_sha256
        != manifest.observation_descriptor_set_sha256
        or frame.source_state_ref.state_class is not StateClassV1.BRANCHABLE_EXACT
    ):
        raise ExpeditionOperatorError(
            "initial exact ObservationFrame differs from its manifest binding"
        )
    projection = _direct_exact_observation_projection_v1(
        source,
        manifest=manifest,
    )
    if projection is None or projection.frames != (frame,):
        raise ExpeditionOperatorError(
            "initial exact ObservationFrame cannot reopen its selected state"
        )
    return frame


def _file_sha256_v1(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _fsync_directory_v1(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file_v1(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _materialized_state_ref_v1(
    *,
    original: StateRefV1,
    package_root: Path,
) -> StateRefV1:
    manifest_path = package_root / WORKHORSE_SELECTED_WORLD_CHECKPOINT_FILE_V1
    authority_path = package_root / WORKHORSE_SELECTED_WORLD_AUTHORITY_FILE_V1
    try:
        manifest, state_path = inspect_workhorse_selected_world_checkpoint_v1(
            manifest_path
        )
        projection = _read_json_mapping_v1(
            authority_path,
            field="operator-owned exact founder occurrence projection",
        )
    except Exception as error:
        raise ExpeditionOperatorError(
            "operator-owned exact founder package cannot reopen"
        ) from error
    occurrence_root = package_root / "occurrence"
    expected_paths = {
        str((occurrence_root / "descriptor.json").resolve()),
        str((occurrence_root / "commit.json").resolve()),
    }
    if {
        projection.get("occurrence_descriptor_path"),
        projection.get("occurrence_commit_path"),
    } != expected_paths or projection.get("occurrence_root") != str(
        occurrence_root.resolve()
    ):
        raise ExpeditionOperatorError(
            "operator-owned exact founder still depends on external occurrence files"
        )
    manifest_file_sha256 = _file_sha256_v1(manifest_path)
    state_value = original.to_dict()
    artifact = dict(state_value["source_artifact_ref"])
    artifact.update(
        {
            "artifact_id": (
                "workhorse-selected-world-" + str(manifest["manifest_body_sha256"])[:24]
            ),
            "locator": str(manifest_path.resolve()),
            "content_digest": {
                "algorithm": "sha256",
                "hex": manifest_file_sha256,
            },
            "size_bytes": sum(
                path.stat().st_size
                for path in (manifest_path, authority_path, state_path)
            ),
        }
    )
    state_value["source_artifact_ref"] = artifact
    state_value["source_object_sha256"] = artifact["content_digest"]
    try:
        materialized = StateRefV1.from_dict(state_value)
        authority_ref = workhorse_selected_world_authority_ref_v1(materialized)
        resolve_workhorse_selected_world_continuation_v1(
            state_ref=materialized,
            authority_receipt_ref=authority_ref,
        )
    except (KeyError, TypeError, ValueError, SelectedWorldBranchError) as error:
        raise ExpeditionOperatorError(
            "operator-owned exact founder package changed identity"
        ) from error
    for field in (
        "logical_world_id",
        "next_absolute_tick",
        "semantic_world_sha256",
        "control_history_sha256",
        "exact_state_sha256",
        "checkpoint_occurrence_sha256",
        "rng_state_sha256",
        "event_prefix_sha256",
        "parent_occurrence_sha256",
    ):
        if getattr(materialized, field) != getattr(original, field):
            raise ExpeditionOperatorError(
                "operator-owned exact founder changed a scientific identity"
            )
    return materialized


def _materialize_initial_exact_state_package_v1(
    *,
    original: StateRefV1,
    destination: Path,
    maximum_bytes: int,
) -> StateRefV1:
    """Relocate one exact StateRef package plus compact occurrence authority."""

    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir():
            raise ExpeditionOperatorError(
                "operator-owned exact founder destination is invalid"
            )
        return _materialized_state_ref_v1(
            original=original,
            package_root=destination,
        )
    try:
        source_authority_ref = workhorse_selected_world_authority_ref_v1(original)
        resolve_workhorse_selected_world_continuation_v1(
            state_ref=original,
            authority_receipt_ref=source_authority_ref,
        )
        source_manifest = Path(original.source_artifact_ref.locator).resolve(
            strict=True
        )
        selected, source_state = inspect_workhorse_selected_world_checkpoint_v1(
            source_manifest
        )
        source_authority = Path(source_authority_ref.locator).resolve(strict=True)
        projection = _read_json_mapping_v1(
            source_authority,
            field="source exact founder occurrence projection",
        )
        source_descriptor = Path(str(projection["occurrence_descriptor_path"])).resolve(
            strict=True
        )
        source_commit = Path(str(projection["occurrence_commit_path"])).resolve(
            strict=True
        )
    except (KeyError, OSError, SelectedWorldBranchError) as error:
        raise ExpeditionOperatorError(
            "initial exact founder source cannot be materialized independently"
        ) from error
    source_files = (
        source_manifest,
        source_authority,
        source_state,
        source_descriptor,
        source_commit,
    )
    if any(path.is_symlink() or not path.is_file() for path in source_files):
        raise ExpeditionOperatorError(
            "initial exact founder source contains an unstable file"
        )
    total_source_bytes = sum(path.stat().st_size for path in source_files)
    if total_source_bytes <= 0 or total_source_bytes > maximum_bytes:
        raise ExpeditionOperatorError(
            "initial exact founder package exceeds its outer durable byte bound"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise ExpeditionOperatorError("operator-owned exact founder parent is invalid")
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            suffix=".exact-founder.tmp",
            dir=destination.parent,
        )
    )
    try:
        occurrence_stage = stage / "occurrence"
        occurrence_stage.mkdir()
        final_occurrence = destination / "occurrence"
        final_descriptor = final_occurrence / "descriptor.json"
        final_commit = final_occurrence / "commit.json"
        copied_descriptor = occurrence_stage / "descriptor.json"
        copied_commit = occurrence_stage / "commit.json"
        shutil.copyfile(source_descriptor, copied_descriptor)
        shutil.copyfile(source_commit, copied_commit)
        copied_descriptor.chmod(0o444)
        copied_commit.chmod(0o444)
        _fsync_file_v1(copied_descriptor)
        _fsync_file_v1(copied_commit)
        _fsync_directory_v1(occurrence_stage)

        copied_state = stage / WORKHORSE_SELECTED_WORLD_STATE_FILE_V1
        shutil.copyfile(source_state, copied_state)
        copied_state.chmod(0o444)
        _fsync_file_v1(copied_state)
        if (
            _file_sha256_v1(copied_state) != _file_sha256_v1(source_state)
            or _file_sha256_v1(copied_descriptor)
            != str(projection["occurrence_descriptor_file_sha256"])
            or _file_sha256_v1(copied_commit)
            != str(projection["occurrence_commit_file_sha256"])
        ):
            raise ExpeditionOperatorError(
                "initial exact founder changed while its bytes were copied"
            )

        projection.update(
            {
                "occurrence_root": str(final_occurrence.resolve()),
                "occurrence_descriptor_path": str(final_descriptor.resolve()),
                "occurrence_commit_path": str(final_commit.resolve()),
            }
        )
        projection_core = {
            key: value
            for key, value in projection.items()
            if key != "projection_content_sha256"
        }
        projection["projection_content_sha256"] = canonical_sha256(projection_core).hex
        authority_bytes = canonical_json_bytes(projection)
        copied_authority = stage / WORKHORSE_SELECTED_WORLD_AUTHORITY_FILE_V1
        copied_authority.write_bytes(authority_bytes)
        copied_authority.chmod(0o444)
        _fsync_file_v1(copied_authority)

        selected_body = {
            key: value
            for key, value in selected.items()
            if key != "manifest_body_sha256"
        }
        selected_body["authority"] = dict(selected_body["authority"])
        selected_body["authority"]["checkpoint_occurrence_projection"] = {
            "file_name": WORKHORSE_SELECTED_WORLD_AUTHORITY_FILE_V1,
            "bytes": len(authority_bytes),
            "sha256": hashlib.sha256(authority_bytes).hexdigest(),
            "projection_content_sha256": projection["projection_content_sha256"],
        }
        relocated_manifest = {
            **selected_body,
            "manifest_body_sha256": canonical_sha256(selected_body).hex,
        }
        copied_manifest = stage / WORKHORSE_SELECTED_WORLD_CHECKPOINT_FILE_V1
        copied_manifest.write_bytes(canonical_json_bytes(relocated_manifest))
        copied_manifest.chmod(0o444)
        _fsync_file_v1(copied_manifest)
        actual_bytes = sum(
            path.stat().st_size
            for path in stage.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
        if actual_bytes > maximum_bytes:
            raise ExpeditionOperatorError(
                "materialized exact founder exceeds its outer durable byte bound"
            )
        occurrence_stage.chmod(0o555)
        stage.chmod(0o555)
        _fsync_directory_v1(stage)

        # Reauthenticate the source after copying so a concurrent mutation can
        # never be mistaken for a stable founder materialization.
        resolve_workhorse_selected_world_continuation_v1(
            state_ref=original,
            authority_receipt_ref=source_authority_ref,
        )
        if destination.exists() or destination.is_symlink():
            raise ExpeditionOperatorError(
                "operator-owned exact founder destination appeared during publication"
            )
        os.rename(stage, destination)
        _fsync_directory_v1(destination.parent)
    except BaseException:
        if stage.exists() and not stage.is_symlink():
            stage.chmod(0o700)
            occurrence = stage / "occurrence"
            if occurrence.exists() and not occurrence.is_symlink():
                occurrence.chmod(0o700)
            shutil.rmtree(stage)
        raise
    return _materialized_state_ref_v1(
        original=original,
        package_root=destination,
    )


def _materialize_initial_exact_observation_sources_v1(
    *,
    manifest: ExpeditionManifestV1,
    control_root: Path,
) -> tuple[str, ...]:
    refs = tuple(
        ref
        for ref in manifest.founder_basis_refs
        if ref.artifact_kind == EXACT_OBSERVATION_FRAME_ARTIFACT_KIND_V1
    )
    retained_sources: list[str] = []
    bindings: list[dict[str, object]] = []
    lifecycle = resolve_expedition_lifecycle_policy_v1(manifest)
    durable_max_bytes = int(lifecycle.outer_contract["storage"]["durable_max_bytes"])
    founder_basis_root = control_root.parent / "durable" / "founder-basis"
    materialized_bytes = (
        sum(
            path.stat().st_size
            for path in founder_basis_root.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
        if founder_basis_root.is_dir() and not founder_basis_root.is_symlink()
        else 0
    )
    if materialized_bytes > durable_max_bytes:
        raise ExpeditionOperatorError(
            "operator-owned exact founders exceed the outer durable byte bound"
        )
    for ref in refs:
        retained = (
            control_root
            / "initial-observation-frames"
            / f"{_safe_component_v1(str(ref.artifact_id))}.json"
        )
        binding_path = retained.with_suffix(".binding.json")
        package_root = founder_basis_root / _safe_component_v1(str(ref.artifact_id))
        if binding_path.exists() or binding_path.is_symlink():
            recovered_binding = _read_json_mapping_v1(
                binding_path,
                field="retained initial exact binding",
            )
            manifest_state_ref_sha256 = recovered_binding.get(
                "manifest_state_ref_sha256"
            )
            raw = _read_json_mapping_v1(
                retained,
                field="retained initial exact ObservationFrame",
            )
            try:
                candidate = ObservationFrameV1.from_dict(raw)
            except (KeyError, TypeError, ValueError) as error:
                raise ExpeditionOperatorError(
                    "retained initial exact ObservationFrame is invalid"
                ) from error
            retained_ref = RunArtifactRefV1(
                artifact_id=ref.artifact_id,
                artifact_kind=ref.artifact_kind,
                artifact_schema_version=ref.artifact_schema_version,
                locator=str(retained),
                content_digest=candidate.observation_frame_sha256(),
                size_bytes=len(canonical_json_bytes(candidate.to_dict())),
            )
            frame = _authenticate_initial_exact_observation_frame_v1(
                ref=retained_ref,
                source=retained,
                manifest=manifest,
            )
        else:
            source = _local_file_locator_v1(ref.locator)
            source_frame = _authenticate_initial_exact_observation_frame_v1(
                ref=ref,
                source=source,
                manifest=manifest,
            )
            manifest_state_ref_sha256 = (
                source_frame.source_state_ref.state_ref_sha256().to_dict()
            )
            package_existed = package_root.exists()
            materialized_state = _materialize_initial_exact_state_package_v1(
                original=source_frame.source_state_ref,
                destination=package_root,
                maximum_bytes=durable_max_bytes - materialized_bytes,
            )
            if not package_existed:
                materialized_bytes += sum(
                    path.stat().st_size
                    for path in package_root.rglob("*")
                    if path.is_file() and not path.is_symlink()
                )
            frame_value = source_frame.to_dict()
            frame_value["source_state_ref"] = materialized_state.to_dict()
            frame = ObservationFrameV1.from_dict(frame_value)
            _publish_canonical_once_v1(retained, frame.to_dict())
            frame = _authenticate_initial_exact_observation_frame_v1(
                ref=RunArtifactRefV1(
                    artifact_id=ref.artifact_id,
                    artifact_kind=ref.artifact_kind,
                    artifact_schema_version=ref.artifact_schema_version,
                    locator=str(retained),
                    content_digest=frame.observation_frame_sha256(),
                    size_bytes=len(canonical_json_bytes(frame.to_dict())),
                ),
                source=retained,
                manifest=manifest,
            )
        bound_package_root = Path(
            frame.source_state_ref.source_artifact_ref.locator
        ).parent
        try:
            bound_package_root.relative_to(founder_basis_root)
        except ValueError as error:
            raise ExpeditionOperatorError(
                "retained exact founder frame is not operator-owned"
            ) from error
        if bound_package_root != package_root:
            raise ExpeditionOperatorError(
                "retained exact founder is bound to another operator package"
            )
        if (
            _materialized_state_ref_v1(
                original=frame.source_state_ref,
                package_root=package_root,
            )
            != frame.source_state_ref
        ):
            raise ExpeditionOperatorError(
                "retained exact founder StateRef differs from its operator package"
            )
        retained_sources.append(str(retained.resolve()))
        binding_core = {
            "schema_version": (
                "eiel_adaptive_expedition_initial_exact_binding_internal_v1"
            ),
            "manifest_sha256": manifest.manifest_sha256().to_dict(),
            "artifact_ref": ref.to_dict(),
            "retained_path": str(retained.resolve()),
            "manifest_observation_frame_sha256": ref.content_digest.to_dict(),
            "manifest_state_ref_sha256": manifest_state_ref_sha256,
            "materialized_observation_frame_sha256": (
                frame.observation_frame_sha256().to_dict()
            ),
            "materialized_state_ref_sha256": (
                frame.source_state_ref.state_ref_sha256().to_dict()
            ),
            "operator_owned_exact_package_root": str(package_root.resolve()),
            "external_frame_required_after_start": False,
            "external_selected_state_required_after_start": False,
            "external_occurrence_files_required_after_start": False,
        }
        binding = {
            **binding_core,
            "binding_sha256": canonical_sha256(binding_core).to_dict(),
        }
        _publish_canonical_once_v1(binding_path, binding)
        bindings.append(binding)
    _publish_canonical_once_v1(
        control_root / "initial_observation_bindings.json",
        {
            "schema_version": (
                "eiel_adaptive_expedition_initial_observation_bindings_internal_v1"
            ),
            "manifest_sha256": manifest.manifest_sha256().to_dict(),
            "bindings": bindings,
        },
    )
    return tuple(retained_sources)


def _format_utc_v1(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ExpeditionOperatorError(
            "operator clock must return timezone-aware UTC time"
        )
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _parse_utc_v1(value: object, *, field: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise ExpeditionOperatorError(f"{field} is not canonical UTC time")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ExpeditionOperatorError(f"{field} is not canonical UTC time") from error
    if _format_utc_v1(parsed) != value:
        raise ExpeditionOperatorError(f"{field} is not canonical UTC time")
    return parsed


def _try_create_canonical_once_v1(path: Path, value: object) -> bool:
    """Atomically create canonical bytes, or return False for an existing path."""

    if path.is_symlink():
        raise ExpeditionOperatorError(f"immutable output is a symbolic link: {path}")
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ExpeditionOperatorError(f"immutable output parent is invalid: {parent}")
    expected = canonical_json_bytes(value)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        created = True
        try:
            os.link(temporary, path)
        except FileExistsError:
            created = False
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return created


def _publish_canonical_once_v1(path: Path, value: object) -> Path:
    """Atomically create one immutable canonical file or accept identical bytes."""

    expected = canonical_json_bytes(value)
    if not _try_create_canonical_once_v1(path, value):
        if path.is_symlink() or not path.is_file():
            raise ExpeditionOperatorError(
                f"immutable output is not a regular file: {path}"
            )
        try:
            existing = path.read_bytes()
        except OSError as error:
            raise ExpeditionOperatorError(
                f"immutable output is unreadable: {path}"
            ) from error
        if existing != expected:
            raise ExpeditionOperatorError(
                f"immutable output differs from the recovered run: {path}"
            )
    return path


def _frontier_dispatch_target_v1(
    ledger: Mapping[str, object], *, maximum_frontiers: int
) -> int | None:
    """Return the next frontier that could dispatch, or None after the last one."""

    projection = reconstruct_expedition_frontier_v1(ledger)
    current = projection["current_frontier"]
    if current is None:
        target = 0
    else:
        decision = current["decision"]
        if decision is None or (
            bool(decision["action_ids"]) and not current["actions"]
        ):
            target = int(current["frontier_ordinal"])
        elif all(
            action["settlement"] is not None for action in current["actions"].values()
        ):
            target = int(current["frontier_ordinal"]) + 1
        else:
            raise ExpeditionOperatorError(
                "safe-frontier callback was invoked with an active child"
            )
    return None if target >= maximum_frontiers else target


def _read_frontier_timing_records_v1(
    root: Path,
    *,
    start_record: Mapping[str, object],
) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    if root.is_symlink() or not root.is_dir():
        raise ExpeditionOperatorError("frontier timing root is invalid")
    removed_temporary = False
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        match = _FRONTIER_TIMING_TEMP_FILE_RE_V1.fullmatch(path.name)
        if match is None:
            continue
        try:
            metadata = path.lstat()
        except OSError as error:
            raise ExpeditionOperatorError(
                "frontier timing temporary debris cannot be inspected"
            ) from error
        target = root / f"{match.group(1)}.json"
        target_metadata = None
        if target.exists() and not target.is_symlink():
            try:
                target_metadata = target.stat()
            except OSError as error:
                raise ExpeditionOperatorError(
                    "frontier timing target cannot be inspected"
                ) from error
        linked_to_target = (
            metadata.st_nlink == 2
            and target_metadata is not None
            and (metadata.st_dev, metadata.st_ino)
            == (target_metadata.st_dev, target_metadata.st_ino)
        )
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size > _MAX_FRONTIER_TIMING_TEMP_BYTES_V1
            or metadata.st_nlink not in {1, 2}
            or (metadata.st_nlink == 2 and not linked_to_target)
        ):
            raise ExpeditionOperatorError(
                "frontier timing temporary debris is not a strict private file"
            )
        try:
            path.unlink()
        except OSError as error:
            raise ExpeditionOperatorError(
                "frontier timing temporary debris cannot be recovered"
            ) from error
        removed_temporary = True
    if removed_temporary:
        _fsync_directory_v1(root)
    records: list[dict[str, Any]] = []
    previous_sha256: str | None = None
    previous_time = _parse_utc_v1(
        start_record["started_at_utc"], field="started_at_utc"
    )
    for expected_ordinal, path in enumerate(sorted(root.iterdir())):
        match = _FRONTIER_TIMING_FILE_RE_V1.fullmatch(path.name)
        if match is None or int(match.group(1)) != expected_ordinal:
            raise ExpeditionOperatorError(
                "frontier timing records must be contiguous canonical files"
            )
        value = _read_json_mapping_v1(path, field="frontier timing record")
        required = {
            "schema_version",
            "manifest_sha256",
            "frontier_ordinal",
            "dispatch_observed_at_utc",
            "deadline_at_utc",
            "reserve_basis",
            "reserve_seconds_before_dispatch",
            "maximum_observed_frontier_duration_seconds",
            "observed_frontier_count",
            "previous_timing_sha256",
            "timing_sha256",
        }
        core = {key: item for key, item in value.items() if key != "timing_sha256"}
        observed = _parse_utc_v1(
            value.get("dispatch_observed_at_utc"),
            field="dispatch_observed_at_utc",
        )
        if (
            set(value) != required
            or value.get("schema_version") != FRONTIER_TIMING_SCHEMA_VERSION_V1
            or value.get("manifest_sha256") != start_record["manifest_sha256"]
            or value.get("frontier_ordinal") != expected_ordinal
            or value.get("deadline_at_utc") != start_record["deadline_at_utc"]
            or value.get("reserve_basis") != SAFE_FRONTIER_RESERVE_BASIS_V1
            or value.get("previous_timing_sha256") != previous_sha256
            or value.get("timing_sha256") != canonical_sha256(core).hex
            or type(value.get("reserve_seconds_before_dispatch")) is not int
            or int(value["reserve_seconds_before_dispatch"]) < 1
            or type(value.get("maximum_observed_frontier_duration_seconds")) is not int
            or int(value["maximum_observed_frontier_duration_seconds"]) < 0
            or value.get("observed_frontier_count") != expected_ordinal
            or observed < previous_time
        ):
            raise ExpeditionOperatorError(
                "frontier timing record differs from its frozen chain"
            )
        records.append(value)
        previous_sha256 = value["timing_sha256"]
        previous_time = observed
    return records


def _safe_frontier_reserve_v1(
    *,
    wall_time_budget_seconds: int,
    records: Sequence[Mapping[str, object]],
    candidate_observed_at: datetime | None,
) -> tuple[int, int, int]:
    cold = min(
        wall_time_budget_seconds,
        max(
            1,
            min(
                SAFE_FRONTIER_COLD_RESERVE_MAX_SECONDS_V1,
                math.ceil(wall_time_budget_seconds * 5 / 100),
            ),
        ),
    )
    times = [
        _parse_utc_v1(row["dispatch_observed_at_utc"], field="dispatch timing")
        for row in records
    ]
    durations = [
        max(0, math.ceil((later - earlier).total_seconds()))
        for earlier, later in zip(times, times[1:])
    ]
    observed_count = len(durations)
    if candidate_observed_at is not None and times:
        if candidate_observed_at < times[-1]:
            raise ExpeditionOperatorError("operator clock moved backwards")
        durations.append(
            max(0, math.ceil((candidate_observed_at - times[-1]).total_seconds()))
        )
        observed_count += 1
    maximum_observed = max(durations, default=0)
    reserve = min(
        wall_time_budget_seconds,
        max(cold, cold + (2 * maximum_observed)),
    )
    return reserve, maximum_observed, observed_count


def _publish_frontier_timing_record_v1(
    *,
    root: Path,
    start_record: Mapping[str, object],
    frontier_ordinal: int,
    observed_at: datetime,
    reserve_seconds: int,
    maximum_observed_seconds: int,
    observed_frontier_count: int,
    previous_sha256: str | None,
) -> dict[str, Any]:
    core = {
        "schema_version": FRONTIER_TIMING_SCHEMA_VERSION_V1,
        "manifest_sha256": start_record["manifest_sha256"],
        "frontier_ordinal": frontier_ordinal,
        "dispatch_observed_at_utc": _format_utc_v1(observed_at),
        "deadline_at_utc": start_record["deadline_at_utc"],
        "reserve_basis": SAFE_FRONTIER_RESERVE_BASIS_V1,
        "reserve_seconds_before_dispatch": reserve_seconds,
        "maximum_observed_frontier_duration_seconds": maximum_observed_seconds,
        "observed_frontier_count": observed_frontier_count,
        "previous_timing_sha256": previous_sha256,
    }
    value = {**core, "timing_sha256": canonical_sha256(core).hex}
    _publish_canonical_once_v1(
        root / f"{frontier_ordinal:08d}.json",
        value,
    )
    return value


def _frontier_timing_summary_v1(
    *,
    timing_root: Path,
    start_record: Mapping[str, object],
    terminal_payload: Mapping[str, object],
    ledger: Mapping[str, object],
) -> dict[str, object]:
    records = _read_frontier_timing_records_v1(timing_root, start_record=start_record)
    stop = reconstruct_expedition_frontier_v1(ledger)["stop"] or {}
    maximum_observed = max(
        [int(row["maximum_observed_frontier_duration_seconds"]) for row in records]
        + [int(stop.get("maximum_observed_frontier_duration_seconds", 0))]
    )
    reserve = max(
        [int(row["reserve_seconds_before_dispatch"]) for row in records]
        + [int(stop.get("reserve_seconds_before_dispatch", 0))]
    )
    return {
        "schema_version": "eiel_adaptive_expedition_frontier_timing_summary_internal_v1",
        "wall_time_budget_seconds": start_record["wall_time_budget_seconds"],
        "deadline_at_utc": start_record["deadline_at_utc"],
        "dispatch_timing_record_count": len(records),
        "maximum_observed_frontier_duration_seconds": maximum_observed,
        "maximum_safe_frontier_reserve_seconds": reserve,
        "reserve_basis": SAFE_FRONTIER_RESERVE_BASIS_V1,
        "terminal_status": terminal_payload["terminal_status"],
        "terminal_stop_reason": stop.get("reason"),
        "timing_records_relative_path": "../control/frontier-dispatch-timing",
    }


def _read_start_record_v1(
    path: Path,
    *,
    manifest: ExpeditionManifestV1,
    policy: ExpeditionPolicyV1,
    runtime_root: Path,
    child_runtime_parent: Path,
    child_materialization_parent: Path,
    execution_mode: str,
    current_source_commit: str,
) -> tuple[dict[str, Any], EielScienceRequestV1]:
    value = _read_json_mapping_v1(path, field="operator start record")
    required = {
        "schema_version",
        "expedition_id",
        "manifest_sha256",
        "policy_sha256",
        "source_commit",
        "founder_request_sha256",
        "founder_request_canonical_size_bytes",
        "founder_source_locator",
        "founder_request",
        "runtime_root",
        "child_runtime_parent",
        "child_materialization_parent",
        "started_at_utc",
        "deadline_at_utc",
        "wall_time_budget_seconds",
        "maximum_frontiers",
        "maximum_actions_per_frontier",
        "maximum_atlas_entries",
        "physical_batch_worlds",
        "execution_mode",
        "no_claims",
        "qualification_status",
    }
    if set(value) != required:
        raise ExpeditionOperatorError("operator start record fields differ")
    founder_ref = _founder_request_ref_v1(manifest)
    try:
        founder_request = validate_eiel_science_request_v1(
            _canonical_mapping_v1(
                value["founder_request"], field="retained founder Science Request"
            )
        )
    except (TypeError, ValueError, KeyError) as error:
        raise ExpeditionOperatorError(
            "operator start record has an invalid retained founder request"
        ) from error
    founder_bytes = canonical_json_bytes(founder_request.to_dict())
    if (
        founder_request.request_sha256() != founder_ref.content_digest
        or founder_ref.size_bytes is None
        or len(founder_bytes) != founder_ref.size_bytes
        or value["founder_request_canonical_size_bytes"] != len(founder_bytes)
        or value["founder_source_locator"] != founder_ref.locator
    ):
        raise ExpeditionOperatorError(
            "retained founder request differs from the immutable manifest reference"
        )
    expected = {
        "schema_version": OPERATOR_START_SCHEMA_VERSION_V1,
        "expedition_id": manifest.expedition_id,
        "manifest_sha256": manifest.manifest_sha256().hex,
        "policy_sha256": canonical_sha256(canonical_plain_value_v1(policy.binding)).hex,
        "source_commit": value["source_commit"],
        "founder_request_sha256": founder_request.request_sha256().hex,
        "runtime_root": str(runtime_root),
        "child_runtime_parent": str(child_runtime_parent),
        "child_materialization_parent": str(child_materialization_parent),
        "wall_time_budget_seconds": manifest.wall_time_budget_seconds,
        "maximum_frontiers": manifest.maximum_frontiers,
        "maximum_actions_per_frontier": manifest.maximum_actions_per_frontier,
        "maximum_atlas_entries": manifest.maximum_atlas_entries,
        "physical_batch_worlds": _founder_nominal_batch_worlds_v1(founder_request),
        "execution_mode": execution_mode,
        "no_claims": True,
        "qualification_status": "NOT_REQUESTED",
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ExpeditionOperatorError(
            "operator start record differs from the immutable manifest or policy"
        )
    source_commit = value.get("source_commit")
    if (
        type(source_commit) is not str
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ExpeditionOperatorError("operator start record source commit is invalid")
    if source_commit != current_source_commit:
        raise ExpeditionOperatorError(
            "operator recovery source commit differs from the immutable start record"
        )
    started = _parse_utc_v1(value["started_at_utc"], field="started_at_utc")
    deadline_raw = value["deadline_at_utc"]
    expected_deadline = (
        None
        if manifest.wall_time_budget_seconds is None
        else _format_utc_v1(
            started + timedelta(seconds=manifest.wall_time_budget_seconds)
        )
    )
    if deadline_raw != expected_deadline:
        raise ExpeditionOperatorError(
            "operator deadline differs from the frozen budget"
        )
    return value, founder_request


def _initialize_start_record_v1(
    path: Path,
    *,
    manifest: ExpeditionManifestV1,
    policy: ExpeditionPolicyV1,
    founder_request: EielScienceRequestV1 | None,
    runtime_root: Path,
    child_runtime_parent: Path,
    child_materialization_parent: Path,
    clock: ClockV1,
    execution_mode: str,
    source_commit: str,
) -> tuple[dict[str, Any], EielScienceRequestV1]:
    if path.exists() or path.is_symlink():
        return _read_start_record_v1(
            path,
            manifest=manifest,
            policy=policy,
            runtime_root=runtime_root,
            child_runtime_parent=child_runtime_parent,
            child_materialization_parent=child_materialization_parent,
            execution_mode=execution_mode,
            current_source_commit=source_commit,
        )
    if founder_request is None:
        raise ExpeditionOperatorError(
            "initial launch requires the manifest-bound founder request"
        )
    founder_ref = _founder_request_ref_v1(manifest)
    founder_bytes = canonical_json_bytes(founder_request.to_dict())
    started = clock()
    started_text = _format_utc_v1(started)
    deadline_text = (
        None
        if manifest.wall_time_budget_seconds is None
        else _format_utc_v1(
            started + timedelta(seconds=manifest.wall_time_budget_seconds)
        )
    )
    value = {
        "schema_version": OPERATOR_START_SCHEMA_VERSION_V1,
        "expedition_id": manifest.expedition_id,
        "manifest_sha256": manifest.manifest_sha256().hex,
        "policy_sha256": canonical_sha256(canonical_plain_value_v1(policy.binding)).hex,
        "source_commit": source_commit,
        "founder_request_sha256": founder_request.request_sha256().hex,
        "founder_request_canonical_size_bytes": len(founder_bytes),
        "founder_source_locator": founder_ref.locator,
        "founder_request": founder_request.to_dict(),
        "runtime_root": str(runtime_root),
        "child_runtime_parent": str(child_runtime_parent),
        "child_materialization_parent": str(child_materialization_parent),
        "started_at_utc": started_text,
        "deadline_at_utc": deadline_text,
        "wall_time_budget_seconds": manifest.wall_time_budget_seconds,
        "maximum_frontiers": manifest.maximum_frontiers,
        "maximum_actions_per_frontier": manifest.maximum_actions_per_frontier,
        "maximum_atlas_entries": manifest.maximum_atlas_entries,
        "physical_batch_worlds": _founder_nominal_batch_worlds_v1(founder_request),
        "execution_mode": execution_mode,
        "no_claims": True,
        "qualification_status": "NOT_REQUESTED",
    }
    _try_create_canonical_once_v1(path, value)
    return _read_start_record_v1(
        path,
        manifest=manifest,
        policy=policy,
        runtime_root=runtime_root,
        child_runtime_parent=child_runtime_parent,
        child_materialization_parent=child_materialization_parent,
        execution_mode=execution_mode,
        current_source_commit=source_commit,
    )


def _empty_scientific_state_v1(
    manifest: ExpeditionManifestV1,
) -> tuple[EvolutionaryAtlasV1, RetainedParentBankV1]:
    atlas = EvolutionaryAtlasV1.from_dict(
        {
            "schema_version": "eiel_adaptive_science_evolutionary_atlas_v1",
            "expedition_id": manifest.expedition_id,
            "manifest_sha256": manifest.manifest_sha256().to_dict(),
            "revision": 0,
            "through_frontier_ordinal": 0,
            "descriptor_set_sha256": (
                manifest.observation_descriptor_set_sha256.to_dict()
            ),
            "maximum_entries": manifest.maximum_atlas_entries,
            "entries": [],
            "previous_atlas_sha256": None,
        }
    )
    bank = RetainedParentBankV1.from_dict(
        {
            "schema_version": "eiel_adaptive_science_retained_parent_bank_v1",
            "expedition_id": manifest.expedition_id,
            "manifest_sha256": manifest.manifest_sha256().to_dict(),
            "revision": 0,
            "maximum_entries": manifest.maximum_retained_parent_entries,
            "maximum_total_bytes": manifest.maximum_retained_parent_bytes,
            "current_entry_count": 0,
            "current_total_bytes": 0,
            "entries": [],
            "previous_bank_sha256": None,
        }
    )
    bank.validate_against_manifest(manifest)
    return atlas, bank


def run_expedition_operator_v1(
    *,
    manifest: ExpeditionManifestV1,
    policy: ExpeditionPolicyV1,
    runtime_parent: str | Path = DEFAULT_EXPEDITION_RUNTIME_PARENT_V1,
    repo_root: str | Path | None = None,
    dependencies: ExpeditionOperatorDependenciesV1 = ExpeditionOperatorDependenciesV1(),
) -> ExpeditionOperatorResultV1:
    """Launch or resume one manifest-bound Expedition through canonical children."""

    if (
        type(dependencies.execution_mode) is not str
        or not dependencies.execution_mode
        or type(dependencies.programme_stage) is not str
        or not dependencies.programme_stage
        or type(dependencies.historical_continuation_authority) is not bool
    ):
        raise ExpeditionOperatorError("operator loop identity is invalid")
    try:
        validate_policy_review_presentation_v1(
            policy.profile.get("human_review_projection")
        )
    except ValueError as error:
        raise ExpeditionOperatorError(
            "manifest-bound policy human-review presentation is invalid"
        ) from error
    lifecycle_policy = resolve_expedition_lifecycle_policy_v1(manifest)
    try:
        _preflight_expedition_outer_object_budget_v1(manifest)
    except ExpeditionLifecycleError as error:
        raise ExpeditionOperatorError(str(error)) from error
    parent = Path(runtime_parent).expanduser().absolute()
    if parent.exists() and (parent.is_symlink() or not parent.is_dir()):
        raise ExpeditionOperatorError("Expedition runtime parent is invalid")
    parent.mkdir(parents=True, exist_ok=True)
    manifest_sha256 = manifest.manifest_sha256().hex
    runtime_root = (
        parent / f"{_safe_component_v1(manifest.expedition_id)}--{manifest_sha256[:24]}"
    )
    if runtime_root.exists() and (
        runtime_root.is_symlink() or not runtime_root.is_dir()
    ):
        raise ExpeditionOperatorError("Expedition runtime root is invalid")
    machine_runtime_root = parent.parent
    child_runtime_parent = machine_runtime_root / "children" / runtime_root.name
    child_materialization_parent = (
        machine_runtime_root / "materializations" / runtime_root.name
    )
    if (
        runtime_root == child_runtime_parent
        or runtime_root in child_runtime_parent.parents
        or child_runtime_parent in runtime_root.parents
        or runtime_root == child_materialization_parent
        or runtime_root in child_materialization_parent.parents
        or child_materialization_parent in runtime_root.parents
    ):
        raise ExpeditionOperatorError(
            "child execution roots must be disjoint from the compact outer lifecycle root"
        )
    lifecycle_context: ExpeditionLifecycleContextV1 = open_expedition_lifecycle_v1(
        manifest=manifest,
        run_root=runtime_root,
    )
    control = runtime_root / "control"
    closeout = runtime_root / "closeout"
    try:
        start_path = control / "operator_start.json"
        source_commit = _source_commit_v1(repo_root)
        founder_candidate: EielScienceRequestV1 | None = None
        if not start_path.exists() and not start_path.is_symlink():
            try:
                founder_candidate, _ = resolve_founder_request_v1(manifest)
            except ExpeditionOperatorError:
                if not start_path.exists() and not start_path.is_symlink():
                    raise
        # Exact founder packages must be independently reopenable before the
        # operator start record freezes the Expedition.  After that record is
        # published, recovery never depends on the manifest's external frame,
        # selected-state payload, or occurrence files remaining present.
        initial_observation_sources = _materialize_initial_exact_observation_sources_v1(
            manifest=manifest,
            control_root=control,
        )
        start_record, founder_request = _initialize_start_record_v1(
            start_path,
            manifest=manifest,
            policy=policy,
            founder_request=founder_candidate,
            runtime_root=runtime_root,
            child_runtime_parent=child_runtime_parent,
            child_materialization_parent=child_materialization_parent,
            clock=dependencies.clock,
            execution_mode=dependencies.execution_mode,
            source_commit=source_commit,
        )
        _publish_canonical_once_v1(
            control / "expedition_manifest.json", manifest.to_dict()
        )
        _publish_canonical_once_v1(
            control / "founder_science_request.json", founder_request.to_dict()
        )
        _publish_canonical_once_v1(
            control / "founder_source_binding.json",
            {
                "schema_version": (
                    "eiel_adaptive_expedition_founder_source_binding_v1"
                ),
                "founder_request_sha256": founder_request.request_sha256().hex,
                "source_locator": start_record["founder_source_locator"],
                "retained_request_path": str(control / "founder_science_request.json"),
                "external_source_required_after_start": False,
                "manifest_sha256": manifest_sha256,
            },
        )
        atlas, bank = _empty_scientific_state_v1(manifest)
        deadline = (
            None
            if start_record["deadline_at_utc"] is None
            else _parse_utc_v1(start_record["deadline_at_utc"], field="deadline_at_utc")
        )
        timing_root = control / "frontier-dispatch-timing"

        def active_parent_release_boundary(
            *,
            manifest: ExpeditionManifestV1,
            ledger: Mapping[str, object],
            frontier_ordinal: int,
            decision_payload: Mapping[str, object],
            selected_children: Sequence[object],
        ) -> Mapping[str, object]:
            selected_parent_state_ref_ids = []
            for child in selected_children:
                parent_state_ref = getattr(child, "parent_state_ref", None)
                if parent_state_ref is not None:
                    selected_parent_state_ref_ids.append(
                        f"SR-{parent_state_ref.state_ref_sha256().hex}"
                    )
            return release_active_parent_packages_v1(
                lifecycle_context,
                ledger=ledger,
                frontier_ordinal=frontier_ordinal,
                decision_payload=decision_payload,
                selected_parent_state_ref_ids=selected_parent_state_ref_ids,
            )

        def safe_frontier_stop(
            frontier_ordinal: int,
            current_ledger: Mapping[str, object],
        ) -> Mapping[str, object] | None:
            dispatch_target = _frontier_dispatch_target_v1(
                current_ledger,
                maximum_frontiers=manifest.maximum_frontiers,
            )
            if dispatch_target is None:
                return None
            observed = dependencies.clock()
            observed_text = _format_utc_v1(observed)
            started_at = _parse_utc_v1(
                start_record["started_at_utc"], field="started_at_utc"
            )
            if observed < started_at:
                raise ExpeditionOperatorError(
                    "operator clock moved backwards during recovery"
                )
            if deadline is None:
                enforce_expedition_storage_v1(
                    lifecycle_context,
                    stage=f"before_frontier_{dispatch_target:04d}_dispatch",
                )
                return None
            records = _read_frontier_timing_records_v1(
                timing_root,
                start_record=start_record,
            )
            previous_observed = _parse_utc_v1(
                (
                    start_record["started_at_utc"]
                    if not records
                    else records[-1]["dispatch_observed_at_utc"]
                ),
                field="recovery timing boundary",
            )
            if observed < previous_observed:
                raise ExpeditionOperatorError(
                    "operator clock moved backwards during recovery"
                )
            if dispatch_target > len(records):
                raise ExpeditionOperatorError(
                    "frontier timing chain skipped a dispatch boundary"
                )
            existing = (
                None if dispatch_target == len(records) else records[dispatch_target]
            )
            if existing is None:
                reserve_seconds, maximum_observed, observed_count = (
                    _safe_frontier_reserve_v1(
                        wall_time_budget_seconds=int(manifest.wall_time_budget_seconds),
                        records=records,
                        candidate_observed_at=observed,
                    )
                )
            else:
                reserve_seconds = int(existing["reserve_seconds_before_dispatch"])
                maximum_observed = int(
                    existing["maximum_observed_frontier_duration_seconds"]
                )
                observed_count = int(existing["observed_frontier_count"])
            remaining_seconds = max(
                0,
                math.ceil((deadline - observed).total_seconds()),
            )
            common_stop = {
                "frontier_ordinal": frontier_ordinal,
                "next_frontier_ordinal": dispatch_target,
                "deadline_at_utc": start_record["deadline_at_utc"],
                "observed_at_utc": observed_text,
                "remaining_wall_time_seconds": remaining_seconds,
                "reserve_seconds_before_dispatch": reserve_seconds,
                "reserve_basis": SAFE_FRONTIER_RESERVE_BASIS_V1,
                "maximum_observed_frontier_duration_seconds": maximum_observed,
                "observed_frontier_count": observed_count,
                "budget_includes_recovery_downtime": True,
            }
            if observed >= deadline:
                return {
                    "reason": "WALL_TIME_BUDGET_REACHED_AT_SAFE_FRONTIER",
                    **common_stop,
                }
            if remaining_seconds <= reserve_seconds:
                return {
                    "reason": "WALL_TIME_SAFE_FRONTIER_RESERVE_REACHED",
                    **common_stop,
                }
            enforce_expedition_storage_v1(
                lifecycle_context,
                stage=f"before_frontier_{dispatch_target:04d}_dispatch",
            )
            if existing is None:
                _publish_frontier_timing_record_v1(
                    root=timing_root,
                    start_record=start_record,
                    frontier_ordinal=dispatch_target,
                    observed_at=observed,
                    reserve_seconds=reserve_seconds,
                    maximum_observed_seconds=maximum_observed,
                    observed_frontier_count=observed_count,
                    previous_sha256=(
                        None if not records else records[-1]["timing_sha256"]
                    ),
                )
            return None

        closed_loop_dependencies = dependencies.closed_loop_dependencies
        if isinstance(closed_loop_dependencies, GeneralClosedLoopDependenciesV1):
            # The operator owns the immutable programme identity for this run;
            # keep the generic loop's injected mechanics while binding that
            # identity instead of silently labelling every Expedition G4.
            closed_loop_dependencies = replace(
                closed_loop_dependencies,
                programme_stage=dependencies.programme_stage,
                historical_continuation_authority=(
                    dependencies.historical_continuation_authority
                ),
                retention_lifecycle_boundary=active_parent_release_boundary,
            )
        loop_result = dependencies.closed_loop_runner(
            manifest=manifest,
            policy=policy,
            founder_request=founder_request,
            initial_observation_sources=initial_observation_sources,
            ledger_root=runtime_root / "durable" / "ledger",
            runtime_parent=child_runtime_parent,
            atlas=atlas,
            retained_parent_bank=bank,
            maximum_frontiers=manifest.maximum_frontiers,
            logical_capacity=manifest.maximum_actions_per_frontier,
            # This is the immutable founder request's nominal batch
            # projection.  Actual physical partitioning is selected by the
            # Science Request authority/front door for each child.
            physical_batch_worlds=int(start_record["physical_batch_worlds"]),
            repo_root=None if repo_root is None else str(Path(repo_root).absolute()),
            materialization_parent=str(child_materialization_parent),
            shadow_mode=False,
            dependencies=closed_loop_dependencies,
            safe_frontier_stop=safe_frontier_stop,
        )
        if not loop_result.terminal or loop_result.shadow_mode:
            raise ExpeditionOperatorError(
                "canonical Expedition loop returned without live terminal settlement"
            )
        terminal_payload = loop_result.ledger["events"][-1]["payload"]
        if (
            terminal_payload.get("programme_stage") != dependencies.programme_stage
            or terminal_payload.get("historical_continuation_authority")
            is not dependencies.historical_continuation_authority
        ):
            raise ExpeditionOperatorError(
                "canonical Expedition loop terminal identity differs from the operator"
            )

        enforce_expedition_storage_v1(
            lifecycle_context, stage="before_operator_closeout"
        )
        atlas_path = _publish_canonical_once_v1(
            closeout / "evolutionary_atlas.json", loop_result.atlas.to_dict()
        )
        bank_path = _publish_canonical_once_v1(
            closeout / "retained_parent_bank.json",
            loop_result.retained_parent_bank.to_dict(),
        )
        public_journal = project_ledger_to_public_journal_v1(loop_result.ledger)
        journal_path = _publish_canonical_once_v1(
            closeout / "expedition_journal.json", public_journal
        )
        experiment_dag = validate_experiment_dag_v1(
            build_experiment_dag_v1(loop_result.ledger)
        )
        experiment_dag_path = _publish_canonical_once_v1(
            closeout / "experiment_dag.json", experiment_dag
        )
        experiment_dag_sha256 = canonical_sha256(experiment_dag)
        ledger_sha256 = canonical_sha256(loop_result.ledger)
        journal_sha256 = canonical_sha256(public_journal)
        artifact_index = {
            "manifest": {
                "artifact_id": "EXPEDITION_MANIFEST",
                "relative_path": "../control/expedition_manifest.json",
                "content_sha256": manifest.manifest_sha256().to_dict(),
            },
            "ledger": {
                "artifact_id": "EXPEDITION_LEDGER",
                "relative_path": "../durable/ledger",
                "content_sha256": ledger_sha256.to_dict(),
                "head_event_sha256": loop_result.ledger["head_event_sha256"],
            },
            "atlas": {
                "artifact_id": "EVOLUTIONARY_ATLAS",
                "relative_path": "evolutionary_atlas.json",
                "content_sha256": loop_result.atlas.atlas_sha256().to_dict(),
            },
            "retained_parent_bank": {
                "artifact_id": "RETAINED_PARENT_BANK",
                "relative_path": "retained_parent_bank.json",
                "content_sha256": (
                    loop_result.retained_parent_bank.bank_sha256().to_dict()
                ),
            },
            "journal": {
                "artifact_id": "EXPEDITION_JOURNAL",
                "relative_path": "expedition_journal.json",
                "content_sha256": journal_sha256.to_dict(),
            },
            "experiment_dag": {
                "artifact_id": "EXPERIMENT_DAG",
                "relative_path": "experiment_dag.json",
                "content_sha256": experiment_dag_sha256.to_dict(),
            },
        }
        review_projection = build_human_review_projection_v1(
            manifest=manifest,
            policy_profile=policy.profile,
            ledger=loop_result.ledger,
            atlas=loop_result.atlas,
            retained_parent_bank=loop_result.retained_parent_bank,
            public_journal=public_journal,
            artifact_index=artifact_index,
            programme_stage=dependencies.programme_stage,
            historical_continuation_authority=(
                dependencies.historical_continuation_authority
            ),
        )
        performance_summary = _frontier_timing_summary_v1(
            timing_root=timing_root,
            start_record=start_record,
            terminal_payload=terminal_payload,
            ledger=loop_result.ledger,
        )
        human_review_value = build_expedition_closeout_v1(
            expedition_id=manifest.expedition_id,
            terminal_status=str(terminal_payload["terminal_status"]),
            programme_stage=dependencies.programme_stage,
            source_commit=str(start_record["source_commit"]),
            manifest_sha256=manifest.manifest_sha256().to_dict(),
            policy_sha256={
                "algorithm": "sha256",
                "hex": start_record["policy_sha256"],
            },
            ledger_sha256=ledger_sha256.to_dict(),
            atlas_sha256=loop_result.atlas.atlas_sha256().to_dict(),
            retained_parent_bank_sha256=(
                loop_result.retained_parent_bank.bank_sha256().to_dict()
            ),
            review_projection=review_projection,
            performance_summary=performance_summary,
            lifecycle_summary={
                "outer_lifecycle_contract_sha256": lifecycle_context.contract[
                    "contract_sha256"
                ],
                "child_contract_class": lifecycle_policy.child_contract_class,
                "storage_caps": lifecycle_context.contract["storage"],
                "retained_parent_entry_count": (
                    loop_result.retained_parent_bank.current_entry_count
                ),
                "retained_parent_bytes": (
                    loop_result.retained_parent_bank.current_total_bytes
                ),
                "deletion_authority": False,
                "slow_archive_is_background": True,
            },
            validation_summary={
                "authority_effect": "NONE_REPORT_ONLY",
                "authenticated_observations_only": True,
                "policy_replay": "BYTE_IDENTITY_ENFORCED_PER_DECISION",
                "qualification_status": "NOT_REQUESTED",
            },
        )
        human_review_path = _publish_canonical_once_v1(
            closeout / "human_review_package.json",
            human_review_value,
        )
        closeout_value = {
            "schema_version": OPERATOR_CLOSEOUT_SCHEMA_VERSION_V1,
            "expedition_id": manifest.expedition_id,
            "manifest_sha256": manifest_sha256,
            "policy_sha256": start_record["policy_sha256"],
            # Keep the coordinator implementation identity at the outer
            # durable closeout boundary as well as in the immutable start
            # record and human review package.  Recovery already rejects a
            # source change; repeating the binding here makes the terminal
            # package independently provenance-auditable.
            "source_commit": start_record["source_commit"],
            "founder_request_sha256": founder_request.request_sha256().hex,
            "runtime_root": str(runtime_root),
            "child_runtime_parent": str(child_runtime_parent),
            "child_materialization_parent": str(child_materialization_parent),
            "child_contract_class": lifecycle_policy.child_contract_class,
            "outer_lifecycle_contract_sha256": lifecycle_context.contract[
                "contract_sha256"
            ],
            "terminal_status": terminal_payload["terminal_status"],
            "head_event_sha256": loop_result.ledger["head_event_sha256"],
            "frontier_count": len(loop_result.decisions),
            "child_count": len(loop_result.children),
            "atlas_sha256": loop_result.atlas.atlas_sha256().hex,
            "retained_parent_bank_sha256": (
                loop_result.retained_parent_bank.bank_sha256().hex
            ),
            "public_journal_sha256": journal_sha256.hex,
            "human_review_sha256": canonical_sha256(human_review_value).hex,
            "human_review_path": str(human_review_path),
            "experiment_dag_path": str(experiment_dag_path),
            "experiment_dag_sha256": experiment_dag_sha256.hex,
            "no_claims": True,
            "qualification_status": "NOT_REQUESTED",
            "historical_continuation_authority": (
                dependencies.historical_continuation_authority
            ),
        }
        closeout_path = _publish_canonical_once_v1(
            closeout / "operator_closeout.json", closeout_value
        )
        lifecycle_closeout = close_expedition_lifecycle_v1(lifecycle_context)
        return ExpeditionOperatorResultV1(
            runtime_root=runtime_root,
            start_record=start_record,
            loop_result=loop_result,
            atlas_path=atlas_path,
            retained_parent_bank_path=bank_path,
            journal_path=journal_path,
            human_review_path=human_review_path,
            closeout_path=closeout_path,
            lifecycle_closeout=lifecycle_closeout,
            experiment_dag_path=experiment_dag_path,
        )
    finally:
        release_expedition_storage_v1(lifecycle_context)


__all__ = (
    "DEFAULT_EXPEDITION_RUNTIME_PARENT_V1",
    "ExpeditionOperatorDependenciesV1",
    "ExpeditionOperatorError",
    "ExpeditionOperatorResultV1",
    "OPERATOR_CLOSEOUT_SCHEMA_VERSION_V1",
    "OPERATOR_START_SCHEMA_VERSION_V1",
    "resolve_founder_request_v1",
    "run_expedition_operator_v1",
)
