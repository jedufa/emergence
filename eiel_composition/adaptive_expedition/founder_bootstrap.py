"""Build one manifest-bound exact founder ObservationFrameV1 artifact.

The builder consumes only a completed canonical Workhorse regular row and the
existing Expedition manifest fields.  It publishes no descriptor
interpretation and creates no execution, science-claim, or Qualification
authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Final, Mapping

from emergence_contracts import canonical_json_bytes
from emergence_contracts.adaptive_science_expedition_v1 import (
    ExpeditionManifestV1,
    ObservationFrameV1,
)
from emergence_contracts.artifacts import RunArtifactRefV1
from emergence_contracts.canonical import canonical_sha256
from emergence_contracts.ids import ArtifactId, SchemaVersion
from emergence_contracts.payloads import CanonicalPayloadV1
from engine.perf.cuda_full_evolution_v2.workhorse.contract import (
    FullEvolutionCudaWorkhorseContractV2,
)
from engine.perf.cuda_full_evolution_v2.workhorse.scientific_control import (
    validate_workhorse_active_scientific_control_state_v1,
)
from engine.perf.cuda_full_evolution_v2.workhorse.selected_world_observation import (
    WorkhorseSelectedWorldObservationError,
    project_workhorse_exact_state_observation_v1,
)

from .continuation_adapter import (
    SelectedWorldBranchError,
    materialize_workhorse_founder_exact_state_ref_v1,
    resolve_workhorse_selected_world_continuation_v1,
    workhorse_selected_world_authority_ref_v1,
)
from .general_closed_loop import EXACT_OBSERVATION_FRAME_ARTIFACT_KIND_V1


FOUNDER_EXACT_OBSERVATION_ADAPTER_ID_V1: Final[str] = (
    "EIEL_WORKHORSE_EXACT_FOUNDER_OBSERVATION_ADAPTER_V1"
)
_REDUCER_IMPLEMENTATION_DESCRIPTOR_V1: Final[Mapping[str, object]] = {
    "schema_version": "eiel_workhorse_exact_founder_reducer_descriptor_v1",
    "adapter_id": FOUNDER_EXACT_OBSERVATION_ADAPTER_ID_V1,
    "authenticated_facts": [
        "active_count",
        "capacity",
        "birth_count",
        "death_count",
        "integrity_ok",
        "terminal_absolute_tick",
    ],
    "unavailable_fact_policy": "OMIT_NEVER_SYNTHESIZE",
    "descriptor_interpretation": "NONE_POLICY_PROFILE_OWNS_INTERPRETATION",
    "named_phenotype_reward_authority": False,
}
FOUNDER_EXACT_REDUCER_IMPLEMENTATION_SHA256_V1: Final = canonical_sha256(
    _REDUCER_IMPLEMENTATION_DESCRIPTOR_V1
)


class FounderBootstrapError(ValueError):
    """The exact founder frame cannot be derived without inventing authority."""


def _fail(message: str) -> None:
    raise FounderBootstrapError(message)


def _read_object(path: Path, *, field: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        _fail(f"{field} is not a regular file")
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FounderBootstrapError(f"{field} is not valid JSON") from error
    if type(value) is not dict:
        _fail(f"{field} must contain an object")
    return value, payload


def _load_manifest(path: str | Path) -> ExpeditionManifestV1:
    requested = Path(path).expanduser()
    value, _payload = _read_object(requested, field="Expedition manifest")
    try:
        return ExpeditionManifestV1.from_dict(value)
    except (KeyError, TypeError, ValueError) as error:
        raise FounderBootstrapError("Expedition manifest contract is invalid") from error


def _authenticated_founder_contract_v1(
    batch_manifest_path: Path,
) -> tuple[FullEvolutionCudaWorkhorseContractV2, str, str]:
    requested = batch_manifest_path.expanduser()
    if requested.is_symlink():
        _fail("founder batch manifest path is invalid")
    try:
        path = requested.resolve(strict=True)
    except OSError as error:
        raise FounderBootstrapError("founder batch manifest is unavailable") from error
    if path.name != "manifest.json" or path.parent.parent.name != "batches":
        _fail("founder batch manifest is outside a canonical run")
    run_root = path.parent.parent.parent.resolve(strict=True)
    batch, batch_bytes = _read_object(path, field="founder batch manifest")
    plan_path = run_root / "execution_plan.json"
    plan, _plan_bytes = _read_object(plan_path, field="founder execution plan")
    plan_body = {key: item for key, item in plan.items() if key != "plan_sha256"}
    analysis = plan.get("analysis")
    extension = analysis.get("extension") if isinstance(analysis, Mapping) else None
    operation = (
        extension.get("contract") if isinstance(extension, Mapping) else None
    )
    execution = plan.get("execution")
    if (
        batch.get("schema_version") != "eiel_campaign_batch_manifest_v1"
        or batch.get("status") != "completed"
        or batch.get("provider_id")
        != "eiel-full-evolution-cuda-workhorse-v2-provider-v1"
        or batch.get("batch_id") != path.parent.name
        or batch.get("plan_sha256") != plan.get("plan_sha256")
        or plan.get("plan_sha256") != canonical_sha256(plan_body).hex
        or not isinstance(operation, Mapping)
        or operation.get("operation") != "EXECUTION_ROW"
        or not isinstance(execution, Mapping)
    ):
        _fail("founder batch does not bind a completed regular Workhorse plan")
    try:
        contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
            operation["contract"]
        )
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise FounderBootstrapError("founder Workhorse contract is invalid") from error
    terminal_tick = contract.start_next_tick + contract.total_ticks
    if (
        contract.start_next_tick != 0
        or execution.get("acs_checkpoint_slots") != 2
        or execution.get("checkpoint_ticks")
        != [contract.temporal_k, terminal_tick]
    ):
        _fail("founder observation requires the exact ACS2 bootstrap schedule")
    return (
        contract,
        hashlib.sha256(batch_bytes).hexdigest(),
        str(plan["plan_sha256"]),
    )


def _publish_canonical_once(path: Path, value: object) -> Path:
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.is_symlink():
        _fail("founder observation destination contains a symbolic link")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            _fail("recovered founder observation artifact differs")
        return path.resolve()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    path.chmod(0o444)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return path.resolve()


def _trajectory_id_v1(
    *,
    manifest: ExpeditionManifestV1,
    batch_file_sha256: str,
    world_uid: int,
) -> str:
    identity = canonical_sha256(
        {
            "schema_version": "eiel_workhorse_exact_founder_trajectory_identity_v1",
            "expedition_id": manifest.expedition_id,
            "observation_profile_id": manifest.observation_profile_id,
            "observation_descriptor_set_sha256": (
                manifest.observation_descriptor_set_sha256.to_dict()
            ),
            "batch_manifest_file_sha256": batch_file_sha256,
            "world_uid": world_uid,
        }
    )
    return f"trajectory-founder-exact-{identity.hex[:32]}"


def materialize_workhorse_founder_exact_observation_frame_v1(
    *,
    manifest_path: str | Path,
    batch_manifest_path: str | Path,
    destination_root: str | Path,
    world_uid: int,
) -> tuple[ObservationFrameV1, RunArtifactRefV1, Path, Path]:
    """Publish one exact founder frame and its existing artifact-ref contract."""

    if type(world_uid) is not int or world_uid < 0:
        _fail("founder world UID must be a nonnegative integer")
    manifest = _load_manifest(manifest_path)
    batch_path = Path(batch_manifest_path).expanduser()
    contract, batch_file_sha256, plan_sha256 = _authenticated_founder_contract_v1(
        batch_path
    )
    trajectory_id = _trajectory_id_v1(
        manifest=manifest,
        batch_file_sha256=batch_file_sha256,
        world_uid=world_uid,
    )
    root = Path(destination_root).expanduser().absolute()
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        _fail("founder observation destination root is invalid")
    root.mkdir(parents=True, exist_ok=True)
    try:
        state, authority, published = materialize_workhorse_founder_exact_state_ref_v1(
            batch_path,
            root / "selected-world",
            world_uid=world_uid,
            expedition_id=manifest.expedition_id,
            world_trajectory_id=trajectory_id,
        )
        expected_authority = workhorse_selected_world_authority_ref_v1(state)
        reopened = resolve_workhorse_selected_world_continuation_v1(
            state_ref=state,
            authority_receipt_ref=expected_authority,
        )
    except SelectedWorldBranchError as error:
        raise FounderBootstrapError(
            f"founder exact StateRef cannot publish/reopen: {error}"
        ) from error
    reopened_contract, reopened_batch_sha256, reopened_plan_sha256 = (
        _authenticated_founder_contract_v1(batch_path)
    )
    if (
        reopened_contract != contract
        or reopened_batch_sha256 != batch_file_sha256
        or reopened_plan_sha256 != plan_sha256
        or _load_manifest(manifest_path) != manifest
    ):
        _fail("founder source/manifest changed during exact observation publication")
    if authority != expected_authority or published != reopened:
        _fail("founder StateRef publisher returned inconsistent authority")
    if (
        state.world_trajectory_id != trajectory_id
        or state.expedition_id != manifest.expedition_id
        or state.logical_world_id != world_uid
        or state.next_absolute_tick
        != contract.start_next_tick + contract.total_ticks
    ):
        _fail("founder exact StateRef differs from its manifest/source identity")
    source_path = Path(reopened.source_object_path)
    if source_path.is_symlink():
        _fail("founder selected state is a symbolic link")
    selected_root = (root / "selected-world").resolve()
    try:
        resolved_source = source_path.resolve(strict=True)
        resolved_source.relative_to(selected_root)
    except (OSError, ValueError) as error:
        raise FounderBootstrapError(
            "founder selected state is outside its retained package"
        ) from error
    if resolved_source.is_symlink() or not resolved_source.is_file():
        _fail("founder selected state is not a regular file")
    state_bytes = resolved_source.read_bytes()
    try:
        world = project_workhorse_exact_state_observation_v1(
            state_bytes,
            contract=contract,
            world_uid=world_uid,
            expected_state_sha256=state.exact_state_sha256.hex,
        )
        active_controls = validate_workhorse_active_scientific_control_state_v1(
            dict(reopened.active_scientific_control_state or {})
        )
    except (WorkhorseSelectedWorldObservationError, ValueError) as error:
        raise FounderBootstrapError(
            f"founder authenticated observation cannot project: {error}"
        ) from error
    counters = world["cumulative_counters"]
    facts = {
        "active_count": int(world["active_population"]),
        "capacity": int(contract.shape.capacity),
        "birth_count": int(counters["birth_count"]),
        "death_count": int(counters["death_count"]),
        "integrity_ok": True,
        "terminal_absolute_tick": state.next_absolute_tick,
        "authenticated_fact_scope": (
            "EXACT_TERMINAL_STATE_HEADER_AND_LAYOUT_ONLY"
        ),
        "unavailable_authenticated_facts": ["maximum_lineage_depth"],
        "source_state_ref_sha256": state.state_ref_sha256().to_dict(),
    }
    profile_descriptor = {
        "schema_version": "eiel_workhorse_exact_founder_observation_profile_v1",
        "adapter_id": FOUNDER_EXACT_OBSERVATION_ADAPTER_ID_V1,
        "observation_profile_id": manifest.observation_profile_id,
        "observation_descriptor_set_sha256": (
            manifest.observation_descriptor_set_sha256.to_dict()
        ),
        "projection": {
            "world_state": "authenticated_exact_terminal_copy",
            "control_readback": "authenticated_provider_active_control_state",
            "policy_facts": list(
                _REDUCER_IMPLEMENTATION_DESCRIPTOR_V1["authenticated_facts"]
            ),
            "descriptor_interpretation": "none",
            "state_authority": "branchable_exact_state_ref",
        },
    }
    frame = ObservationFrameV1(
        expedition_id=manifest.expedition_id,
        world_trajectory_id=trajectory_id,
        logical_world_id=world_uid,
        next_absolute_tick=state.next_absolute_tick,
        semantic_world_sha256=state.semantic_world_sha256,
        control_history_sha256=state.control_history_sha256,
        observation_profile_id=manifest.observation_profile_id,
        observation_profile_sha256=canonical_sha256(profile_descriptor),
        observation_descriptor_set_sha256=(
            manifest.observation_descriptor_set_sha256
        ),
        reducer_implementation_sha256=(
            FOUNDER_EXACT_REDUCER_IMPLEMENTATION_SHA256_V1
        ),
        observations=CanonicalPayloadV1.from_value(
            SchemaVersion("eiel_workhorse_exact_founder_observations_internal_v1"),
            {
                "claim_class": "TECHNICAL_NO_CLAIMS_ENGINEERING_OBSERVATION",
                "qualification_authority_granted": False,
                "named_phenotype_reward_authority_granted": False,
                "world_observation": world,
                "source_binding": {
                    "batch_manifest_file_sha256": batch_file_sha256,
                    "execution_plan_sha256": plan_sha256,
                    "state_ref_sha256": state.state_ref_sha256().to_dict(),
                    "checkpoint_occurrence_sha256": (
                        state.checkpoint_occurrence_sha256.to_dict()
                    ),
                    "event_prefix_sha256": state.event_prefix_sha256.to_dict(),
                },
            },
        ),
        control_readback=CanonicalPayloadV1.from_value(
            SchemaVersion("eiel_workhorse_exact_founder_control_readback_internal_v1"),
            {
                "coverage": "AUTHENTICATED_PROVIDER_ACTIVE_CONTROL_STATE",
                "controls": dict(active_controls["controls"]),
                "active_scientific_control_state": active_controls,
                "branch_authority": False,
                "state_ref_owns_branch_authority": True,
            },
        ),
        policy_facing_facts=CanonicalPayloadV1.from_value(
            SchemaVersion("eiel_workhorse_exact_founder_policy_facts_internal_v1"),
            facts,
        ),
        descriptor_values=CanonicalPayloadV1.from_value(
            SchemaVersion("eiel_uninterpreted_descriptor_values_v1"),
            {
                "status": "NOT_INTERPRETED_BY_GENERIC_ADAPTER",
                "descriptor_set_sha256": (
                    manifest.observation_descriptor_set_sha256.to_dict()
                ),
            },
        ),
        calibration=CanonicalPayloadV1.from_value(
            SchemaVersion("eiel_manifest_bound_calibration_reference_v1"),
            {
                "status": "OWNED_BY_FROZEN_POLICY_PROFILE",
                "descriptor_set_sha256": (
                    manifest.observation_descriptor_set_sha256.to_dict()
                ),
                "adapter_applied_calibration": False,
            },
        ),
        source_state_ref=state,
        previous_frame_sha256=None,
    )
    frame_path = _publish_canonical_once(root / "observation-frame.json", frame.to_dict())
    frame_bytes = frame_path.read_bytes()
    if frame_bytes != canonical_json_bytes(frame.to_dict()):
        _fail("founder observation frame changed before artifact-ref publication")
    artifact_ref = RunArtifactRefV1(
        artifact_id=ArtifactId(
            f"expedition-founder-frame-{frame.observation_frame_sha256().hex[:32]}"
        ),
        artifact_kind=EXACT_OBSERVATION_FRAME_ARTIFACT_KIND_V1,
        artifact_schema_version=SchemaVersion(
            EXACT_OBSERVATION_FRAME_ARTIFACT_KIND_V1
        ),
        locator=frame_path.as_uri(),
        content_digest=frame.observation_frame_sha256(),
        size_bytes=len(frame_bytes),
    )
    ref_path = _publish_canonical_once(
        root / "observation-frame.ref.json", artifact_ref.to_dict()
    )
    return frame, artifact_ref, frame_path, ref_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--batch-manifest", required=True)
    parser.add_argument("--destination-root", required=True)
    parser.add_argument("--world-uid", type=int, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    frame, artifact_ref, frame_path, ref_path = (
        materialize_workhorse_founder_exact_observation_frame_v1(
            manifest_path=args.manifest,
            batch_manifest_path=args.batch_manifest,
            destination_root=args.destination_root,
            world_uid=args.world_uid,
        )
    )
    print(
        json.dumps(
            {
                "observation_frame_path": str(frame_path),
                "observation_frame_sha256": frame.observation_frame_sha256().hex,
                "artifact_ref_path": str(ref_path),
                "founder_basis_ref": artifact_ref.to_dict(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "FOUNDER_EXACT_OBSERVATION_ADAPTER_ID_V1",
    "FOUNDER_EXACT_REDUCER_IMPLEMENTATION_SHA256_V1",
    "FounderBootstrapError",
    "materialize_workhorse_founder_exact_observation_frame_v1",
)
