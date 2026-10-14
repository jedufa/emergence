"""Compatibility views over authoritative CheckpointBundleV2 and RNGManager.

No state is copied into a second owner.  Artifact references point at existing
bundle material and restore delegates to the current strict loader/apply seams.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from emergence_contracts import (
    ArtifactId,
    BranchId,
    CanonicalPayloadV1,
    CheckpointId,
    CompositeCheckpointViewV1,
    ContentDigest,
    ContractError,
    ContractErrorCategory,
    ModuleId,
    ParticipantId,
    RNGLeaseDescriptorV1,
    RunArtifactRefV1,
    RunCursorV1,
    RunId,
    SchemaVersion,
    SnapshotParticipantV1,
    StateMaterialRoleV1,
    StateParticipantDescriptorV1,
)


LEGACY_BUNDLE_SCHEMA = SchemaVersion("checkpoint_bundle_v2")
LEGACY_CONTROL_POSITION_SCHEMA = SchemaVersion("checkpoint_bundle_v2_control_position_v1")
LEGACY_COMPATIBILITY_SCHEMA = SchemaVersion("checkpoint_bundle_v2_compatibility_view_v1")
LEGACY_FILE_SCHEMA = SchemaVersion("checkpoint_bundle_v2_file_v1")


def _state_error(
    category: ContractErrorCategory,
    message: str,
    *,
    details: Mapping[str, object] | None = None,
) -> ContractError:
    return ContractError(
        category,
        owner="eiel_composition.adapters.legacy_state",
        message=message,
        details=details,
    )


def _sha256(path: Path) -> ContentDigest:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return ContentDigest("sha256", digest.hexdigest())


def _file_schema(path: Path) -> SchemaVersion:
    if path.suffix == ".json":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            schema = value.get("schema_version") if isinstance(value, dict) else None
            if isinstance(schema, str) and schema:
                return SchemaVersion(schema)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ContractError):
            pass
    return LEGACY_FILE_SCHEMA


def _artifact_ref(root: Path, path: Path) -> RunArtifactRefV1:
    relative = path.relative_to(root).as_posix()
    return RunArtifactRefV1(
        artifact_id=ArtifactId(f"checkpoint-bundle-v2:{relative}"),
        artifact_kind="checkpoint_bundle_v2_file",
        artifact_schema_version=_file_schema(path),
        locator=str(path.resolve()),
        content_digest=_sha256(path),
        size_bytes=path.stat().st_size,
    )


@dataclass(frozen=True, slots=True)
class LegacySnapshotParticipantView:
    """One existing bundle file; independent import is intentionally disabled."""

    participant_descriptor: StateParticipantDescriptorV1
    payload_ref: RunArtifactRefV1

    def descriptor(self) -> StateParticipantDescriptorV1:
        return self.participant_descriptor

    def export_state(self, checkpoint_context: CanonicalPayloadV1) -> RunArtifactRefV1:
        self.validate_state(self.payload_ref)
        return self.payload_ref

    def validate_state(self, payload_ref: RunArtifactRefV1) -> None:
        path = Path(payload_ref.locator)
        if not path.is_file() or _sha256(path) != payload_ref.content_digest:
            raise _state_error(
                ContractErrorCategory.STATE_INTEGRITY_FAILURE,
                "checkpoint participant artifact is absent or changed",
                details={"locator": payload_ref.locator},
            )

    def import_state(
        self, payload_ref: RunArtifactRefV1, restore_context: CanonicalPayloadV1
    ) -> None:
        raise _state_error(
            ContractErrorCategory.CHECKPOINT_INCOMPATIBLE,
            "CheckpointBundleV2 participants cannot be restored independently",
        )

    def state_digest(self, payload_ref: RunArtifactRefV1) -> ContentDigest:
        self.validate_state(payload_ref)
        return payload_ref.content_digest

    def compatibility_check(self, payload_metadata: CanonicalPayloadV1) -> None:
        if payload_metadata.schema_version != LEGACY_COMPATIBILITY_SCHEMA:
            raise _state_error(
                ContractErrorCategory.CHECKPOINT_INCOMPATIBLE,
                "participant requires the CheckpointBundleV2 compatibility view",
            )


class LegacyCompositeCheckpointAdapter:
    """Strict, read-only C-080 view plus delegated whole-bundle restore."""

    _CORE = frozenset({"metadata.json", "rng_snapshot_v2.json", "control_plane.json"})

    def __init__(self, bundle_dir: str | Path) -> None:
        self.bundle_dir = Path(bundle_dir).resolve()

    def load(self, *, strict: bool = True) -> Mapping[str, object]:
        from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_load

        return checkpoint_bundle_v2_load(str(self.bundle_dir), strict=strict)

    def apply(
        self,
        simulation: object,
        *,
        restore_rng: bool = True,
    ) -> None:
        from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply

        checkpoint_bundle_v2_apply(
            simulation,
            self.load(strict=True),
            restore_rng=restore_rng,
        )

    def describe(
        self,
        *,
        run_id: RunId,
        branch_id: BranchId,
        run_cursor: RunCursorV1,
        assembly_identity: ContentDigest,
    ) -> CompositeCheckpointViewV1:
        # The current loader remains the format/compatibility authority.
        self.load(strict=True)
        files = tuple(
            sorted(
                (path for path in self.bundle_dir.rglob("*") if path.is_file()),
                key=lambda path: path.relative_to(self.bundle_dir).as_posix(),
            )
        )
        by_name = {
            path.relative_to(self.bundle_dir).as_posix(): path for path in files
        }
        try:
            metadata_ref = _artifact_ref(self.bundle_dir, by_name["metadata.json"])
            rng_ref = _artifact_ref(self.bundle_dir, by_name["rng_snapshot_v2.json"])
        except KeyError as exc:
            raise _state_error(
                ContractErrorCategory.CHECKPOINT_INCOMPATIBLE,
                "CheckpointBundleV2 lacks required metadata or RNG material",
            ) from exc

        control_path = by_name.get("control_plane.json")
        if control_path is None:
            control_value: object = {}
        else:
            control_value = json.loads(control_path.read_text(encoding="utf-8"))
        backend_name = next(
            (
                name
                for name in (
                    "native_runtime_checkpoint_projection.json",
                    "native_runtime_checkpoint_projection.json.gz",
                )
                if name in by_name
            ),
            None,
        )
        excluded = set(self._CORE)
        if backend_name is not None:
            excluded.add(backend_name)
        participant_paths = [
            (name, path) for name, path in by_name.items() if name not in excluded
        ]
        participants: list[SnapshotParticipantV1] = []
        file_digests: dict[str, str] = {}
        for order, (name, path) in enumerate(participant_paths):
            ref = _artifact_ref(self.bundle_dir, path)
            file_digests[name] = ref.content_digest.hex
            descriptor = StateParticipantDescriptorV1(
                participant_id=ParticipantId(f"legacy-bundle-file:{name}"),
                # Current Bundle V2 is still the sole integrated owner in P2.
                module_id=ModuleId("eiel"),
                state_schema_version=ref.artifact_schema_version,
                material_role=StateMaterialRoleV1.CONTINUATION_REQUIRED,
                stable_order=order,
            )
            participants.append(
                SnapshotParticipantV1(
                    descriptor=descriptor,
                    payload_ref=ref,
                    payload_digest=ref.content_digest,
                )
            )
        compatibility = CanonicalPayloadV1.from_value(
            LEGACY_COMPATIBILITY_SCHEMA,
            {
                "authoritative_format": str(LEGACY_BUNDLE_SCHEMA),
                "bundle_root": str(self.bundle_dir),
                "participant_file_sha256": file_digests,
                "state_ownership": "current_checkpoint_bundle_v2_integrated_owner",
            },
        )
        return CompositeCheckpointViewV1(
            checkpoint_id=CheckpointId(self.bundle_dir.name),
            run_id=run_id,
            branch_id=branch_id,
            run_cursor=run_cursor,
            runtime_state_ref=metadata_ref,
            rng_root_snapshot_ref=rng_ref,
            control_position=CanonicalPayloadV1.from_value(
                LEGACY_CONTROL_POSITION_SCHEMA, control_value
            ),
            participants=tuple(participants),
            backend_state_ref=(
                None
                if backend_name is None
                else _artifact_ref(self.bundle_dir, by_name[backend_name])
            ),
            assembly_identity=assembly_identity,
            compatibility_metadata=compatibility,
        )


class _LegacyRNGLease:
    def __init__(
        self,
        manager: object,
        request: RNGLeaseDescriptorV1,
        handle: object,
        index_key: str,
    ) -> None:
        self._manager = manager
        self._request = request
        self._handle = handle
        self._index_key = index_key

    def descriptor(self) -> RNGLeaseDescriptorV1:
        return self._request

    def tracked_handle(self) -> object:
        return self._handle

    def index_snapshot(self) -> int:
        return int(self._manager.indices_snapshot().get(self._index_key, 0))

    def assert_budget(self) -> None:
        if self.index_snapshot() < self._request.draw_index:
            raise _state_error(
                ContractErrorCategory.DETERMINISM_VIOLATION,
                "legacy RNG index moved behind its lease boundary",
            )


class LegacyRNGAuthorityAdapter:
    """Borrow existing tracked handles; never instantiate an RNG or draw."""

    def __init__(
        self,
        manager: object,
        *,
        root_snapshot_ref: RunArtifactRefV1,
    ) -> None:
        self._manager = manager
        self._root_snapshot_ref = root_snapshot_ref

    def lease(self, request: RNGLeaseDescriptorV1) -> _LegacyRNGLease:
        try:
            stream, domain = request.stream_identity.split(".", 1)
        except ValueError as exc:
            raise _state_error(
                ContractErrorCategory.CONFIGURATION_INVALID,
                "legacy stream identity must be '<stream>.<domain>'",
            ) from exc
        index_key = request.stream_identity
        current = int(self._manager.indices_snapshot().get(index_key, 0))
        if current != request.draw_index:
            raise _state_error(
                ContractErrorCategory.DETERMINISM_VIOLATION,
                "lease draw index differs from current RNGManager authority",
                details={"requested": request.draw_index, "current": current},
            )
        if domain == "py":
            handle = self._manager.py_rng(stream)
        elif domain == "np":
            handle = self._manager.np_rng(stream)
        elif domain == "torch_cpu":
            handle = self._manager.torch_rng(stream, "cpu")
        elif domain.startswith("torch_cuda."):
            handle = self._manager.torch_rng(stream, domain.removeprefix("torch_cuda."))
        else:
            raise _state_error(
                ContractErrorCategory.CAPABILITY_UNSUPPORTED,
                "legacy RNG domain is unsupported",
                details={"domain": domain},
            )
        if handle is None:
            raise _state_error(
                ContractErrorCategory.RESOURCE_UNAVAILABLE,
                "requested RNG device handle is unavailable",
            )
        return _LegacyRNGLease(self._manager, request, handle, index_key)

    def root_snapshot_ref(self) -> RunArtifactRefV1:
        return self._root_snapshot_ref

    def restore_root_snapshot(self, snapshot_ref: RunArtifactRefV1) -> None:
        path = Path(snapshot_ref.locator)
        if not path.is_file() or _sha256(path) != snapshot_ref.content_digest:
            raise _state_error(
                ContractErrorCategory.STATE_INTEGRITY_FAILURE,
                "RNG snapshot artifact is absent or changed",
            )
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise _state_error(
                ContractErrorCategory.CHECKPOINT_INCOMPATIBLE,
                "RNG snapshot must be an object",
            )
        self._manager.apply_snapshot_dict(value)
