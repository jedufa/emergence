"""Sustained committed-throughput benchmark for direct sparse Full Evolution."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from time import perf_counter
from typing import Any, Final, Sequence

from .benchmark import (
    _fsync_directory_v2,
    _read_exact_descriptor_v2,
    _unlink_exact_inode_v2,
    _write_all_v2,
)
from .contract import (
    CUDA_NATIVE_TARGET_WTPS_100X,
    FULL_EVOLUTION_CAPABILITIES_V2,
    CudaFullEvolutionNativeContractV2,
)
from .direct_sparse_runtime import (
    CudaFullEvolutionDirectSparseOwnerV3,
    PreparedDirectSparseDurableEnvelopeV3,
    VerifiedDirectSparseDurableEnvelopeV3,
    open_verified_prepared_direct_sparse_durable_envelope_v3,
    prepare_direct_sparse_durable_envelope_v3,
    seal_direct_sparse_durable_sink_receipt_v3,
)
from .native_loader import PROJECT_ROOT, canonical_sha256, cuda_device_identity_v2
from .runtime import (
    FULL_CAPACITY_BENCHMARK_FIXTURE_V2,
    CudaFullEvolutionRuntimeError,
    CudaFullEvolutionSparseEventDispatchV3,
)

BENCHMARK_SCHEMA_V2: Final[str] = (
    "eiel_cuda_full_evolution_direct_sparse_sustained_benchmark_v2"
)
CLAIM_CLASS: Final[str] = "TECHNICAL_ONLY"
RUNNER_SOURCE_RELATIVE_PATH_V2: Final[str] = (
    "engine/perf/cuda_full_evolution_v2/direct_sparse_sustained_benchmark.py"
)


class DirectSparseSustainedBenchmarkError(RuntimeError):
    """Raised when the direct sparse benchmark loses exact authority."""


class _BoundImmutableInputV2:
    """Hold one exact read-only input inode through result publication."""

    __slots__ = (
        "__closed",
        "__descriptor",
        "__identity",
        "__label",
        "__mode",
        "__mtime_ns",
        "__ctime_ns",
        "__nbytes",
        "__path",
        "__payload",
        "__sha256",
    )

    def __init__(self, path: Path, *, label: str) -> None:
        if not path.is_absolute() or path.resolve(strict=True) != path:
            raise DirectSparseSustainedBenchmarkError(
                f"{label} path must be canonical, absolute, and symlink-free"
            )
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            descriptor_state = os.fstat(descriptor)
            path_state = os.lstat(path)
            identity = (descriptor_state.st_dev, descriptor_state.st_ino)
            payload = _read_exact_descriptor_v2(descriptor, descriptor_state.st_size)
            mode = stat.S_IMODE(descriptor_state.st_mode)
            if (
                not stat.S_ISREG(descriptor_state.st_mode)
                or not stat.S_ISREG(path_state.st_mode)
                or (path_state.st_dev, path_state.st_ino) != identity
                or descriptor_state.st_nlink != 1
                or path_state.st_nlink != 1
                or mode != stat.S_IMODE(path_state.st_mode)
                or mode & 0o222
                or descriptor_state.st_size != path_state.st_size
                or descriptor_state.st_mtime_ns != path_state.st_mtime_ns
                or descriptor_state.st_ctime_ns != path_state.st_ctime_ns
            ):
                raise DirectSparseSustainedBenchmarkError(
                    f"{label} is not an exact single-link read-only regular file"
                )
            self.__descriptor = descriptor
            self.__path = path
            self.__label = label
            self.__identity = identity
            self.__mode = mode
            self.__nbytes = descriptor_state.st_size
            self.__mtime_ns = descriptor_state.st_mtime_ns
            self.__ctime_ns = descriptor_state.st_ctime_ns
            self.__payload = payload
            self.__sha256 = hashlib.sha256(payload).hexdigest()
            self.__closed = False
        except Exception:
            os.close(descriptor)
            raise

    @property
    def sha256(self) -> str:
        return self.__sha256

    @property
    def payload(self) -> bytes:
        self.verify()
        return self.__payload

    def verify(self) -> None:
        if self.__closed:
            raise DirectSparseSustainedBenchmarkError(
                f"{self.__label} binding is closed"
            )
        try:
            resolved_path = self.__path.resolve(strict=True)
            descriptor_state = os.fstat(self.__descriptor)
            path_state = os.lstat(self.__path)
            payload = _read_exact_descriptor_v2(self.__descriptor, self.__nbytes)
        except OSError as exc:
            raise DirectSparseSustainedBenchmarkError(
                f"{self.__label} became unavailable: {exc}"
            ) from exc
        if (
            resolved_path != self.__path
            or (descriptor_state.st_dev, descriptor_state.st_ino) != self.__identity
            or (path_state.st_dev, path_state.st_ino) != self.__identity
            or descriptor_state.st_nlink != 1
            or path_state.st_nlink != 1
            or stat.S_IMODE(descriptor_state.st_mode) != self.__mode
            or stat.S_IMODE(path_state.st_mode) != self.__mode
            or self.__mode & 0o222
            or descriptor_state.st_size != self.__nbytes
            or path_state.st_size != self.__nbytes
            or descriptor_state.st_mtime_ns != self.__mtime_ns
            or path_state.st_mtime_ns != self.__mtime_ns
            or descriptor_state.st_ctime_ns != self.__ctime_ns
            or path_state.st_ctime_ns != self.__ctime_ns
            or hashlib.sha256(payload).hexdigest() != self.__sha256
        ):
            raise DirectSparseSustainedBenchmarkError(
                f"{self.__label} changed during the benchmark transaction"
            )

    def projection(self) -> dict[str, Any]:
        self.verify()
        return {
            "path": str(self.__path),
            "sha256": self.__sha256,
            "device": self.__identity[0],
            "inode": self.__identity[1],
            "bytes": self.__nbytes,
            "mode": oct(self.__mode),
            "link_count": 1,
        }

    def close(self) -> None:
        if not self.__closed:
            os.close(self.__descriptor)
            self.__closed = True


class _BoundArtifactManifestV2:
    """Bind the exact native artifact and manifest used by every sample."""

    __slots__ = ("__artifact", "__manifest", "__manifest_contract_sha256")

    def __init__(self, artifact_path: Path) -> None:
        self.__artifact = _BoundImmutableInputV2(artifact_path, label="native artifact")
        try:
            self.__manifest = _BoundImmutableInputV2(
                artifact_path.with_suffix(artifact_path.suffix + ".manifest.json"),
                label="native artifact manifest",
            )
            parsed = json.loads(self.__manifest.payload.decode("utf-8"))
            if not isinstance(parsed, dict):
                raise DirectSparseSustainedBenchmarkError(
                    "native artifact manifest is not a JSON object"
                )
            manifest_contract_sha256 = parsed.get("manifest_sha256")
            manifest_body = {
                key: value for key, value in parsed.items() if key != "manifest_sha256"
            }
            output = parsed.get("output")
            if (
                not isinstance(manifest_contract_sha256, str)
                or len(manifest_contract_sha256) != 64
                or canonical_sha256(manifest_body) != manifest_contract_sha256
                or not isinstance(output, dict)
                or output.get("sha256") != self.__artifact.sha256
            ):
                raise DirectSparseSustainedBenchmarkError(
                    "native artifact manifest identity is invalid"
                )
            self.__manifest_contract_sha256 = manifest_contract_sha256
        except Exception:
            if hasattr(self, "_BoundArtifactManifestV2__manifest"):
                self.__manifest.close()
            self.__artifact.close()
            raise

    def verify(
        self,
        *,
        artifact_sha256: str | None = None,
        manifest_sha256: str | None = None,
    ) -> None:
        self.__artifact.verify()
        self.__manifest.verify()
        if (
            artifact_sha256 is not None and artifact_sha256 != self.__artifact.sha256
        ) or (
            manifest_sha256 is not None
            and manifest_sha256 != self.__manifest_contract_sha256
        ):
            raise DirectSparseSustainedBenchmarkError(
                "runtime artifact identity differs from the held input binding"
            )

    def projection(self) -> dict[str, Any]:
        artifact = self.__artifact.projection()
        manifest = self.__manifest.projection()
        return {
            **artifact,
            "manifest_path": manifest.pop("path"),
            "manifest_sha256": self.__manifest_contract_sha256,
            "manifest_file_sha256": manifest.pop("sha256"),
            "manifest_binding": manifest,
            "binding": "open_descriptors_start_checkpoint_result_equality",
        }

    def close(self) -> None:
        self.__manifest.close()
        self.__artifact.close()

    def __enter__(self) -> "_BoundArtifactManifestV2":
        self.verify()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class _BoundRunnerSourceV2:
    """Hold and rehash the exact manifest-bound runner source through publication."""

    __slots__ = (
        "__closed",
        "__descriptor",
        "__identity",
        "__mode",
        "__mtime_ns",
        "__ctime_ns",
        "__nbytes",
        "__path",
        "__sha256",
    )

    def __init__(self, *, artifact_path: Path) -> None:
        manifest_path = artifact_path.with_suffix(
            artifact_path.suffix + ".manifest.json"
        )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DirectSparseSustainedBenchmarkError(
                f"cannot read the native manifest for runner binding: {exc}"
            ) from exc
        rows = manifest.get("source_inventory")
        matches = (
            [row for row in rows if row.get("path") == RUNNER_SOURCE_RELATIVE_PATH_V2]
            if isinstance(rows, list) and all(isinstance(row, dict) for row in rows)
            else []
        )
        if len(matches) != 1:
            raise DirectSparseSustainedBenchmarkError(
                "native manifest does not bind the exact sustained runner source"
            )
        expected_sha256 = matches[0].get("sha256")
        path = (PROJECT_ROOT / RUNNER_SOURCE_RELATIVE_PATH_V2).resolve(strict=True)
        if path != Path(__file__).resolve(strict=True):
            raise DirectSparseSustainedBenchmarkError(
                "imported sustained runner path is not the manifest-bound source"
            )
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            descriptor_state = os.fstat(descriptor)
            path_state = os.lstat(path)
            identity = (descriptor_state.st_dev, descriptor_state.st_ino)
            payload = _read_exact_descriptor_v2(descriptor, descriptor_state.st_size)
            actual_sha256 = hashlib.sha256(payload).hexdigest()
            if (
                path.resolve(strict=True) != path
                or not stat.S_ISREG(descriptor_state.st_mode)
                or not stat.S_ISREG(path_state.st_mode)
                or (path_state.st_dev, path_state.st_ino) != identity
                or descriptor_state.st_nlink != 1
                or path_state.st_nlink != 1
                or descriptor_state.st_size != path_state.st_size
                or descriptor_state.st_mtime_ns != path_state.st_mtime_ns
                or descriptor_state.st_ctime_ns != path_state.st_ctime_ns
                or actual_sha256 != expected_sha256
            ):
                raise DirectSparseSustainedBenchmarkError(
                    "sustained runner source does not match the sealed manifest row"
                )
            self.__descriptor = descriptor
            self.__path = path
            self.__identity = identity
            self.__mode = stat.S_IMODE(descriptor_state.st_mode)
            self.__nbytes = descriptor_state.st_size
            self.__mtime_ns = descriptor_state.st_mtime_ns
            self.__ctime_ns = descriptor_state.st_ctime_ns
            self.__sha256 = actual_sha256
            self.__closed = False
        except Exception:
            os.close(descriptor)
            raise

    def verify(self) -> None:
        if self.__closed:
            raise DirectSparseSustainedBenchmarkError(
                "sustained runner source binding is closed"
            )
        try:
            descriptor_state = os.fstat(self.__descriptor)
            path_state = os.lstat(self.__path)
            payload = _read_exact_descriptor_v2(self.__descriptor, self.__nbytes)
        except OSError as exc:
            raise DirectSparseSustainedBenchmarkError(
                f"sustained runner source became unavailable: {exc}"
            ) from exc
        if (
            self.__path.resolve(strict=True) != self.__path
            or (descriptor_state.st_dev, descriptor_state.st_ino) != self.__identity
            or (path_state.st_dev, path_state.st_ino) != self.__identity
            or descriptor_state.st_nlink != 1
            or path_state.st_nlink != 1
            or stat.S_IMODE(descriptor_state.st_mode) != self.__mode
            or stat.S_IMODE(path_state.st_mode) != self.__mode
            or descriptor_state.st_size != self.__nbytes
            or path_state.st_size != self.__nbytes
            or descriptor_state.st_mtime_ns != self.__mtime_ns
            or path_state.st_mtime_ns != self.__mtime_ns
            or descriptor_state.st_ctime_ns != self.__ctime_ns
            or path_state.st_ctime_ns != self.__ctime_ns
            or hashlib.sha256(payload).hexdigest() != self.__sha256
        ):
            raise DirectSparseSustainedBenchmarkError(
                "sustained runner source changed during the benchmark transaction"
            )

    def projection(self) -> dict[str, Any]:
        self.verify()
        return {
            "path": str(self.__path),
            "relative_path": RUNNER_SOURCE_RELATIVE_PATH_V2,
            "sha256": self.__sha256,
            "device": self.__identity[0],
            "inode": self.__identity[1],
            "bytes": self.__nbytes,
            "mode": oct(self.__mode),
            "binding": "manifest_row_plus_open_descriptor_start_end_equality",
        }

    def close(self) -> None:
        if not self.__closed:
            os.close(self.__descriptor)
            self.__closed = True

    def __enter__(self) -> "_BoundRunnerSourceV2":
        self.verify()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _publish_immutable_bytes(
    payload: bytes,
    *,
    output: Path,
    temporary_name: str,
    expected_sha256: str | None = None,
    defer_content_verification_to_canonical_reopen: bool = False,
) -> dict[str, Any]:
    if expected_sha256 is not None and (
        len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise DirectSparseSustainedBenchmarkError(
            "immutable publication expected SHA-256 is invalid"
        )
    if defer_content_verification_to_canonical_reopen and expected_sha256 is None:
        raise DirectSparseSustainedBenchmarkError(
            "deferred immutable content verification requires a prepared SHA-256"
        )
    temporary = output.parent / temporary_name
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    created_identity: tuple[int, int] | None = None
    linked_identity: tuple[int, int] | None = None
    temporary_created = False
    output_linked = False
    try:
        descriptor = os.open(temporary, flags, stat.S_IRUSR)
        temporary_created = True
        created = os.fstat(descriptor)
        created_identity = (created.st_dev, created.st_ino)
        if not stat.S_ISREG(created.st_mode) or created.st_nlink != 1:
            raise DirectSparseSustainedBenchmarkError(
                "exclusive temporary is not a single-link regular file"
            )
        _write_all_v2(descriptor, payload)
        os.fchmod(descriptor, stat.S_IRUSR)
        os.fsync(descriptor)
        sealed = os.fstat(descriptor)
        temporary_state = os.lstat(temporary)
        if (
            (sealed.st_dev, sealed.st_ino) != created_identity
            or (temporary_state.st_dev, temporary_state.st_ino) != created_identity
            or sealed.st_nlink != 1
            or temporary_state.st_nlink != 1
            or stat.S_IMODE(sealed.st_mode) != stat.S_IRUSR
            or stat.S_IMODE(temporary_state.st_mode) != stat.S_IRUSR
            or sealed.st_size != len(payload)
            or temporary_state.st_size != len(payload)
            or (
                not defer_content_verification_to_canonical_reopen
                and _read_exact_descriptor_v2(descriptor, len(payload)) != payload
            )
        ):
            raise DirectSparseSustainedBenchmarkError(
                "immutable temporary changed before publication"
            )
        try:
            os.link(temporary, output, follow_symlinks=False)
            output_linked = True
        except FileExistsError as exc:
            raise DirectSparseSustainedBenchmarkError(
                "immutable publication cannot replace existing evidence"
            ) from exc
        output_state = os.lstat(output)
        linked_identity = (output_state.st_dev, output_state.st_ino)
        temporary_state = os.lstat(temporary)
        linked = os.fstat(descriptor)
        if (
            (linked.st_dev, linked.st_ino) != created_identity
            or (temporary_state.st_dev, temporary_state.st_ino) != created_identity
            or (output_state.st_dev, output_state.st_ino) != created_identity
            or linked.st_nlink != 2
            or temporary_state.st_nlink != 2
            or output_state.st_nlink != 2
            or stat.S_IMODE(linked.st_mode) != stat.S_IRUSR
            or stat.S_IMODE(temporary_state.st_mode) != stat.S_IRUSR
            or stat.S_IMODE(output_state.st_mode) != stat.S_IRUSR
            or linked.st_size != len(payload)
            or temporary_state.st_size != len(payload)
            or output_state.st_size != len(payload)
        ):
            raise DirectSparseSustainedBenchmarkError(
                "no-replace publication did not preserve the created inode"
            )
        if not _unlink_exact_inode_v2(temporary, created_identity):
            raise DirectSparseSustainedBenchmarkError(
                "temporary identity changed before publication closeout"
            )
        temporary_created = False
        _fsync_directory_v2(output.parent)
        final_state = os.lstat(output)
        final_descriptor_state = os.fstat(descriptor)
        if (
            (final_state.st_dev, final_state.st_ino) != created_identity
            or (final_descriptor_state.st_dev, final_descriptor_state.st_ino)
            != created_identity
            or final_state.st_nlink != 1
            or final_descriptor_state.st_nlink != 1
            or stat.S_IMODE(final_state.st_mode) != stat.S_IRUSR
            or stat.S_IMODE(final_descriptor_state.st_mode) != stat.S_IRUSR
            or final_state.st_size != len(payload)
            or final_descriptor_state.st_size != len(payload)
            or (
                not defer_content_verification_to_canonical_reopen
                and _read_exact_descriptor_v2(descriptor, len(payload)) != payload
            )
        ):
            raise DirectSparseSustainedBenchmarkError(
                "published evidence changed during final verification"
            )
        return {
            "path": str(output.resolve()),
            "bytes": len(payload),
            "sha256": (
                expected_sha256
                if expected_sha256 is not None
                else hashlib.sha256(payload).hexdigest()
            ),
            "mode": oct(stat.S_IMODE(final_state.st_mode)),
            "device": final_state.st_dev,
            "inode": final_state.st_ino,
            "link_count": final_state.st_nlink,
        }
    except Exception:
        if output_linked:
            for identity in (created_identity, linked_identity):
                if identity is not None and _unlink_exact_inode_v2(output, identity):
                    break
        if temporary_created and created_identity is not None:
            _unlink_exact_inode_v2(temporary, created_identity)
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _publish_dispatch(
    dispatch: CudaFullEvolutionSparseEventDispatchV3,
    *,
    output_directory: Path,
    sample_id: str,
) -> tuple[dict[str, Any], VerifiedDirectSparseDurableEnvelopeV3]:
    prepared = prepare_direct_sparse_durable_envelope_v3(dispatch, sample_id=sample_id)
    return _publish_prepared_dispatch(
        prepared, output_directory=output_directory, sample_id=sample_id
    )


def _publish_prepared_dispatch(
    prepared: PreparedDirectSparseDurableEnvelopeV3,
    *,
    output_directory: Path,
    sample_id: str,
) -> tuple[dict[str, Any], VerifiedDirectSparseDurableEnvelopeV3]:
    if type(prepared) is not PreparedDirectSparseDurableEnvelopeV3:
        raise DirectSparseSustainedBenchmarkError(
            "direct sparse publication requires an exact prepared capability"
        )
    if sample_id != prepared.sample_id:
        raise DirectSparseSustainedBenchmarkError(
            "direct sparse publication sample differs from prepared capability"
        )
    payload = prepared.payload_for_publication()
    receipt = _publish_immutable_bytes(
        payload,
        output=output_directory / f"{sample_id}.direct-sparse-events-v3.bin",
        temporary_name=f".{sample_id}.{os.getpid()}.tmp",
        expected_sha256=prepared.content_sha256,
        defer_content_verification_to_canonical_reopen=True,
    )
    try:
        durable_envelope = open_verified_prepared_direct_sparse_durable_envelope_v3(
            prepared,
            durable_path=receipt["path"],
        )
    except Exception:
        removed = _unlink_exact_inode_v2(
            Path(receipt["path"]), (int(receipt["device"]), int(receipt["inode"]))
        )
        if removed:
            _fsync_directory_v2(output_directory)
        raise
    return receipt, durable_envelope


def _aggregate_counters(rows: list[dict[str, int]]) -> dict[str, int]:
    if not rows or any(tuple(row) != FULL_EVOLUTION_CAPABILITIES_V2 for row in rows):
        raise DirectSparseSustainedBenchmarkError(
            "runtime did not expose the exact seven-capability identity"
        )
    return {
        capability: sum(row[capability] for row in rows)
        for capability in FULL_EVOLUTION_CAPABILITIES_V2
    }


def _verify_result_bindings_or_remove_v2(
    result_receipt: dict[str, Any],
    *,
    runner_source: _BoundRunnerSourceV2,
    artifact_binding: _BoundArtifactManifestV2,
) -> None:
    try:
        runner_source.verify()
        artifact_binding.verify()
    except Exception:
        _unlink_exact_inode_v2(
            Path(result_receipt["path"]),
            (int(result_receipt["device"]), int(result_receipt["inode"])),
        )
        raise


def _run_direct_sparse_sustained_benchmark_bound(
    *,
    artifact_path: Path,
    output_directory: Path,
    batch_size: int,
    temporal_k: int,
    samples: int,
    warmups: int,
    seed: int,
    initial_capacity_rows: int | None,
    runner_source: _BoundRunnerSourceV2,
    artifact_binding: _BoundArtifactManifestV2,
) -> dict[str, Any]:
    if batch_size < 1 or not 1 <= temporal_k <= 64:
        raise DirectSparseSustainedBenchmarkError(
            "B must be positive and K must be in [1,64]"
        )
    if samples < 1 or warmups < 0:
        raise DirectSparseSustainedBenchmarkError(
            "samples must be positive and warmups nonnegative"
        )
    if output_directory.exists():
        raise DirectSparseSustainedBenchmarkError(
            "output directory must not already exist"
        )
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(mode=0o700)
    _fsync_directory_v2(output_directory.parent)

    import torch

    device_index = torch.cuda.current_device()
    device_identity = cuda_device_identity_v2(torch, device_index)
    initial_free_vram, total_vram = torch.cuda.mem_get_info(device_index)
    contract = CudaFullEvolutionNativeContractV2(
        B=batch_size, total_ticks=temporal_k, temporal_k=temporal_k
    )
    fixture_owner = CudaFullEvolutionDirectSparseOwnerV3.initialize(
        contract,
        seed=seed,
        artifact_path=artifact_path,
        initial_capacity_rows=initial_capacity_rows,
    )
    try:
        fixture_checkpoint = fixture_owner.force_full_capacity_benchmark_fixture_v2()
        artifact_binding.verify(
            artifact_sha256=fixture_checkpoint.artifact_sha256,
            manifest_sha256=fixture_checkpoint.manifest_sha256,
        )
        fixture_admission = dict(fixture_owner.resource_admission_v3)
    finally:
        fixture_owner.close()
    del fixture_owner
    gc.collect()
    torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    for ordinal in range(warmups + samples):
        measured = ordinal >= warmups
        sample_number = ordinal - warmups if measured else ordinal
        sample_id = (
            f"sample-{sample_number:03d}" if measured else f"warmup-{sample_number:03d}"
        )
        artifact_binding.verify(
            artifact_sha256=fixture_checkpoint.artifact_sha256,
            manifest_sha256=fixture_checkpoint.manifest_sha256,
        )
        owner = CudaFullEvolutionDirectSparseOwnerV3.from_checkpoint(
            fixture_checkpoint,
            artifact_path=artifact_path,
            initial_capacity_rows=initial_capacity_rows,
        )
        wall_started = perf_counter()
        dispatch = owner.advance_and_drain_v3(temporal_k)
        if (
            dispatch.artifact_sha256 != fixture_checkpoint.artifact_sha256
            or dispatch.manifest_sha256 != fixture_checkpoint.manifest_sha256
        ):
            mismatch = DirectSparseSustainedBenchmarkError(
                "sample dispatch differs from the checkpoint-bound artifact"
            )
            try:
                owner.reject_pending_sparse_event_drain_v3(
                    reason="sample dispatch differs from checkpoint-bound artifact"
                )
            except CudaFullEvolutionRuntimeError as rollback_exc:
                raise rollback_exc from mismatch
            raise mismatch
        publication_started = perf_counter()
        prepared = owner.prepare_sparse_event_durable_envelope_v3(
            dispatch, sample_id=sample_id
        )
        envelope: dict[str, Any] | None = None
        durable_envelope: VerifiedDirectSparseDurableEnvelopeV3 | None = None
        try:
            envelope, durable_envelope = _publish_prepared_dispatch(
                prepared, output_directory=output_directory, sample_id=sample_id
            )
            publication_elapsed = perf_counter() - publication_started
            receipt_started = perf_counter()
            receipt = seal_direct_sparse_durable_sink_receipt_v3(
                dispatch,
                sink_id="immutable-local-direct-sparse-envelope-v3",
                sink_receipt_id=f"{Path(envelope['path']).name}:{envelope['sha256']}",
                sample_id=sample_id,
                durable_envelope=durable_envelope,
            )
            receipt_elapsed = perf_counter() - receipt_started
        except Exception as exc:
            if durable_envelope is not None:
                durable_envelope.close()
            if envelope is not None and _unlink_exact_inode_v2(
                Path(envelope["path"]),
                (int(envelope["device"]), int(envelope["inode"])),
            ):
                _fsync_directory_v2(output_directory)
            try:
                owner.reject_pending_sparse_event_drain_v3(
                    reason=(
                        "direct sparse durable publication or receipt seal failed: "
                        f"{type(exc).__name__}"
                    )
                )
            except CudaFullEvolutionRuntimeError as rollback_exc:
                raise rollback_exc from exc
            raise
        assert envelope is not None
        assert durable_envelope is not None
        try:
            acknowledgement_started = perf_counter()
            owner.acknowledge_sparse_event_drain_v3(
                receipt, durable_envelope=durable_envelope
            )
            acknowledgement_elapsed = perf_counter() - acknowledgement_started
        except Exception:
            durable_envelope.close()
            if _unlink_exact_inode_v2(
                Path(envelope["path"]),
                (int(envelope["device"]), int(envelope["inode"])),
            ):
                _fsync_directory_v2(output_directory)
            raise
        finally:
            durable_envelope.close()
        wall_elapsed = perf_counter() - wall_started
        if owner.advanced_ticks != temporal_k or owner.has_pending_dispatch:
            raise DirectSparseSustainedBenchmarkError(
                "direct sparse owner did not commit the measured frontier"
            )
        counters = _aggregate_counters(owner.counters())
        if any(value <= 0 for value in counters.values()):
            raise DirectSparseSustainedBenchmarkError(
                "C512 sample did not exercise all seven capabilities"
            )
        metrics = owner.dispatch_metrics_v3
        if metrics is None:
            raise DirectSparseSustainedBenchmarkError(
                "direct sparse dispatch metrics are missing"
            )
        execution_receipt = owner.execution_receipt_v3()
        if (
            not isinstance(execution_receipt, dict)
            or execution_receipt.get("artifact_sha256")
            != fixture_checkpoint.artifact_sha256
            or execution_receipt.get("manifest_sha256")
            != fixture_checkpoint.manifest_sha256
        ):
            raise DirectSparseSustainedBenchmarkError(
                "sample execution receipt differs from the checkpoint-bound artifact"
            )
        committed_world_ticks = batch_size * temporal_k
        rows.append(
            {
                "sample_id": sample_id,
                "measured": measured,
                "committed_world_ticks": committed_world_ticks,
                "wall_elapsed_seconds": wall_elapsed,
                "committed_world_ticks_per_second": (
                    committed_world_ticks / wall_elapsed
                ),
                "kernel_elapsed_ms_diagnostic_only": (
                    metrics.kernel_elapsed_ms_diagnostic_only
                ),
                "snapshot_host_enqueue_seconds": metrics.snapshot_elapsed_seconds,
                "rollback_elapsed_seconds": metrics.rollback_elapsed_seconds,
                "overflow_retry_count": metrics.retry_count,
                "selected_capacity_rows_per_world": (
                    metrics.selected_capacity_rows_per_world
                ),
                "event_count": dispatch.event_count,
                "events_per_world_tick": dispatch.event_count / committed_world_ticks,
                "sparse_wire_bytes": dispatch.wire_nbytes,
                "event_content_sha256": dispatch.content_sha256,
                "artifact_sha256": fixture_checkpoint.artifact_sha256,
                "manifest_sha256": fixture_checkpoint.manifest_sha256,
                "event_count_projection_seconds": (
                    dispatch.count_projection_elapsed_seconds
                ),
                "event_pack_transfer_validation_seconds": (
                    dispatch.compaction_and_transfer_elapsed_seconds
                ),
                "event_content_hash_seconds": (dispatch.content_hash_elapsed_seconds),
                "event_drain_seconds_including_snapshot_and_kernel_wait": (
                    dispatch.drain_elapsed_seconds
                ),
                "durable_publication_seconds": publication_elapsed,
                "sink_receipt_seal_seconds": receipt_elapsed,
                "acknowledgement_seconds": acknowledgement_elapsed,
                "sink_receipt_sha256": receipt.receipt_sha256,
                "durable_envelope": envelope,
                "capability_counters": counters,
                "peak_allocated_bytes": metrics.peak_allocated_bytes,
                "peak_reserved_bytes": metrics.peak_reserved_bytes,
                "runtime_execution_receipt": execution_receipt,
            }
        )
        owner.close()
        del owner, dispatch
        gc.collect()
        torch.cuda.empty_cache()

    measured_rows = [row for row in rows if row["measured"]]
    throughputs = sorted(
        float(row["committed_world_ticks_per_second"]) for row in measured_rows
    )
    p10 = throughputs[max(0, math.ceil(0.10 * len(throughputs)) - 1)]
    target = max(float(CUDA_NATIVE_TARGET_WTPS_100X), 446_719.311_516)
    runner_source.verify()
    artifact_binding.verify(
        artifact_sha256=fixture_checkpoint.artifact_sha256,
        manifest_sha256=fixture_checkpoint.manifest_sha256,
    )
    result = {
        "schema_version": BENCHMARK_SCHEMA_V2,
        "claim_class": CLAIM_CLASS,
        "status": (
            "FULL_TARGET_PASS"
            if min(throughputs) >= target
            else "FULL_TARGET_NOT_DEMONSTRATED"
        ),
        "target_world_ticks_per_second": target,
        "criterion": (
            "minimum measured wall-clock throughput from immediately before the "
            "full-state rollback snapshot through direct sparse transition, exact "
            "pack/validation/host transfer/hash, immutable inode-bound envelope "
            "fsync/no-replace publication/parent fsync, canonical-path reopen, "
            "structured receipt seal, descriptor-held revalidation, and committed "
            "frontier acknowledgement"
        ),
        "candidate": {
            "batch_size": batch_size,
            "temporal_k": temporal_k,
            "world_width": 32,
            "initial_population": 256,
            "resident_capacity": 512,
            "fixture_id": FULL_CAPACITY_BENCHMARK_FIXTURE_V2,
            "initial_capacity_rows_per_world": initial_capacity_rows,
        },
        "samples": samples,
        "warmups": warmups,
        "minimum_committed_world_ticks_per_second": min(throughputs),
        "p10_committed_world_ticks_per_second": p10,
        "median_committed_world_ticks_per_second": (
            throughputs[len(throughputs) // 2]
            if len(throughputs) % 2
            else (
                throughputs[len(throughputs) // 2 - 1]
                + throughputs[len(throughputs) // 2]
            )
            / 2
        ),
        "maximum_committed_world_ticks_per_second": max(throughputs),
        "all_seven_capabilities_positive_each_sample": all(
            all(value > 0 for value in row["capability_counters"].values())
            for row in measured_rows
        ),
        "full_evolution_capabilities": list(FULL_EVOLUTION_CAPABILITIES_V2),
        "absolute_tick_authority": True,
        "horizon_dependent_resident_bytes": 0,
        "event_transport": {
            "production_wire": "eiel_full_evolution_cuda_sparse_lifecycle_event_v3",
            "production_emission": "direct_per_world_bounded_staging",
            "dense_v2_role": "tiny_gpu_differential_oracle_only",
            "overflow": "full_state_rollback_re_admit_grow_retry",
            "durability": (
                "exact_inode_fsync_no_replace_parent_fsync_reopen_hold_then_ack"
            ),
        },
        "resource_admission": fixture_admission,
        "artifact": artifact_binding.projection(),
        "benchmark_source": runner_source.projection(),
        "fixture_checkpoint": {
            "canonical_state_sha256": fixture_checkpoint.canonical_state_sha256,
            "metadata_seal_sha256": fixture_checkpoint.metadata_seal_sha256,
            "population_fixture_id": fixture_checkpoint.population_fixture_id,
            "artifact_sha256": fixture_checkpoint.artifact_sha256,
            "manifest_sha256": fixture_checkpoint.manifest_sha256,
        },
        "device_identity": device_identity,
        "initial_vram": {
            "free_bytes": int(initial_free_vram),
            "total_bytes": int(total_vram),
        },
        "measurement_rows": rows,
    }
    result["result_body_sha256"] = hashlib.sha256(_canonical_bytes(result)).hexdigest()
    result_payload = (
        json.dumps(result, sort_keys=True, indent=2).encode("ascii") + b"\n"
    )
    result_receipt = _publish_immutable_bytes(
        result_payload,
        output=output_directory / "result.json",
        temporary_name=f".result.{os.getpid()}.tmp",
    )
    _verify_result_bindings_or_remove_v2(
        result_receipt,
        runner_source=runner_source,
        artifact_binding=artifact_binding,
    )
    result["result_path"] = result_receipt["path"]
    return result


def run_direct_sparse_sustained_benchmark(
    *,
    artifact_path: Path,
    output_directory: Path,
    batch_size: int,
    temporal_k: int,
    samples: int,
    warmups: int,
    seed: int,
    initial_capacity_rows: int | None,
) -> dict[str, Any]:
    resolved_artifact = artifact_path.resolve(strict=True)
    with (
        _BoundArtifactManifestV2(resolved_artifact) as artifact_binding,
        _BoundRunnerSourceV2(artifact_path=resolved_artifact) as runner_source,
    ):
        return _run_direct_sparse_sustained_benchmark_bound(
            artifact_path=resolved_artifact,
            output_directory=output_directory,
            batch_size=batch_size,
            temporal_k=temporal_k,
            samples=samples,
            warmups=warmups,
            seed=seed,
            initial_capacity_rows=initial_capacity_rows,
            runner_source=runner_source,
            artifact_binding=artifact_binding,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=896)
    parser.add_argument("--temporal-k", type=int, default=64)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--initial-capacity-rows", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_direct_sparse_sustained_benchmark(
        artifact_path=args.artifact,
        output_directory=args.output,
        batch_size=args.batch_size,
        temporal_k=args.temporal_k,
        samples=args.samples,
        warmups=args.warmups,
        seed=args.seed,
        initial_capacity_rows=args.initial_capacity_rows,
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["status"] == "FULL_TARGET_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
