from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from engine.perf.cuda_full_evolution_v2 import (
    direct_sparse_runtime as runtime_subject,
    direct_sparse_sustained_benchmark as subject,
)
from engine.perf.cuda_full_evolution_v2.direct_sparse_runtime import (
    CudaFullEvolutionDirectSparseDurableSinkReceiptV3,
    CudaFullEvolutionDirectSparseOwnerV3,
    PreparedDirectSparseDurableEnvelopeV3,
    VerifiedDirectSparseDurableEnvelopeV3,
    _PendingDirectSparseDispatchV3,
    _capture_direct_sparse_dispatch_authority_v3,
    _durable_sink_receipt_body_v3,
    encode_direct_sparse_durable_envelope_v3,
    prepare_direct_sparse_durable_envelope_v3,
    seal_direct_sparse_durable_sink_receipt_v3,
    validate_direct_sparse_durable_sink_receipt_v3,
)
from engine.perf.cuda_full_evolution_v2.runtime import (
    CudaFullEvolutionRuntimeError,
    CudaFullEvolutionSparseEventDispatchV3,
    _canonical_sha256,
    _sparse_event_content_sha256,
    seal_sparse_event_sink_receipt_v3,
)
from engine.perf.cuda_full_evolution_v2.native_loader import canonical_sha256


def _dispatch() -> CudaFullEvolutionSparseEventDispatchV3:
    values = {
        "packed_events": b"",
        "world_offsets": (0, 0),
        "event_counts": (0,),
        "final_next_agent_ids": (257,),
        "world_uids": (71_001,),
        "start_next_absolute_tick": 10,
        "end_next_absolute_tick": 11,
        "tick_count": 1,
        "artifact_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
    }
    content_sha256 = _sparse_event_content_sha256(**values)
    return CudaFullEvolutionSparseEventDispatchV3(
        **values,
        drain_elapsed_seconds=0.0,
        count_projection_elapsed_seconds=0.0,
        compaction_and_transfer_elapsed_seconds=0.0,
        content_hash_elapsed_seconds=0.0,
        content_sha256=content_sha256,
    )


def _publish(tmp_path: Path):
    dispatch = _dispatch()
    publication, handle = subject._publish_dispatch(
        dispatch, output_directory=tmp_path, sample_id="sample-000"
    )
    return dispatch, publication, handle


def _owner_prepared(
    dispatch: CudaFullEvolutionSparseEventDispatchV3,
    *,
    sample_id: str = "sample-000",
) -> tuple[
    CudaFullEvolutionDirectSparseOwnerV3,
    PreparedDirectSparseDurableEnvelopeV3,
]:
    owner = object.__new__(CudaFullEvolutionDirectSparseOwnerV3)
    owner._CudaFullEvolutionDirectSparseOwnerV3__closed = False
    owner._CudaFullEvolutionDirectSparseOwnerV3__poisoned_reason = None
    owner._CudaFullEvolutionDirectSparseOwnerV3__pending = (
        _PendingDirectSparseDispatchV3(
            start_next_absolute_tick=dispatch.start_next_absolute_tick,
            tick_count=dispatch.tick_count,
            dispatch=dispatch,
            dispatch_authority=_capture_direct_sparse_dispatch_authority_v3(dispatch),
        )
    )
    owner._CudaFullEvolutionDirectSparseOwnerV3__pending_prepared_envelope = None
    prepared = owner.prepare_sparse_event_durable_envelope_v3(
        dispatch, sample_id=sample_id
    )
    return owner, prepared


def _receipt(dispatch, publication, handle):
    return seal_direct_sparse_durable_sink_receipt_v3(
        dispatch,
        sink_id="pytest-exact-local-envelope",
        sink_receipt_id=f"sample-000:{publication['sha256']}",
        sample_id="sample-000",
        durable_envelope=handle,
    )


def _bound_artifact_manifest(tmp_path: Path):
    artifact = (tmp_path / "artifact.so").resolve()
    manifest = artifact.with_suffix(".so.manifest.json")
    artifact.write_bytes(b"EXACT-NATIVE-ARTIFACT")
    manifest_body = {
        "schema_version": "pytest-exact-manifest",
        "output": {"sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()},
    }
    manifest_payload = {
        **manifest_body,
        "manifest_sha256": canonical_sha256(manifest_body),
    }
    manifest.write_text(
        json.dumps(manifest_payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    artifact.chmod(0o444)
    manifest.chmod(0o444)
    return artifact, manifest, subject._BoundArtifactManifestV2(artifact)


def _reseal(
    receipt: CudaFullEvolutionDirectSparseDurableSinkReceiptV3,
    **changes: object,
) -> CudaFullEvolutionDirectSparseDurableSinkReceiptV3:
    provisional = replace(receipt, **changes, receipt_sha256="")
    return replace(
        provisional,
        receipt_sha256=_canonical_sha256(_durable_sink_receipt_body_v3(provisional)),
    )


def test_exact_durable_receipt_binds_open_inode_and_canonical_bytes(
    tmp_path: Path,
) -> None:
    dispatch, publication, handle = _publish(tmp_path)
    try:
        receipt = _receipt(dispatch, publication, handle)
        binding = validate_direct_sparse_durable_sink_receipt_v3(
            dispatch, receipt, durable_envelope=handle
        )
        assert binding["durable_path"] == publication["path"]
        assert binding["durable_inode"] == publication["inode"]
        assert binding["durable_device"] == publication["device"]
        assert binding["durable_content_sha256"] == publication["sha256"]
        assert binding["durable_mode"] == stat.S_IRUSR
        assert binding["durable_link_count"] == 1
    finally:
        handle.close()


def test_owner_issued_prepared_capability_is_exact_and_memoized(
    tmp_path: Path,
) -> None:
    dispatch = _dispatch()
    owner, prepared = _owner_prepared(dispatch)
    assert (
        owner.prepare_sparse_event_durable_envelope_v3(dispatch, sample_id="sample-000")
        is prepared
    )
    publication, handle = subject._publish_prepared_dispatch(
        prepared, output_directory=tmp_path, sample_id="sample-000"
    )
    try:
        assert handle.is_bound_to_prepared(prepared)
        assert publication["sha256"] == prepared.content_sha256
        assert publication["bytes"] == prepared.nbytes
    finally:
        handle.close()


def test_prepared_capability_constructor_rejects_forged_factory_token() -> None:
    dispatch = _dispatch()
    payload = encode_direct_sparse_durable_envelope_v3(dispatch, sample_id="sample-000")
    with pytest.raises(CudaFullEvolutionRuntimeError, match="owner-factory-owned"):
        PreparedDirectSparseDurableEnvelopeV3(
            dispatch=dispatch,
            sample_id="sample-000",
            payload=payload,
            content_sha256=hashlib.sha256(payload).hexdigest(),
            _factory_token=object(),
        )


def test_prepared_capability_rejects_equal_forged_dispatch_object() -> None:
    dispatch = _dispatch()
    _, prepared = _owner_prepared(dispatch)
    forged = replace(dispatch)
    with pytest.raises(CudaFullEvolutionRuntimeError, match="another dispatch object"):
        prepared.payload_for_publication(forged)


def test_verified_handle_rejects_equal_value_prepared_clone_for_owner_binding(
    tmp_path: Path,
) -> None:
    dispatch = _dispatch()
    _, owner_prepared = _owner_prepared(dispatch)
    clone = prepare_direct_sparse_durable_envelope_v3(
        dispatch, sample_id="sample-000"
    )
    assert clone is not owner_prepared
    publication, handle = subject._publish_prepared_dispatch(
        clone, output_directory=tmp_path, sample_id="sample-000"
    )
    try:
        assert publication["sha256"] == owner_prepared.content_sha256
        assert not handle.is_bound_to_prepared(owner_prepared)
    finally:
        handle.close()


def test_prepared_capability_rejects_constituent_replacement() -> None:
    dispatch = _dispatch()
    _, prepared = _owner_prepared(dispatch)
    object.__setattr__(dispatch, "world_offsets", tuple(list(dispatch.world_offsets)))
    with pytest.raises(
        CudaFullEvolutionRuntimeError, match="changed after preparation"
    ):
        prepared.payload_for_publication(dispatch)


def test_verified_handle_rejects_forged_prepared_byte_identity(
    tmp_path: Path,
) -> None:
    dispatch = _dispatch()
    _, prepared = _owner_prepared(dispatch)
    path = (tmp_path / "false-envelope.bin").resolve()
    false_payload = b"X" * prepared.nbytes
    path.write_bytes(false_payload)
    path.chmod(stat.S_IRUSR)
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    state = os.fstat(descriptor)
    try:
        with pytest.raises(
            CudaFullEvolutionRuntimeError,
            match="byte identity differs from preparation",
        ):
            VerifiedDirectSparseDurableEnvelopeV3(
                descriptor=descriptor,
                path=path,
                identity=(state.st_dev, state.st_ino),
                mode=stat.S_IRUSR,
                nbytes=state.st_size,
                content_sha256=hashlib.sha256(false_payload).hexdigest(),
                mtime_ns=state.st_mtime_ns,
                ctime_ns=state.st_ctime_ns,
                prepared=prepared,
                _factory_token=runtime_subject._VERIFIED_ENVELOPE_FACTORY_TOKEN_V3,
            )
    finally:
        os.close(descriptor)


def test_pending_authority_rejects_pre_preparation_constituent_replacement() -> None:
    dispatch = _dispatch()
    pending = _PendingDirectSparseDispatchV3(
        start_next_absolute_tick=dispatch.start_next_absolute_tick,
        tick_count=dispatch.tick_count,
        dispatch=dispatch,
        dispatch_authority=_capture_direct_sparse_dispatch_authority_v3(dispatch),
    )
    object.__setattr__(dispatch, "event_counts", tuple(list(dispatch.event_counts)))
    with pytest.raises(CudaFullEvolutionRuntimeError, match="changed after owner"):
        pending.verify_dispatch(dispatch)


@pytest.mark.parametrize("replace_inode", [False, True])
def test_prepared_publication_reopen_failure_removes_only_owned_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replace_inode: bool,
) -> None:
    dispatch = _dispatch()
    _, prepared = _owner_prepared(dispatch)
    real_open = subject.open_verified_prepared_direct_sparse_durable_envelope_v3
    replacement = b"F" * prepared.nbytes

    def corrupt_before_reopen(capability, *, durable_path):
        path = Path(durable_path)
        if replace_inode:
            path.unlink()
            path.write_bytes(replacement)
        else:
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
            path.write_bytes(replacement)
        path.chmod(stat.S_IRUSR)
        return real_open(capability, durable_path=durable_path)

    monkeypatch.setattr(
        subject,
        "open_verified_prepared_direct_sparse_durable_envelope_v3",
        corrupt_before_reopen,
    )
    with pytest.raises(CudaFullEvolutionRuntimeError, match="bytes or immutable"):
        subject._publish_prepared_dispatch(
            prepared, output_directory=tmp_path, sample_id="sample-000"
        )
    output = tmp_path / "sample-000.direct-sparse-events-v3.bin"
    if replace_inode:
        assert output.read_bytes() == replacement
    else:
        assert not output.exists()


def test_ack_rehash_rejects_content_even_if_inode_metadata_is_rebound(
    tmp_path: Path,
) -> None:
    dispatch, publication, handle = _publish(tmp_path)
    receipt = _receipt(dispatch, publication, handle)
    output = Path(publication["path"])
    output.chmod(stat.S_IRUSR | stat.S_IWUSR)
    output.write_bytes(b"X" * publication["bytes"])
    output.chmod(stat.S_IRUSR)
    rebound = os.lstat(output)
    handle._VerifiedDirectSparseDurableEnvelopeV3__mtime_ns = rebound.st_mtime_ns
    handle._VerifiedDirectSparseDurableEnvelopeV3__ctime_ns = rebound.st_ctime_ns
    try:
        with pytest.raises(
            CudaFullEvolutionRuntimeError, match="content digest changed"
        ):
            validate_direct_sparse_durable_sink_receipt_v3(
                dispatch, receipt, durable_envelope=handle
            )
    finally:
        handle.close()


def test_publication_collision_preserves_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "evidence.bin"
    output.write_bytes(b"existing")
    with pytest.raises(
        subject.DirectSparseSustainedBenchmarkError, match="cannot replace"
    ):
        subject._publish_immutable_bytes(
            b"new", output=output, temporary_name=".collision.tmp"
        )
    assert output.read_bytes() == b"existing"


def test_temporary_substitution_removes_only_observed_false_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "evidence.bin"
    temporary = tmp_path / ".race.tmp"
    real_link = os.link

    def substitute_then_link(source, destination, *, follow_symlinks=True):
        Path(source).unlink()
        Path(source).write_bytes(b"FOREIGN-RACE-PAYLOAD")
        Path(source).chmod(stat.S_IRUSR)
        return real_link(source, destination, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(subject.os, "link", substitute_then_link)
    with pytest.raises(
        subject.DirectSparseSustainedBenchmarkError,
        match="did not preserve the created inode",
    ):
        subject._publish_immutable_bytes(
            b"owned", output=output, temporary_name=temporary.name
        )
    assert not output.exists()
    assert temporary.read_bytes() == b"FOREIGN-RACE-PAYLOAD"


def test_final_substitution_during_link_validation_leaves_no_false_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "evidence.bin"
    real_lstat = os.lstat
    replaced = False

    def substitute_final(path, *args, **kwargs):
        nonlocal replaced
        if Path(path) == output and not replaced:
            replaced = True
            output.unlink()
            output.write_bytes(b"FOREIGN-FINAL")
            output.chmod(stat.S_IRUSR)
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(subject.os, "lstat", substitute_final)
    with pytest.raises(
        subject.DirectSparseSustainedBenchmarkError,
        match="did not preserve the created inode",
    ):
        subject._publish_immutable_bytes(
            b"owned", output=output, temporary_name=".final-race.tmp"
        )
    assert replaced
    assert not output.exists()


def test_path_replacement_after_receipt_is_rejected_before_commit(
    tmp_path: Path,
) -> None:
    dispatch, publication, handle = _publish(tmp_path)
    receipt = _receipt(dispatch, publication, handle)
    output = Path(publication["path"])
    output.unlink()
    output.write_bytes(b"FOREIGN-AFTER-RECEIPT")
    output.chmod(stat.S_IRUSR)
    try:
        with pytest.raises(CudaFullEvolutionRuntimeError, match="no longer names"):
            validate_direct_sparse_durable_sink_receipt_v3(
                dispatch, receipt, durable_envelope=handle
            )
    finally:
        handle.close()


@pytest.mark.parametrize("mutation", ["writable", "hardlink", "content", "symlink"])
def test_inode_mutation_is_rejected(tmp_path: Path, mutation: str) -> None:
    dispatch, publication, handle = _publish(tmp_path)
    receipt = _receipt(dispatch, publication, handle)
    output = Path(publication["path"])
    if mutation == "writable":
        output.chmod(stat.S_IRUSR | stat.S_IWUSR)
    elif mutation == "hardlink":
        os.link(output, tmp_path / "extra-link")
    elif mutation == "content":
        output.chmod(stat.S_IRUSR | stat.S_IWUSR)
        output.write_bytes(b"X" * publication["bytes"])
        output.chmod(stat.S_IRUSR)
    else:
        target = tmp_path / "foreign-target"
        target.write_bytes(b"FOREIGN")
        output.unlink()
        output.symlink_to(target)
    try:
        with pytest.raises(CudaFullEvolutionRuntimeError, match="no longer names"):
            validate_direct_sparse_durable_sink_receipt_v3(
                dispatch, receipt, durable_envelope=handle
            )
    finally:
        handle.close()


def test_forged_structured_binding_is_rejected_even_when_resealed(
    tmp_path: Path,
) -> None:
    dispatch, publication, handle = _publish(tmp_path)
    try:
        forged = _reseal(
            _receipt(dispatch, publication, handle),
            durable_inode=int(publication["inode"]) + 1,
        )
        with pytest.raises(CudaFullEvolutionRuntimeError, match="open envelope handle"):
            validate_direct_sparse_durable_sink_receipt_v3(
                dispatch, forged, durable_envelope=handle
            )
    finally:
        handle.close()


def test_generic_external_sink_receipt_cannot_commit_direct_owner(
    tmp_path: Path,
) -> None:
    dispatch, _, handle = _publish(tmp_path)
    try:
        generic = seal_sparse_event_sink_receipt_v3(
            dispatch,
            sink_id="generic-external-sink",
            sink_receipt_id="not-a-local-durable-binding",
        )
        with pytest.raises(CudaFullEvolutionRuntimeError, match="exact durable"):
            validate_direct_sparse_durable_sink_receipt_v3(
                dispatch,
                generic,
                durable_envelope=handle,  # type: ignore[arg-type]
            )
    finally:
        handle.close()


def test_forged_dispatch_bytes_with_stale_content_seal_are_rejected() -> None:
    dispatch = _dispatch()
    forged = replace(dispatch, packed_events=b"X" * 32)
    with pytest.raises(CudaFullEvolutionRuntimeError, match="content seal mismatch"):
        encode_direct_sparse_durable_envelope_v3(forged, sample_id="forged")


def test_ancestor_rename_and_symlink_alias_is_not_a_canonical_durable_path(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original"
    moved = tmp_path / "moved"
    original.mkdir()
    dispatch, publication, handle = _publish(original)
    receipt = _receipt(dispatch, publication, handle)
    original.rename(moved)
    original.symlink_to(moved, target_is_directory=True)
    try:
        with pytest.raises(CudaFullEvolutionRuntimeError, match="no longer names"):
            validate_direct_sparse_durable_sink_receipt_v3(
                dispatch, receipt, durable_envelope=handle
            )
    finally:
        handle.close()


def test_bound_artifact_manifest_projects_exact_checkpoint_hashes(
    tmp_path: Path,
) -> None:
    artifact, manifest, binding = _bound_artifact_manifest(tmp_path)
    try:
        artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
        manifest_file_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
        manifest_sha256 = json.loads(manifest.read_text(encoding="utf-8"))[
            "manifest_sha256"
        ]
        binding.verify(
            artifact_sha256=artifact_sha256,
            manifest_sha256=manifest_sha256,
        )
        projection = binding.projection()
        assert projection["path"] == str(artifact)
        assert projection["sha256"] == artifact_sha256
        assert projection["manifest_path"] == str(manifest)
        assert projection["manifest_sha256"] == manifest_sha256
        assert projection["manifest_file_sha256"] == manifest_file_sha256
        assert manifest_sha256 != manifest_file_sha256
        assert projection["link_count"] == 1
        assert projection["manifest_binding"]["link_count"] == 1
        with pytest.raises(
            subject.DirectSparseSustainedBenchmarkError,
            match="differs from the held input binding",
        ):
            binding.verify(artifact_sha256="0" * 64)
    finally:
        binding.close()


@pytest.mark.parametrize("target", ["artifact", "manifest"])
def test_bound_artifact_manifest_rejects_path_replacement(
    tmp_path: Path, target: str
) -> None:
    artifact, manifest, binding = _bound_artifact_manifest(tmp_path)
    replaced = artifact if target == "artifact" else manifest
    replaced.unlink()
    replaced.write_bytes(b"FOREIGN-REPLACEMENT")
    replaced.chmod(0o444)
    try:
        with pytest.raises(
            subject.DirectSparseSustainedBenchmarkError,
            match="changed during the benchmark transaction",
        ):
            binding.verify()
    finally:
        binding.close()


@pytest.mark.parametrize("replace_result", [False, True])
def test_post_publication_binding_failure_removes_only_exact_result(
    tmp_path: Path, replace_result: bool
) -> None:
    artifact, _, binding = _bound_artifact_manifest(tmp_path)
    result_path = tmp_path / "result.json"
    result_receipt = subject._publish_immutable_bytes(
        b'{"status":"FULL_TARGET_PASS"}\n',
        output=result_path,
        temporary_name=".result.tmp",
    )
    artifact.unlink()
    artifact.write_bytes(b"FOREIGN-ARTIFACT")
    artifact.chmod(0o444)
    if replace_result:
        result_path.unlink()
        result_path.write_bytes(b"FOREIGN-RESULT")
        result_path.chmod(stat.S_IRUSR)

    class _UnchangedRunner:
        @staticmethod
        def verify() -> None:
            return None

    try:
        with pytest.raises(
            subject.DirectSparseSustainedBenchmarkError,
            match="changed during the benchmark transaction",
        ):
            subject._verify_result_bindings_or_remove_v2(
                result_receipt,
                runner_source=_UnchangedRunner(),  # type: ignore[arg-type]
                artifact_binding=binding,
            )
        if replace_result:
            assert result_path.read_bytes() == b"FOREIGN-RESULT"
        else:
            assert not result_path.exists()
    finally:
        binding.close()
