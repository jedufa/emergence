from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from engine.campaigns import acs_native_streamed_checkpoint_v1 as streamed
from engine.campaigns.providers import (
    ACS_NATIVE_RUNTIME_ABI_V1,
    ACS_NATIVE_RUNTIME_EXTENSION_ARTIFACT_ID_V1,
    ACS_NATIVE_RUNTIME_ENVIRONMENT_VARIABLE_V1,
    ACS_NATIVE_RUNTIME_EXTENSION_KIND_V1,
    ACS_NATIVE_RUNTIME_MANIFEST_ARTIFACT_ID_V1,
    ACS_NATIVE_RUNTIME_MANIFEST_KIND_V1,
    ACS_NATIVE_RUNTIME_MODULE_NAME_V1,
    ACS_NATIVE_RUNTIME_VERSION_V1,
    PRODUCTION_PROFILE_ID,
    ProductionEngineLifecycleV1,
    ResidentBatchRequestV1,
    _acs_native_runtime_artifact_binding_v1,
    _acs_native_runtime_route_component_v1,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan(tmp_path: Path) -> dict[str, Any]:
    extension = tmp_path / "eiel_native_runtime_ext.so"
    extension.write_bytes(b"native ACS runtime test extension")
    manifest = extension.with_suffix(extension.suffix + ".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "eiel_native_runtime_build_manifest_v1",
                "stable_module_name": ACS_NATIVE_RUNTIME_MODULE_NAME_V1,
                "extension_abi_version": ACS_NATIVE_RUNTIME_ABI_V1,
                "extension_version": ACS_NATIVE_RUNTIME_VERSION_V1,
                "output_sha256": _sha256(extension),
                "runtime_jit_in_measured_execution": False,
            }
        ),
        encoding="utf-8",
    )
    return {
        "execution": {"native_streamed_checkpoint": True},
        "artifacts": {
            ACS_NATIVE_RUNTIME_EXTENSION_ARTIFACT_ID_V1: {
                "kind": ACS_NATIVE_RUNTIME_EXTENSION_KIND_V1,
                "abi": ACS_NATIVE_RUNTIME_ABI_V1,
                "path": str(extension),
                "sha256": _sha256(extension),
            },
            ACS_NATIVE_RUNTIME_MANIFEST_ARTIFACT_ID_V1: {
                "kind": ACS_NATIVE_RUNTIME_MANIFEST_KIND_V1,
                "abi": ACS_NATIVE_RUNTIME_ABI_V1,
                "path": str(manifest),
                "sha256": _sha256(manifest),
            },
        },
    }


def _request() -> ResidentBatchRequestV1:
    return ResidentBatchRequestV1(
        profile_id=PRODUCTION_PROFILE_ID,
        batch_id="binding-test",
        cell_ids=("cell-0",),
        generations=1,
        ticks_per_generation=1,
        checkpoint_milestones=(),
        parameter_arrays=(),
        seed_arrays=(),
        execution_route_sha256=None,
        native_streamed_checkpoint=True,
        total_ticks=1,
        start_next_tick=0,
        checkpoint_ticks=(1,),
    )


def test_native_runtime_binding_validates_manifest_and_route_identity(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)

    binding = _acs_native_runtime_artifact_binding_v1(plan, required=True)
    assert binding is not None
    assert binding["extension_sha256"] == plan["artifacts"][
        ACS_NATIVE_RUNTIME_EXTENSION_ARTIFACT_ID_V1
    ]["sha256"]
    component = _acs_native_runtime_route_component_v1(plan, required=True)
    assert component is not None
    assert component["component_id"] == "acs_native_runtime_binding"
    assert component["manifest_sha256"] == binding["manifest_sha256"]


def test_native_runtime_binding_fails_closed_when_manifest_digest_is_stale(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    plan["artifacts"][ACS_NATIVE_RUNTIME_MANIFEST_ARTIFACT_ID_V1][
        "sha256"
    ] = "0" * 64

    with pytest.raises(RuntimeError, match="mapping is stale"):
        _acs_native_runtime_artifact_binding_v1(plan, required=True)


def test_native_runtime_binding_is_required_for_production_native_checkpoint(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    plan["artifacts"].pop(ACS_NATIVE_RUNTIME_EXTENSION_ARTIFACT_ID_V1)
    plan["artifacts"].pop(ACS_NATIVE_RUNTIME_MANIFEST_ARTIFACT_ID_V1)

    with pytest.raises(RuntimeError, match="binding is missing"):
        _acs_native_runtime_artifact_binding_v1(plan, required=True)


def test_provider_binds_plan_artifact_only_for_execution_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(tmp_path)
    provider = ProductionEngineLifecycleV1()
    observed: dict[str, str | None] = {}

    def fake_execute(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        observed["path"] = os.environ.get(
            ACS_NATIVE_RUNTIME_ENVIRONMENT_VARIABLE_V1
        )
        return {"status": "ok"}

    monkeypatch.delenv(ACS_NATIVE_RUNTIME_ENVIRONMENT_VARIABLE_V1, raising=False)
    monkeypatch.setattr(provider, "_execute_segments", fake_execute)
    result = provider._execute_segments_with_acs_cleanup(plan, _request(), None)

    assert result == {"status": "ok"}
    assert observed["path"] == plan["artifacts"][
        ACS_NATIVE_RUNTIME_EXTENSION_ARTIFACT_ID_V1
    ]["path"]
    assert os.environ.get(ACS_NATIVE_RUNTIME_ENVIRONMENT_VARIABLE_V1) is None


def test_native_loader_retries_after_an_unbound_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import engine.native_runtime.bindings as bindings

    fake_module = SimpleNamespace(
        ACSNativeStreamedCheckpointSlabPool=object,
        acs_native_streamed_checkpoint_copy_tensor_chunks=lambda *args: 0,
    )
    monkeypatch.setattr(bindings, "load_prebuilt_extension", lambda: fake_module)
    monkeypatch.setattr(streamed, "_NATIVE_SLAB_POOL_LOOKUP_DONE", True)
    monkeypatch.setattr(streamed, "_NATIVE_SLAB_POOL_LOOKUP_PATH", "")
    monkeypatch.setattr(streamed, "_NATIVE_SLAB_POOL_TYPE", None)
    monkeypatch.setattr(streamed, "_NATIVE_CHUNK_LOOKUP_DONE", True)
    monkeypatch.setattr(streamed, "_NATIVE_CHUNK_LOOKUP_PATH", "")
    monkeypatch.setattr(streamed, "_NATIVE_CHUNK_COPY", None)
    monkeypatch.delenv(ACS_NATIVE_RUNTIME_ENVIRONMENT_VARIABLE_V1, raising=False)

    assert streamed._native_slab_pool_type() is None
    monkeypatch.setenv(ACS_NATIVE_RUNTIME_ENVIRONMENT_VARIABLE_V1, "/bound.so")
    assert streamed._native_slab_pool_type() is object
    assert callable(streamed._native_chunk_copy())
