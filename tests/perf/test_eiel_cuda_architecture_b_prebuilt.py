from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from perf import cuda_architecture_b_generation as generation
from perf.cuda_architecture_b_prebuilt import (
    EXTENSION_ABI_VERSION,
    EXTENSION_VERSION,
    MANIFEST_SCHEMA_VERSION,
    validate_cuda_prebuilt_manifest,
)
from tools.scripts.build_eiel_cuda_architecture_b_extension import (
    _canonical_json_sha256,
)


def _artifact(tmp_path: Path, **overrides: object) -> Path:
    artifact = tmp_path / "eiel_cuda_architecture_b_ext.so"
    artifact.write_bytes(b"bounded immutable CUDA artifact fixture")
    capability_contract = {
        "schema_version": "test",
        "promotion_qualified": False,
        "throughput_qualified": False,
    }
    manifest: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "extension_abi_version": EXTENSION_ABI_VERSION,
        "extension_version": EXTENSION_VERSION,
        "immutable_output": True,
        "prebuilt_outside_measured_execution": True,
        "runtime_jit_in_measured_execution": False,
        "output_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "output": str(artifact.resolve()),
        "stable_module_name": "eiel_cuda_architecture_b_ext",
        "exported_functions": ["generation", "capabilities"],
        "capability_contract": capability_contract,
        "capability_contract_sha256": _canonical_json_sha256(capability_contract),
    }
    manifest.update(overrides)
    artifact.with_suffix(artifact.suffix + ".manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return artifact


def test_prebuilt_manifest_accepts_exact_immutable_identity(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)

    identity = validate_cuda_prebuilt_manifest(
        artifact,
        required_exports=("generation", "capabilities"),
    )

    assert identity.artifact == artifact.resolve()
    assert identity.artifact_sha256 == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert identity.exported_functions == ("generation", "capabilities")


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"runtime_jit_in_measured_execution": True}, "runtime_jit=false"),
        ({"immutable_output": False}, "not declared immutable"),
        ({"extension_abi_version": "wrong"}, "extension contract mismatch"),
        ({"output_sha256": "0" * 64}, "artifact hash mismatch"),
        ({"output": "/tmp/wrong.so"}, "output path mismatch"),
        ({"stable_module_name": "invalid-module"}, "module name is invalid"),
        ({"capability_contract_sha256": "0" * 64}, "capability contract hash mismatch"),
        (
            {
                "capability_contract": {
                    "promotion_qualified": True,
                    "throughput_qualified": False,
                }
            },
            "capability contract hash mismatch",
        ),
    ],
)
def test_prebuilt_manifest_fails_closed_on_identity_contradictions(
    tmp_path: Path,
    overrides: dict[str, object],
    match: str,
) -> None:
    artifact = _artifact(tmp_path, **overrides)
    with pytest.raises(RuntimeError, match=match):
        validate_cuda_prebuilt_manifest(artifact)


def test_prebuilt_manifest_rejects_missing_required_export(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    with pytest.raises(RuntimeError, match="missing required exports"):
        validate_cuda_prebuilt_manifest(artifact, required_exports=("generation", "checkpoint"))


def test_generation_prebuilt_request_fails_closed_without_jit_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "missing.so"
    monkeypatch.setenv("EIEL_CUDA_ARCHITECTURE_B_PREBUILT", str(artifact))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    jit_called = False

    def _unexpected_jit(**_: object) -> object:
        nonlocal jit_called
        jit_called = True
        raise AssertionError("JIT fallback must not run")

    monkeypatch.setattr(generation, "load", _unexpected_jit)

    with pytest.raises(RuntimeError, match="artifact is missing"):
        generation.load_cuda_generation_extension()
    assert jit_called is False
