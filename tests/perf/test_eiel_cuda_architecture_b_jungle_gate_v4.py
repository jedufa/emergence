from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.scripts import eiel_cuda_architecture_b_jungle_gate_v4 as gate


def test_generation_and_data_path_contracts() -> None:
    assert gate._require_generation_count(32) == 32
    assert gate._require_data_path(Path("/data/eiel"), label="output") == Path(
        "/data/eiel"
    )
    with pytest.raises(ValueError, match="generations must be"):
        gate._require_generation_count(33)
    with pytest.raises(ValueError, match="must be under /data"):
        gate._require_data_path(Path("/tmp/eiel"), label="output")


def test_repository_identity_binds_clean_status(monkeypatch: pytest.MonkeyPatch) -> None:
    head = "a" * 40

    def run(command, **_kwargs):
        stdout = head + "\n" if command[1:3] == ["rev-parse", "HEAD"] else ""
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(gate.subprocess, "run", run)

    assert gate._repository_source_identity() == {
        "schema_version": "eiel_repository_source_identity_v1",
        "git_head": head,
        "worktree_clean": True,
        "status_sha256": hashlib.sha256(b"").hexdigest(),
    }


@pytest.mark.parametrize(
    ("requested", "allocation_extent"), ((1, 3), (2, 3), (3, 3), (32, 32))
)
def test_jungle_runner_uses_the_admitted_allocation_generation_extent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested: int,
    allocation_extent: int,
) -> None:
    captured: dict[str, object] = {}

    def build_runtime_runner(*_args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(runner="runner")

    monkeypatch.setattr(
        gate, "build_patch_jungle_fixed_mlp_runner_v1", build_runtime_runner
    )

    actual = gate._new_jungle_runner(
        tmp_path,
        seed=101,
        generations=requested,
        cuda_artifact=tmp_path / "owner.so",
    )

    assert actual == "runner"
    assert captured["generations"] == allocation_extent


@pytest.mark.parametrize("generations", (1, 2, 3, 7, 14, 32))
def test_dispatch_contract_accepts_one_exact_native_dispatch(
    generations: int,
) -> None:
    result = {
        "jungle_native_owner": True,
        "jungle_python_tick_or_generation_ownership": False,
        "hidden_fallback": False,
        "runtime_jit": False,
    }
    if generations <= 2:
        result["jungle_physical_gate_entrypoint"] = True
    else:
        result.update(
            {
                "completed_generations": generations,
                "logical_generation_count": generations,
                "external_python_window_boundaries": 0,
                "direct_cpp_window_core_dispatch": True,
                "production_native_chunk_loop": True,
            }
        )
    receipt = gate._dispatch_contract(result, generations=generations)
    assert receipt["ok"] is True
    assert receipt["python_native_dispatches"] == 1


def test_cuda_manifest_receipt_rejects_changed_source(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "owner.so"
    artifact.write_bytes(b"artifact")
    source = tmp_path / "source.cu"
    source.write_text("before\n", encoding="utf-8")
    manifest = {
        "schema_version": (
            "eiel_cuda_architecture_b_flattened_jungle_controller_extension_manifest_v4"
        ),
        "module_name": gate.CUDA_MODULE_NAME,
        "source_snapshot_stable": True,
        "runtime_jit_compilation": False,
        "runtime_jit_required_by_validation_route": False,
        "sass_only": True,
        "artifact_sha256": hashlib.sha256(b"artifact").hexdigest(),
        "source_hashes": {
            str(source): hashlib.sha256(b"before\n").hexdigest(),
        },
    }
    manifest_path = Path(f"{artifact}.manifest.json")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt = gate._cuda_artifact_receipt(artifact)
    assert receipt["source_file_count"] == 1

    source.write_text("after\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no longer matches current source"):
        gate._cuda_artifact_receipt(artifact)
