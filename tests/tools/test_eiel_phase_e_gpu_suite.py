from __future__ import annotations

import json
import stat
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.scripts import eiel_phase_e_gpu_suite as suite


NODES = tuple(
    f"tests/perf/test_gpu_suite_contract.py::test_gpu_contract_{index}"
    for index in range(10)
)


def _sha256(path: Path) -> str:
    return suite._sha256_file(path)


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _fake_validate_prebuilt(
    artifact: Path, *, required_exports: tuple[str, ...]
) -> SimpleNamespace:
    manifest_path = artifact.with_suffix(artifact.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("output") != str(artifact):
        raise RuntimeError("output mismatch")
    if manifest.get("output_sha256") != _sha256(artifact):
        raise RuntimeError("hash mismatch")
    if manifest.get("immutable_output") is not True:
        raise RuntimeError("not immutable")
    if manifest.get("exported_functions") != list(required_exports):
        raise RuntimeError("exports mismatch")
    return SimpleNamespace(
        manifest=manifest,
        module_name=str(manifest["stable_module_name"]),
    )


@pytest.fixture
def contract() -> SimpleNamespace:
    source_a = Path(suite.__file__).resolve()
    source_b = Path(__file__).resolve()
    return SimpleNamespace(
        capability_contract={"phase_e": True, "promotion_qualified": False},
        generation_hash_sources=(source_a, source_b),
        modular_tick_hash_sources=(source_b, source_a),
        native_source_files=(),
        test_source_files=(),
        required_nodes=NODES,
        required_exports=("generation_phase_e",),
        validate_prebuilt=_fake_validate_prebuilt,
    )


def _artifact(tmp_path: Path, contract: SimpleNamespace) -> Path:
    artifact = _write(tmp_path / "artifact" / "generation.so", b"generation")
    rows = [
        {"path": str(path), "sha256": _sha256(path)}
        for path in contract.generation_hash_sources
    ]
    manifest = {
        "output": str(artifact),
        "output_sha256": _sha256(artifact),
        "immutable_output": True,
        "stable_module_name": "phase_e_test_generation",
        "exported_functions": list(contract.required_exports),
        "capability_contract": dict(contract.capability_contract),
        "sources": rows,
    }
    artifact.with_suffix(".so.manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return artifact


def _source_identity() -> dict[str, object]:
    return {
        "repository_root": str(suite.ROOT),
        "branch": "integration/evolutionary-capacity-phase-e",
        "commit": "a" * 40,
        "test_commit": "a" * 40,
        "relevant_paths": [
            "tools/scripts/eiel_phase_e_gpu_suite.py",
            "tests/tools/test_eiel_phase_e_gpu_suite.py",
        ],
        "relevant_source_clean": True,
    }


def _cuda_environment() -> dict[str, object]:
    return {
        "python_version": "3.12.0",
        "numpy_version": "2.test",
        "torch_version": "2.test",
        "torch_cuda_version": "12.test",
        "cuda_available": True,
        "visible_device_count": 1,
        "current_device_index": 0,
        "device_name": "mock GPU",
        "compute_capability": [7, 5],
        "device_total_memory_bytes": 6_000_000_000,
    }


def _write_junit(path: Path, nodes: tuple[str, ...], *, outcome: str | None = None) -> None:
    root = ET.Element("testsuites")
    testsuite = ET.SubElement(root, "testsuite", tests=str(len(nodes)))
    for node in nodes:
        file_part, name = node.split("::", 1)
        classname = file_part.removesuffix(".py").replace("/", ".")
        case = ET.SubElement(testsuite, "testcase", classname=classname, name=name)
        if outcome is not None:
            ET.SubElement(case, outcome)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _successful_subprocess(
    calls: list[dict[str, object]],
    *,
    nodes: tuple[str, ...] = NODES,
    modular_count: int = 1,
    outcome: str | None = None,
):
    def run(command, **kwargs):
        calls.append({"command": list(command), **kwargs})
        junit_arg = next(value for value in command if value.startswith("--junitxml="))
        basetemp_arg = next(value for value in command if value.startswith("--basetemp="))
        junit = Path(junit_arg.split("=", 1)[1])
        basetemp = Path(basetemp_arg.split("=", 1)[1])
        _write_junit(junit, nodes, outcome=outcome)
        for index in range(modular_count):
            _write(
                basetemp
                / "phase_e_12_arm_native_tick0"
                / f"eiel_cuda_architecture_b_modular_tick_mock{index}.so",
                f"modular-{index}".encode(),
            )
        kwargs["stdout"].write(b"10 passed\n")
        kwargs["stdout"].flush()
        return subprocess.CompletedProcess(command, 0)

    return run


@pytest.fixture
def controlled(
    monkeypatch: pytest.MonkeyPatch, contract: SimpleNamespace
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(suite, "_load_contract", lambda: contract)
    monkeypatch.setattr(suite, "_source_identity", lambda unused: _source_identity())
    monkeypatch.setattr(suite, "_query_cuda_environment", _cuda_environment)
    monkeypatch.setattr(
        suite.subprocess, "run", _successful_subprocess(calls)
    )
    return calls


def test_runner_owns_exact_command_environment_and_manifest(
    tmp_path: Path,
    contract: SimpleNamespace,
    controlled: list[dict[str, object]],
) -> None:
    artifact = _artifact(tmp_path, contract)
    run_dir = tmp_path / "run"

    output = suite.run_suite(
        run_directory=run_dir,
        generation_artifact=artifact,
        cuda_visible_devices="0",
        python_executable=sys.executable,
    )

    assert output == run_dir / suite.MANIFEST_FILENAME
    assert len(controlled) == 1
    call = controlled[0]
    command = call["command"]
    assert command == suite._pytest_command(
        python_executable=Path(sys.executable),
        junit=run_dir / suite.JUNIT_FILENAME,
        basetemp=run_dir / suite.BASETEMP_DIRNAME,
        nodes=NODES,
    )
    environment = call["env"]
    assert environment[suite.PREBUILT_ENV] == str(artifact)
    assert environment["MAX_JOBS"] == "1"
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["PYTEST_ADDOPTS"] == ""
    assert environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["PYTHONHASHSEED"] == "0"
    assert call["cwd"] == suite.ROOT
    assert call["stderr"] is subprocess.STDOUT
    manifest = suite.validate_execution_manifest(
        output, required_nodes=NODES, contract=contract
    )
    assert manifest["source"]["commit"] == "a" * 40
    assert manifest["source"]["test_commit"] == "a" * 40
    assert manifest["pytest"]["exit_code"] == 0
    assert manifest["cuda_environment"]["device_name"] == "mock GPU"
    assert manifest["generation_artifact"]["sha256"] == _sha256(artifact)
    assert len(manifest["modular_tick_artifact"]["source_rows"]) == 2
    assert stat.S_IMODE(output.stat().st_mode) == 0o444


def test_runner_accepts_bounded_build_concurrency(
    tmp_path: Path,
    contract: SimpleNamespace,
    controlled: list[dict[str, object]],
) -> None:
    artifact = _artifact(tmp_path, contract)
    run_dir = tmp_path / "run-custom-jobs"

    output = suite.run_suite(
        run_directory=run_dir,
        generation_artifact=artifact,
        cuda_visible_devices="0",
        python_executable=sys.executable,
        max_jobs=4,
    )

    assert controlled[0]["env"]["MAX_JOBS"] == "4"
    manifest = suite.validate_execution_manifest(
        output, required_nodes=NODES, contract=contract
    )
    assert manifest["runner_environment"]["max_jobs"] == "4"


@pytest.mark.parametrize("max_jobs", (True, 0, -1, 257, "4"))
def test_runner_rejects_invalid_build_concurrency(
    tmp_path: Path,
    contract: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    max_jobs: object,
) -> None:
    artifact = _artifact(tmp_path, contract)
    monkeypatch.setattr(suite, "_load_contract", lambda: contract)

    with pytest.raises(suite.GpuSuiteError, match="max_jobs"):
        suite.run_suite(
            run_directory=tmp_path / f"invalid-jobs-{max_jobs}",
            generation_artifact=artifact,
            cuda_visible_devices="0",
            max_jobs=max_jobs,  # type: ignore[arg-type]
        )


def test_nonzero_pytest_never_writes_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract: SimpleNamespace,
) -> None:
    artifact = _artifact(tmp_path, contract)
    monkeypatch.setattr(suite, "_load_contract", lambda: contract)
    monkeypatch.setattr(suite, "_source_identity", lambda unused: _source_identity())

    def fail(command, **kwargs):
        kwargs["stdout"].write(b"failed\n")
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(suite.subprocess, "run", fail)
    run_dir = tmp_path / "failed"

    with pytest.raises(suite.GpuSuiteError, match="exit code 1"):
        suite.run_suite(
            run_directory=run_dir,
            generation_artifact=artifact,
            cuda_visible_devices="0",
        )

    assert not (run_dir / suite.MANIFEST_FILENAME).exists()


@pytest.mark.parametrize("outcome", ("skipped", "failure", "error"))
def test_nonpassing_junit_never_writes_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract: SimpleNamespace,
    outcome: str,
) -> None:
    calls: list[dict[str, object]] = []
    artifact = _artifact(tmp_path, contract)
    monkeypatch.setattr(suite, "_load_contract", lambda: contract)
    monkeypatch.setattr(suite, "_source_identity", lambda unused: _source_identity())
    monkeypatch.setattr(
        suite.subprocess,
        "run",
        _successful_subprocess(calls, outcome=outcome),
    )

    with pytest.raises(suite.GpuSuiteError, match="non-passing"):
        suite.run_suite(
            run_directory=tmp_path / f"run-{outcome}",
            generation_artifact=artifact,
            cuda_visible_devices="0",
        )

    assert not (tmp_path / f"run-{outcome}" / suite.MANIFEST_FILENAME).exists()


@pytest.mark.parametrize("count", (0, 2))
def test_missing_or_multiple_modular_artifacts_never_write_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract: SimpleNamespace,
    count: int,
) -> None:
    calls: list[dict[str, object]] = []
    artifact = _artifact(tmp_path, contract)
    monkeypatch.setattr(suite, "_load_contract", lambda: contract)
    monkeypatch.setattr(suite, "_source_identity", lambda unused: _source_identity())
    monkeypatch.setattr(
        suite.subprocess,
        "run",
        _successful_subprocess(calls, modular_count=count),
    )
    run_dir = tmp_path / f"run-{count}"

    with pytest.raises(suite.GpuSuiteError, match="missing|exactly one"):
        suite.run_suite(
            run_directory=run_dir,
            generation_artifact=artifact,
            cuda_visible_devices="0",
        )

    assert not (run_dir / suite.MANIFEST_FILENAME).exists()


def test_wrong_generation_manifest_is_rejected_before_pytest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract: SimpleNamespace,
) -> None:
    artifact = _artifact(tmp_path, contract)
    sidecar = artifact.with_suffix(".so.manifest.json")
    manifest = json.loads(sidecar.read_text())
    manifest["output_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(manifest))
    monkeypatch.setattr(suite, "_load_contract", lambda: contract)
    monkeypatch.setattr(suite, "_source_identity", lambda unused: _source_identity())
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("pytest must not start")

    monkeypatch.setattr(suite.subprocess, "run", forbidden)

    with pytest.raises(suite.GpuSuiteError, match="immutable prebuilt"):
        suite.run_suite(
            run_directory=tmp_path / "wrong-manifest",
            generation_artifact=artifact,
            cuda_visible_devices="0",
        )

    assert called is False


def test_dirty_relevant_source_is_rejected_before_pytest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract: SimpleNamespace,
) -> None:
    artifact = _artifact(tmp_path, contract)
    monkeypatch.setattr(suite, "_load_contract", lambda: contract)
    monkeypatch.setattr(suite, "_source_identity", lambda unused: _source_identity())

    def dirty(unused):
        raise suite.GpuSuiteError("production/test source must be committed and clean")

    monkeypatch.setattr(suite, "_source_identity", dirty)

    with pytest.raises(suite.GpuSuiteError, match="committed and clean"):
        suite.run_suite(
            run_directory=tmp_path / "dirty",
            generation_artifact=artifact,
            cuda_visible_devices="0",
        )


def test_symlink_and_existing_output_paths_are_rejected(
    tmp_path: Path,
    contract: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact(tmp_path, contract)
    monkeypatch.setattr(suite, "_load_contract", lambda: contract)
    monkeypatch.setattr(suite, "_source_identity", lambda unused: _source_identity())
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(suite.GpuSuiteError, match="must be new"):
        suite.run_suite(
            run_directory=existing,
            generation_artifact=artifact,
            cuda_visible_devices="0",
        )

    linked_artifact = tmp_path / "linked.so"
    linked_artifact.symlink_to(artifact)
    with pytest.raises(suite.GpuSuiteError, match="symlink"):
        suite.run_suite(
            run_directory=tmp_path / "linked-artifact-run",
            generation_artifact=linked_artifact,
            cuda_visible_devices="0",
        )

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(suite.GpuSuiteError, match="symlink"):
        suite.run_suite(
            run_directory=linked_parent / "run",
            generation_artifact=artifact,
            cuda_visible_devices="0",
        )


@pytest.mark.parametrize(
    "tamper",
    ("self_digest", "command", "log", "artifact", "modular_source"),
)
def test_manifest_tamper_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract: SimpleNamespace,
    controlled: list[dict[str, object]],
    tamper: str,
) -> None:
    artifact = _artifact(tmp_path, contract)
    output = suite.run_suite(
        run_directory=tmp_path / f"tamper-{tamper}",
        generation_artifact=artifact,
        cuda_visible_devices="0",
    )
    output.chmod(0o600)
    manifest = json.loads(output.read_text())
    if tamper == "self_digest":
        manifest["manifest_sha256"] = "0" * 64
    elif tamper == "command":
        manifest["command"].append("-k=other")
        manifest["manifest_sha256"] = suite._canonical_digest(manifest)
    elif tamper == "log":
        Path(manifest["pytest"]["log_path"]).write_text("tampered")
    elif tamper == "artifact":
        Path(manifest["generation_artifact"]["path"]).write_bytes(b"tampered")
    else:
        manifest["modular_tick_artifact"]["source_rows"][0]["sha256"] = "0" * 64
        manifest["manifest_sha256"] = suite._canonical_digest(manifest)
    output.write_text(json.dumps(manifest))
    output.chmod(0o444)

    with pytest.raises(suite.GpuSuiteError):
        suite.validate_execution_manifest(
            output, required_nodes=NODES, contract=contract
        )


def test_help_and_import_do_not_load_torch_or_cuda_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def forbidden():
        nonlocal called
        called = True
        raise AssertionError("contract must not load for help")

    monkeypatch.setattr(suite, "_load_contract", forbidden)
    with pytest.raises(SystemExit) as exc_info:
        suite.main(["--help"])
    assert exc_info.value.code == 0
    assert called is False


def test_cuda_environment_query_occurs_only_after_successful_pytest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract: SimpleNamespace,
) -> None:
    artifact = _artifact(tmp_path, contract)
    events: list[str] = []
    calls: list[dict[str, object]] = []
    runner = _successful_subprocess(calls)
    monkeypatch.setattr(suite, "_load_contract", lambda: contract)
    monkeypatch.setattr(suite, "_source_identity", lambda unused: _source_identity())

    def process(*args, **kwargs):
        events.append("pytest")
        return runner(*args, **kwargs)

    def cuda():
        events.append("cuda")
        return _cuda_environment()

    monkeypatch.setattr(suite.subprocess, "run", process)
    monkeypatch.setattr(suite, "_query_cuda_environment", cuda)

    suite.run_suite(
        run_directory=tmp_path / "ordered",
        generation_artifact=artifact,
        cuda_visible_devices="0",
    )

    assert events == ["pytest", "cuda"]
