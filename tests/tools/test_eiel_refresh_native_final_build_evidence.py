from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

from tools.scripts.eiel_refresh_native_final_build_evidence import (
    build_refreshed_evidence,
    main,
)


ABI_VERSION = "eiel_native_runtime_abi_v1"
EXTENSION_VERSION = "eiel_native_runtime_flat_v1"


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Any]:
    repo = tmp_path / "repo"
    project = repo / "project"
    binding = project / "engine" / "native_runtime" / "native" / "binding.cpp"
    resident = project / "engine" / "perf" / "native" / "resident.cpp"
    build_tool = project / "tools" / "scripts" / "build_fixture.py"
    tracked_python = project / "engine" / "tracked.py"
    for path in (binding, resident, build_tool, tracked_python):
        path.parent.mkdir(parents=True, exist_ok=True)
    binding.write_text("// immutable binding\n", encoding="utf-8")
    resident.write_text("// immutable resident owner\n", encoding="utf-8")
    build_tool.write_text("print('build fixture')\n", encoding="utf-8")
    tracked_python.write_text("VALUE = 'before'\n", encoding="utf-8")

    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "checkout", "-qb", "fixture-branch")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    extension = artifact_dir / "fixture.so"
    extension.write_bytes(b"already-built immutable extension\n")
    manifest_path = extension.with_suffix(".so.manifest.json")
    manifest = {
        "schema_version": "eiel_native_runtime_build_manifest_v1",
        "extension_abi_version": ABI_VERSION,
        "extension_version": EXTENSION_VERSION,
        "output": str(extension),
        "output_sha256": _sha256(extension),
        "source": str(binding),
        "source_sha256": _sha256(binding),
        "resident_owner_source": str(resident),
        "resident_owner_source_sha256": _sha256(resident),
        "runtime_jit_in_measured_execution": False,
    }
    _write_json(manifest_path, manifest)

    command = f"python {build_tool} --output {extension}"
    evidence_path = extension.with_suffix(".so.build-evidence.json")
    evidence = {
        "schema_version": "eiel_native_runtime_final_build_evidence_v1",
        "captured_at": "2000-01-01T00:00:00+00:00",
        "classification": "final_immutable_native_extension_build_evidence",
        "build_count": 1,
        "simulation_runs": 0,
        "artifact": {
            "path": str(extension),
            "sha256": _sha256(extension),
            "tool_manifest_path": str(manifest_path),
            "tool_manifest_sha256": _sha256(manifest_path),
            "extension_abi_version": ABI_VERSION,
            "extension_version": EXTENSION_VERSION,
            "runtime_jit_in_measured_execution": False,
        },
        "build": {
            "command": command,
            "argv": ["python", str(build_tool), "--output", str(extension)],
            "build_tool": {"path": str(build_tool), "sha256": _sha256(build_tool)},
            "source_immutability_monitor": {
                "result": "passed",
                "pre_post_hashes_equal": True,
            },
        },
        "source_hashes": {
            "resident_owner_cpp": {"path": str(resident), "sha256": _sha256(resident)},
            "binding_sources": [
                {
                    "role": "cpp_binding_translation_unit",
                    "path": str(binding),
                    "sha256": _sha256(binding),
                }
            ],
            "accepted_one_tick_exact_source_match": True,
        },
        "python_source_digest": {
            "scope": "project",
            "selection": "stale",
            "file_count": 0,
            "algorithm": "stale",
            "sha256": "stale",
            "stable_across_two_consecutive_passes": False,
        },
        "dirty_tree_digest": {
            "scope": "project",
            "branch": "stale",
            "head": "stale",
            "sha256": "stale",
        },
        "load_validation": {
            "manifest_sha256_verified": True,
            "abi_compatible": True,
            "module_abi_version_verified": True,
            "module_extension_version_verified": True,
            "runtime_jit": False,
            "runtime_jit_in_measured_execution": False,
            "loaded": True,
        },
    }
    _write_json(evidence_path, evidence)
    return {
        "repo": repo,
        "project": project,
        "binding": binding,
        "resident": resident,
        "tracked_python": tracked_python,
        "extension": extension,
        "manifest_path": manifest_path,
        "evidence_path": evidence_path,
        "evidence": evidence,
        "command": command,
    }


def test_refresh_writes_custom_output_with_current_provenance_and_stable_digests(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    custom_output = fixture["extension"].parent / "refreshed.json"

    result = main(
        [
            "--extension",
            str(fixture["extension"]),
            "--output-evidence",
            str(custom_output),
        ]
    )

    assert result == 0
    refreshed = json.loads(custom_output.read_text(encoding="utf-8"))
    assert refreshed["build"]["command"] == fixture["command"]
    assert refreshed["dirty_tree_digest"]["branch"] == "fixture-branch"
    assert refreshed["dirty_tree_digest"]["head"] == _git(fixture["repo"], "rev-parse", "HEAD")
    assert refreshed["dirty_tree_digest"]["stable_across_two_consecutive_passes"] is True
    assert refreshed["python_source_digest"]["file_count"] == 2
    assert refreshed["python_source_digest"]["stable_across_two_consecutive_passes"] is True
    assert fixture["evidence_path"].read_text(encoding="utf-8") == json.dumps(
        fixture["evidence"], indent=2, sort_keys=True
    ) + "\n"


def test_refreshed_payload_is_deterministic_except_captured_at(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    first = build_refreshed_evidence(
        fixture["extension"], fixture["evidence"], captured_at="2026-01-01T00:00:00+00:00"
    )
    second = build_refreshed_evidence(
        fixture["extension"], fixture["evidence"], captured_at="2026-01-02T00:00:00+00:00"
    )

    assert first.pop("captured_at") != second.pop("captured_at")
    assert first == second


def test_check_does_not_write_and_detects_content_only_untracked_change(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    untracked = fixture["project"] / "note.txt"
    untracked.write_text("first\n", encoding="utf-8")
    assert main(["--extension", str(fixture["extension"])]) == 0
    before = fixture["evidence_path"].read_bytes()

    assert main(["--extension", str(fixture["extension"]), "--check"]) == 0
    assert fixture["evidence_path"].read_bytes() == before

    untracked.write_text("other\n", encoding="utf-8")
    assert main(["--extension", str(fixture["extension"]), "--check"]) == 1
    assert fixture["evidence_path"].read_bytes() == before


@pytest.mark.parametrize("tampered_key", ["extension", "binding"])
def test_check_fails_closed_when_immutable_bytes_change(tmp_path: Path, tampered_key: str) -> None:
    fixture = _fixture(tmp_path)
    assert main(["--extension", str(fixture["extension"])]) == 0
    before = fixture["evidence_path"].read_bytes()
    fixture[tampered_key].write_bytes(fixture[tampered_key].read_bytes() + b"tampered\n")

    assert main(["--extension", str(fixture["extension"]), "--check"]) == 1
    assert fixture["evidence_path"].read_bytes() == before


def test_check_rejects_manifest_that_allows_runtime_jit(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    evidence = copy.deepcopy(fixture["evidence"])
    manifest = json.loads(fixture["manifest_path"].read_text(encoding="utf-8"))
    manifest["runtime_jit_in_measured_execution"] = True
    _write_json(fixture["manifest_path"], manifest)
    evidence["artifact"]["tool_manifest_sha256"] = _sha256(fixture["manifest_path"])
    _write_json(fixture["evidence_path"], evidence)
    before = fixture["evidence_path"].read_bytes()

    assert main(["--extension", str(fixture["extension"]), "--check"]) == 1
    assert fixture["evidence_path"].read_bytes() == before
