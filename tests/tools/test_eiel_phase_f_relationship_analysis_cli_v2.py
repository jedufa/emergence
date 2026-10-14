from __future__ import annotations

from argparse import Namespace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

from evolution.phase_f_v2.canonical import canonical_sha256
from evolution.phase_f_v2.storage import atomic_write_canonical_json


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/scripts/eiel_phase_f_relationship_analysis_v2.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "eiel_phase_f_relationship_analysis_v2_cli", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _finish(value):
    value["receipt_sha256"] = canonical_sha256(value)
    return value


def test_loaders_bind_manifest_ledger_receipt_and_observation(
    tmp_path: Path,
) -> None:
    cli = _module()
    observation = {"surface": [1.0]}
    cpu_expectation = {
        "case_id": "case",
        "case_sha256": "1" * 64,
        "fixture_sha256": "2" * 64,
        "execution_source_sha256": "3" * 64,
        "expectation_kind": "uninterrupted",
        "expectation": observation,
        "expectation_sha256": canonical_sha256(observation),
        "cpu_expectation_evidence": {
            "status": "passed",
            "expectation_sha256": canonical_sha256(observation),
        },
    }
    _finish(cpu_expectation)
    cpu_path = tmp_path / "cpu.json"
    atomic_write_canonical_json(cpu_path, cpu_expectation)
    manifest = {
        "expectation_count": 1,
        "records": [
            {
                "case_id": "case",
                "expectation_kind": "uninterrupted",
                "fixture_sha256": "2" * 64,
                "source_sha256": "3" * 64,
                "expectation_sha256": canonical_sha256(observation),
                "receipt_sha256": cpu_expectation["receipt_sha256"],
                "path": str(cpu_path),
                "file_sha256": _file_sha(cpu_path),
            }
        ],
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = tmp_path / "cpu_manifest.json"
    atomic_write_canonical_json(manifest_path, manifest)

    invocation = {
        "case_id": "case",
        "case_sha256": "1" * 64,
        "execution_id": "case.uninterrupted.0",
        "invocation_kind": "uninterrupted",
        "repeat_index": 0,
        "fixture_sha256": "2" * 64,
        "source_sha256": "3" * 64,
        "expected_sha256": canonical_sha256(observation),
        "observed_sha256": canonical_sha256(observation),
        "owner_id": "owner",
        "profile_id": "profile",
        "lane_id": "lane",
        "outcome": "no_mismatch",
    }
    _finish(invocation)
    receipt_path = tmp_path / "receipt.json"
    observation_path = tmp_path / "observation.json"
    atomic_write_canonical_json(receipt_path, invocation)
    atomic_write_canonical_json(observation_path, observation)
    ledger = {
        "completed_count": 1,
        "completed": [
            {
                "execution_id": "case.uninterrupted.0",
                "case_id": "case",
                "invocation_kind": "uninterrupted",
                "receipt_sha256": invocation["receipt_sha256"],
                "receipt_path": str(receipt_path),
                "receipt_file_sha256": _file_sha(receipt_path),
                "observed_sha256": canonical_sha256(observation),
                "observation_path": str(observation_path),
                "observation_file_sha256": _file_sha(observation_path),
            }
        ],
    }
    ledger["ledger_sha256"] = canonical_sha256(ledger)
    ledger_path = tmp_path / "ledger.json"
    atomic_write_canonical_json(ledger_path, ledger)

    cpu = cli._load_cpu_bindings(manifest_path)
    accelerated = cli._load_accelerated_bindings(ledger_path)
    assert cpu[0]["oracle_receipt_sha256"] == cpu_expectation["receipt_sha256"]
    assert accelerated[0]["outcome"] == "passed"
    assert accelerated[0]["observed_sha256"] == canonical_sha256(observation)


def test_analyze_writes_all_relationship_outputs_atomically(
    monkeypatch, tmp_path: Path
) -> None:
    cli = _module()
    matrix = {"matrix_sha256": "1" * 64}
    fixture = {"cases": []}
    matrix_path = tmp_path / "matrix.json"
    fixture_path = tmp_path / "fixture.json"
    cpu_manifest = tmp_path / "cpu_manifest.json"
    battery_ledger = tmp_path / "ledger.json"
    for path, value in (
        (matrix_path, matrix),
        (fixture_path, fixture),
        (cpu_manifest, {}),
        (battery_ledger, {}),
    ):
        path.write_text(json.dumps(value), encoding="ascii")

    monkeypatch.setattr(cli, "load_phase_f_v2_matrix", lambda path: matrix)
    monkeypatch.setattr(
        cli, "_load_cpu_bindings", lambda path: [{"case_id": "cpu", "expectation_kind": "uninterrupted"}]
    )
    monkeypatch.setattr(
        cli,
        "_load_accelerated_bindings",
        lambda path: [{"execution_id": "execution"}],
    )
    repeats = [{"case_id": "repeat", "invocation_kind": "uninterrupted"}]
    causal = [{"case_id": "causal"}]
    checkpoints = [{"case_id": "checkpoint"}]
    monkeypatch.setattr(
        cli, "build_phase_f_repeat_relationships_v2", lambda **kwargs: repeats
    )
    monkeypatch.setattr(
        cli, "build_phase_f_causal_relationships_v2", lambda **kwargs: causal
    )
    monkeypatch.setattr(
        cli,
        "build_phase_f_checkpoint_relationships_v2",
        lambda **kwargs: checkpoints,
    )
    terminal = {
        "decision": "PHASE-F-V2-A",
        "receipt_sha256": "2" * 64,
    }
    monkeypatch.setattr(
        cli,
        "aggregate_phase_f_relationship_terminal_v2",
        lambda **kwargs: terminal,
    )
    output = tmp_path / "analysis"
    result = cli.analyze(
        Namespace(
            matrix=matrix_path,
            fixture_registry=fixture_path,
            cpu_manifest=cpu_manifest,
            battery_ledger=battery_ledger,
            output_directory=output,
        )
    )
    assert result["ok"] is True
    assert result["decision"] == "PHASE-F-V2-A"
    assert (output / "cpu_bindings/cpu.uninterrupted.json").is_file()
    assert (output / "accelerated_bindings/execution.json").is_file()
    assert (output / "repeat_relationships/repeat.uninterrupted.json").is_file()
    assert (output / "causal_relationships/causal.json").is_file()
    assert (output / "checkpoint_relationships/checkpoint.json").is_file()
    assert (output / "terminal_ledger.json").is_file()
    assert (output / "analysis_summary.json").is_file()
