from __future__ import annotations

import json
from pathlib import Path

from tools.scripts.eiel_option_d_fixture_compare_v1 import compare_fixture_directories


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def test_d1_compare_accepts_environment_only_metadata_drift(tmp_path: Path) -> None:
    accepted = tmp_path / "accepted"
    staged = tmp_path / "staged"
    (accepted / "states").mkdir(parents=True)
    (staged / "states").mkdir(parents=True)
    (accepted / "states" / "state.bin").write_bytes(b"same-state")
    (staged / "states" / "state.bin").write_bytes(b"same-state")
    (accepted / "MANIFEST.sha256").write_text("accepted manifest\n")
    (staged / "MANIFEST.sha256").write_text("staged manifest\n")
    base = {
        "schema_version": "eiel_option_d_d1_fixture_corpus_v1",
        "qualified_scope": {
            "algorithm": "PyTorch CPU at::mt19937",
            "verifier_torch_version": "2.10.0+cpu",
        },
        "environment": {"version": "2.10.0+cpu", "python_version": "3.13.5"},
        "fixtures": [
            {
                "name": "seed0",
                "state_sha256": "a" * 64,
                "source": {"torch_version": "2.10.0+cpu", "torch_state_bytes": 5056},
            }
        ],
    }
    production = json.loads(json.dumps(base))
    production["qualified_scope"]["verifier_torch_version"] = "2.10.0+cu128"
    production["environment"] = {"version": "2.10.0+cu128", "python_version": "3.14.4"}
    production["fixtures"][0]["source"]["torch_version"] = "2.10.0+cu128"
    _write_json(accepted / "d1_fixture_corpus_v1.json", base)
    _write_json(staged / "d1_fixture_corpus_v1.json", production)

    receipt = compare_fixture_directories(profile="d1", accepted=accepted, staged=staged)

    assert receipt["passed"] is True
    statuses = {row["path"]: row["status"] for row in receipt["rows"]}
    assert statuses["d1_fixture_corpus_v1.json"] == "metadata_ignored_payload_match"
    assert statuses["states/state.bin"] == "byte_identical"


def test_d1_compare_rejects_payload_drift(tmp_path: Path) -> None:
    accepted = tmp_path / "accepted"
    staged = tmp_path / "staged"
    payload = {
        "schema_version": "eiel_option_d_d1_fixture_corpus_v1",
        "fixtures": [{"name": "seed0", "state_sha256": "a" * 64}],
    }
    changed = {
        "schema_version": "eiel_option_d_d1_fixture_corpus_v1",
        "fixtures": [{"name": "seed0", "state_sha256": "b" * 64}],
    }
    _write_json(accepted / "d1_fixture_corpus_v1.json", payload)
    _write_json(staged / "d1_fixture_corpus_v1.json", changed)

    receipt = compare_fixture_directories(profile="d1", accepted=accepted, staged=staged)

    assert receipt["passed"] is False
    assert receipt["failures"][0]["status"] == "normalized_payload_mismatch"


def test_d3_compare_accepts_math_environment_only_drift(tmp_path: Path) -> None:
    accepted = tmp_path / "accepted"
    staged = tmp_path / "staged"
    accepted.mkdir()
    staged.mkdir()
    (accepted / "MANIFEST.sha256").write_text("accepted manifest\n")
    (staged / "MANIFEST.sha256").write_text("staged manifest\n")
    base = {
        "schema_version": "eiel_option_d_d3_fixture_corpus_v1",
        "math_environment": {"python": "3.13.5", "libc": ["glibc", "2.41"]},
        "count_fixtures": [{"seed": 1, "raw_words_consumed": 4}],
    }
    production = json.loads(json.dumps(base))
    production["math_environment"] = {
        "python": "3.14.4",
        "libc": ["glibc", "2.43"],
    }
    _write_json(accepted / "d3_fixture_corpus_v1.json", base)
    _write_json(staged / "d3_fixture_corpus_v1.json", production)

    receipt = compare_fixture_directories(profile="d3", accepted=accepted, staged=staged)

    assert receipt["passed"] is True
    assert receipt["rows"][1]["status"] == "metadata_ignored_payload_match"
