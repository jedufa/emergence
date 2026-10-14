from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tests.evidence.cogp1_fixtures import cogp1_manifest
from tests.eiel_actor_v11_helpers import FIXED, REPO_ROOT, run_eiel


def _rewrite_manifest(path: Path, **updates: object) -> Path:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.update(updates)
    path.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
    return path


def test_evidence_manager_blocks_unknown_profile_missing_qualification_and_source_mismatch(tmp_path: Path) -> None:
    state = tmp_path / "state"
    manifest = cogp1_manifest(tmp_path / "campaign", signal="supported")

    unknown = run_eiel(
        [
            "evidenceManager", "assess", str(manifest), "--profile", "UNKNOWN-PROFILE",
            "--out-dir", str(tmp_path / "unknown"), "--deterministic", "--generated-at", FIXED,
        ],
        state_root=state,
        expected=2,
    )
    assert "decision_profile_not_implemented" in unknown["reason_codes"]
    assert unknown["status"] == "BLOCKED"

    _rewrite_manifest(manifest, technical_qualification_required=True)
    missing = run_eiel(
        [
            "evidenceManager", "assess", str(manifest), "--out-dir", str(tmp_path / "missing"),
            "--deterministic", "--generated-at", FIXED,
        ],
        state_root=state,
        expected=2,
    )
    assert "technical_qualification_missing" in missing["reason_codes"]
    assert any("qualificationManager check" in action.get("command", "") for action in missing["next_actions"])

    _rewrite_manifest(manifest, technical_qualification_required=False)
    mismatch = run_eiel(
        [
            "evidenceManager", "validate", str(manifest), "--expected-source-commit", "not-the-fixture-commit",
        ],
        state_root=state,
        expected=1,
    )
    assert "source_identity_mismatch" in mismatch["reason_codes"]
    assert mismatch["status"] == "BLOCKED"


def test_evidence_manager_blocks_reward_leakage_and_preserves_claim_ceiling(tmp_path: Path) -> None:
    state = tmp_path / "state"
    manifest = cogp1_manifest(tmp_path / "campaign", signal="supported")
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    summary = json.loads((Path(raw["run_root"]) / "summary.json").read_text(encoding="utf-8"))
    metrics = Path(summary["runs"][0]["run_dir"]) / "metrics.csv"
    rows = list(csv.DictReader(metrics.open(encoding="utf-8")))
    columns = list(rows[0])
    rows[0]["direct_cognition_reward_present_flag"] = "1"
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    blocked = run_eiel(
        [
            "evidenceManager", "assess", str(manifest), "--out-dir", str(tmp_path / "report"),
            "--deterministic", "--generated-at", FIXED,
        ],
        state_root=state,
        expected=2,
    )
    assert blocked["status"] == "BLOCKED"
    assert any("reward leakage" in problem.lower() for problem in blocked["problems"])
    assert any(action.get("command", "").startswith("python -m tools.eiel evidenceManager inspect") for action in blocked["next_actions"])


def test_repository_real_campaign_output_fixture_is_explicitly_unavailable() -> None:
    manifests = [
        path for path in REPO_ROOT.rglob("campaign_evidence_manifest.json")
        if "tests" not in path.parts and "tools/actors/examples" not in str(path)
    ]
    if not manifests:
        pytest.skip(
            "repository snapshot contains sealed evidence ZIPs and reports but no raw campaign_evidence_manifest.json plus run outputs; "
            "EDM end-to-end verification therefore uses the existing COG-P1 synthetic fixture"
        )
    assert manifests
