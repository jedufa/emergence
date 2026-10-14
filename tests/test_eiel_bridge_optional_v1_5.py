from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.eiel_actor_v11_helpers import run_eiel
from tests.eiel_actor_v14_helpers import make_bootstrap_repo, run_bootstrap
from tools.actors import workspace

ROOT = Path(__file__).resolve().parents[1]


def test_bridge_status_is_optional_and_reports_actual_installation(tmp_path: Path) -> None:
    result = run_eiel(["bridgeManager", "status"], state_root=tmp_path / "state")
    assert result["ok"] is True
    assert result["bridge"]["required"] is False
    expected = "OPTIONAL_AVAILABLE" if (ROOT / "bridge").is_dir() else "OPTIONAL_NOT_INSTALLED"
    assert result["bridge"]["status"] == expected
    assert result["bridge"]["automatic_execution"] is False
    assert result["bridge"]["authority_effect"] == "none"


def test_missing_bridge_is_not_a_core_workflow_blocker(tmp_path: Path) -> None:
    missing = tmp_path / "not-installed"
    result = run_eiel(
        ["bridgeManager", "status", "--bridge-root", str(missing)],
        state_root=tmp_path / "state",
    )
    assert result["bridge"]["status"] == "OPTIONAL_NOT_INSTALLED"
    assert result["bridge"]["ordinary_actor_workflow_blocked"] is False
    assert result["next_actions"][0]["command"] == "python -m tools.eiel workspace open"


def test_workspace_only_reports_bridge_presence_without_invoking_it(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    bootstrap_id = run_bootstrap(repo=repo, state_root=state)["bootstrap"]["bootstrap_id"]
    result = run_eiel(["workspace", "open", "--bootstrap-record", bootstrap_id, "--state-root", str(state)], state_root=state)
    bridge = result["workspace"]["bridge"]
    assert bridge == {
        "required": False,
        "status": "available_optional" if (repo / "bridge").is_dir() else "not_installed_optional",
        "manager_command": "python -m tools.eiel bridgeManager status",
        "automatic_execution": False,
        "authority_effect": "none",
    }


def test_workspace_reports_not_installed_without_becoming_blocked(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(workspace, "repository_root", lambda: tmp_path / "empty-repo")
    assert workspace._optional_bridge_summary() == {
        "required": False,
        "status": "not_installed_optional",
        "manager_command": "python -m tools.eiel bridgeManager status",
        "automatic_execution": False,
        "authority_effect": "none",
    }


def test_bridge_source_manifest_matches_integrated_source() -> None:
    manifest = ROOT / "bridge/provenance/EIEL_ChatGPT_Bridge_Manager_V1_Source_Manifest_SHA256.txt"
    if not manifest.is_file():
        pytest.skip("optional Bridge source is not included in this transfer")
    import hashlib

    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, rel = line.split(None, 1)
        path = ROOT / rel.strip()
        assert path.is_file(), rel
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, rel
        checked += 1
    assert checked == 58


def test_bridge_provenance_declares_no_core_requirement() -> None:
    path = ROOT / "bridge/provenance/V1_5_INTEGRATION.json"
    if not path.is_file():
        pytest.skip("optional Bridge provenance is not included in this transfer")
    provenance = json.loads(path.read_text(encoding="utf-8"))
    assert provenance["required_by_core_actor_workflow"] is False
    assert provenance["integration_policy"] == "optional_explicit_transport_only"
    assert provenance["authority_effect"] == "none"
