"""Installed source provenance reaches the canonical Launchpad/CRCA seam."""
from pathlib import Path
import subprocess

import pytest

from engine.campaigns import execution_front_door_v1 as front
from engine.campaigns.canonical import canonical_sha256


COMMIT = "a" * 40


def _no_git(*args, **kwargs):
    raise subprocess.CalledProcessError(128, "git")


def test_installed_commit_is_carried_without_claiming_clean_worktree(monkeypatch, tmp_path):
    monkeypatch.setenv("EIEL_SOURCE_GIT_COMMIT", COMMIT)
    monkeypatch.setattr(front.subprocess, "check_output", _no_git)
    identity = front._bound_campaign_source_identity_v1(
        plan={"plan_sha256": "1" * 64}, bound_plan={"plan_sha256": "2" * 64},
        source_lineage={"campaign_spec_sha256": "3" * 64},
        science_lineage={}, repo_root=tmp_path,
    )
    git = identity["git"]
    assert git["commit"] == COMMIT
    assert git["identity_available"] is True
    assert git["working_tree_clean"] is None
    assert git["working_tree_status_known"] is False
    assert git["working_tree_status_sha256"] is None
    changed = {**identity, "git": {**git, "commit": "b" * 40}}
    assert canonical_sha256(identity) != canonical_sha256(changed)


@pytest.mark.parametrize("value", ["short", "A" * 40, "g" * 40])
def test_malformed_installed_commit_fails_closed(monkeypatch, value):
    monkeypatch.setenv("EIEL_SOURCE_GIT_COMMIT", value)
    monkeypatch.setattr(front.subprocess, "check_output", _no_git)
    with pytest.raises(front.ExecutionFrontDoorError, match="40-character lowercase"):
        front._git_identity(Path("/installed"))


def test_absent_commit_stays_unavailable(monkeypatch):
    monkeypatch.delenv("EIEL_SOURCE_GIT_COMMIT", raising=False)
    monkeypatch.setattr(front.subprocess, "check_output", _no_git)
    identity = front._git_identity(Path("/installed"))
    assert identity["identity_available"] is False
    assert "commit" not in identity


def test_conflicting_checkout_identity_is_rejected(monkeypatch):
    monkeypatch.setenv("EIEL_SOURCE_GIT_COMMIT", "b" * 40)
    monkeypatch.setattr(front.subprocess, "check_output", lambda *a, **k: COMMIT)
    with pytest.raises(front.ExecutionFrontDoorError, match="conflicts"):
        front._git_identity(Path("/checkout"))


@pytest.mark.parametrize("status", ["", " M file.py"])
def test_checkout_identity_preserves_original_dirty_status(monkeypatch, status):
    monkeypatch.setenv("EIEL_SOURCE_GIT_COMMIT", COMMIT)
    monkeypatch.setattr(front.subprocess, "check_output",
                        lambda argv, **k: COMMIT if "rev-parse" in argv else status)
    identity = front._git_identity(Path("/checkout"))
    assert identity["commit"] == COMMIT
    assert identity["working_tree_clean"] is (not status)
    assert identity["working_tree_status_lines"] == status.strip().splitlines()
    assert "discovery" not in identity
