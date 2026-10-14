from __future__ import annotations

import subprocess
from pathlib import Path

from tools.scripts.eiel_dirty_tree_ownership import build_manifest


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def test_manifest_hashes_scope_and_preserves_declared_unrelated_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    scope = repo / "scope"
    scope.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test User")
    owned = scope / "owned.txt"
    owned.write_text("before\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "initial")
    owned.write_text("after\n", encoding="utf-8")
    preserved = scope / "preserved-note.txt"
    preserved.write_text("external\n", encoding="utf-8")

    manifest = build_manifest(scope, ["preserved-*"])

    assert manifest["repo"] == str(scope)
    assert manifest["git_root"] == str(repo)
    assert manifest["status_path_count"] == 2
    assert manifest["owned_path_count"] == 1
    assert manifest["preserved_unrelated_path_count"] == 1
    rows = {row["scope_path"]: row for row in manifest["entries"]}
    assert rows["owned.txt"]["kind"] == "file"
    assert rows["owned.txt"]["sha256"]
    assert rows["preserved-note.txt"]["classification"] == "preserved_unrelated"
