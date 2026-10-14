from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from tools.scripts import eiel_worktree_audit as audit


def _run(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo, "init", "-b", "main")
    _run(repo, "config", "user.name", "Test")
    _run(repo, "config", "user.email", "test@example.invalid")
    (repo / "tracked.txt").write_text("base\n", encoding="utf-8")
    _run(repo, "add", "tracked.txt")
    _run(repo, "commit", "-m", "base")
    return repo


def _complete_process_snapshot(*paths: Path) -> dict:
    return {
        "state": "complete",
        "permission_errors": 0,
        "truncated": False,
        "fd_links_scanned": 0,
        "matches_by_worktree": {str(path.resolve()): [] for path in paths},
    }


def test_inventory_classifies_contained_and_uncontained_branches(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _run(repo, "branch", "contained")
    _run(repo, "switch", "-c", "uncontained")
    (repo / "unique.txt").write_text("unique\n", encoding="utf-8")
    _run(repo, "add", "unique.txt")
    _run(repo, "commit", "-m", "unique")
    _run(repo, "switch", "main")

    result = audit.build_inventory(
        repo,
        frontier="main",
        process_snapshot=_complete_process_snapshot(repo),
        generated_at="fixed",
    )
    branches = {row["branch"]: row for row in result["branches"]}
    assert branches["main"]["disposition"] == "preserve_active"
    assert branches["contained"]["disposition"] == "candidate_only_after_explicit_authority"
    assert branches["uncontained"]["disposition"] == "preserve_uncontained"
    assert branches["uncontained"]["relation_to_frontier"]["unique_commits_vs_frontier"] == 1


def test_linked_dirty_worktree_and_non_cache_ignored_are_preserved(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    linked = tmp_path / "linked"
    _run(repo, "worktree", "add", "-b", "work/topic", str(linked), "main")
    (linked / "untracked.txt").write_text("keep\n", encoding="utf-8")
    (linked / ".gitignore").write_text("retained.bin\n.pytest_cache/\n", encoding="utf-8")
    _run(linked, "add", ".gitignore")
    _run(linked, "commit", "-m", "ignore policy")
    _run(repo, "merge", "--ff-only", "work/topic")
    (linked / "retained.bin").write_text("retained\n", encoding="utf-8")
    (linked / ".pytest_cache").mkdir()
    (linked / ".pytest_cache" / "cache").write_text("cache\n", encoding="utf-8")

    result = audit.build_inventory(
        repo,
        frontier="main",
        process_snapshot=_complete_process_snapshot(repo, linked),
    )
    row = next(row for row in result["worktrees"] if row["path"] == str(linked.resolve()))
    assert row["disposition"] == "preserve_dirty"
    assert row["status"]["untracked_count"] == 1
    assert row["status"]["non_cache_ignored_paths"] == ["retained.bin"]


def test_active_process_and_partial_collection_fail_closed(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _run(repo, "branch", "other")
    base = _complete_process_snapshot(repo)
    base["matches_by_worktree"][str(repo.resolve())] = [
        {"pid": 123, "command": "worker", "kind": "cwd", "target": str(repo)}
    ]
    active = audit.build_inventory(repo, frontier="other", process_snapshot=base)
    assert active["worktrees"][0]["disposition"] == "preserve_active"

    linked = tmp_path / "linked"
    _run(repo, "worktree", "add", "-b", "contained", str(linked), "main")
    partial = _complete_process_snapshot(repo)
    partial["matches_by_worktree"][str(linked.resolve())] = []
    partial["state"] = "partial"
    ambiguous = audit.build_inventory(repo, frontier="main", process_snapshot=partial)
    linked_row = next(
        row
        for row in ambiguous["worktrees"]
        if row["path"] == str(linked.resolve())
    )
    assert linked_row["disposition"] == "preserve_ambiguous"


def test_detached_clean_worktree_is_ambiguous(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    linked = tmp_path / "detached"
    _run(repo, "worktree", "add", "--detach", str(linked), "main")
    result = audit.build_inventory(
        repo,
        frontier="main",
        process_snapshot=_complete_process_snapshot(repo, linked),
    )
    row = next(row for row in result["worktrees"] if row["path"] == str(linked.resolve()))
    assert row["disposition"] == "preserve_ambiguous"


def test_inventory_digest_changes_with_head_and_status(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    snapshot = _complete_process_snapshot(repo)
    first = audit.build_inventory(repo, frontier="main", process_snapshot=snapshot)
    (repo / "new.txt").write_text("new\n", encoding="utf-8")
    second = audit.build_inventory(repo, frontier="main", process_snapshot=snapshot)
    assert first["inventory_sha256"] != second["inventory_sha256"]
    _run(repo, "add", "new.txt")
    _run(repo, "commit", "-m", "new")
    third = audit.build_inventory(repo, frontier="main", process_snapshot=snapshot)
    assert second["inventory_sha256"] != third["inventory_sha256"]


def test_markdown_contains_required_manifest_fields(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    result = audit.build_inventory(
        repo,
        frontier="main",
        process_snapshot=_complete_process_snapshot(repo),
    )
    text = audit.render_markdown(result)
    assert "Inventory SHA-256" in text
    assert "Candidate status is not deletion authority" in text
    assert "Local Branches" in text
    assert "safe_to_delete" not in text


def test_stdout_cli_is_read_only_and_has_no_mutation_subcommand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    before = _run(repo, "status", "--porcelain=v1", "--untracked-files=all")
    monkeypatch.setattr(
        audit,
        "collect_process_evidence",
        lambda paths: _complete_process_snapshot(*paths),
    )
    assert audit.main(["inspect", "--repo", str(repo), "--frontier", "main"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == audit.SCHEMA_VERSION
    assert _run(repo, "status", "--porcelain=v1", "--untracked-files=all") == before
    parser = audit.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["delete"])


def test_explicit_outputs_require_existing_parent(tmp_path: Path) -> None:
    with pytest.raises(audit.WorktreeAuditError, match="output parent"):
        audit._write_explicit(tmp_path / "missing" / "audit.json", "{}\n")
