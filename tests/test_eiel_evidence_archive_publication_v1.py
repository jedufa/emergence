from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tools.evidence import archive_publication_v1 as pub


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "tools/evidence/librarian/catalog_source/00-catalog"


def _paths(tmp_path: Path) -> pub.PublicationPaths:
    repo = tmp_path / "repo"
    shutil.copytree(SOURCE_ROOT, repo / "tools/evidence/librarian/catalog_source/00-catalog")
    evidence = repo / "evidences"
    evidence.mkdir(parents=True)
    (evidence / "OLD_CAMPAIGN").mkdir()
    (evidence / "OLD_CAMPAIGN" / "Result Report.md").write_text("# Result\n", encoding="utf-8")
    (evidence / "OLD_CAMPAIGN" / "raw_metrics.csv").write_text("generation,value\n0,1\n", encoding="utf-8")
    (evidence / "OLD_CAMPAIGN" / "small.evidence.zip").write_bytes(b"not a real zip, just identity payload")
    return pub.PublicationPaths(
        repo_root=repo,
        evidence_root=evidence,
        source_root=repo / "tools/evidence/librarian/catalog_source/00-catalog",
        runtime_root=tmp_path / "runtime",
        freeze_root=tmp_path / "runtime/freeze",
        candidate_root=tmp_path / "runtime/published_archive_candidate/evidences",
        raw_root=tmp_path / "runtime/relocated_raw",
    )


def test_archive_publication_assigns_ids_and_keeps_raw_out_of_candidate(tmp_path: Path, monkeypatch) -> None:
    paths = _paths(tmp_path)

    def fake_git(repo_root: Path, *args: str, check: bool = True) -> str:
        if args == ("rev-parse", "HEAD"):
            return "a" * 40
        if args == ("branch", "--show-current"):
            return "test"
        if args == ("ls-files", "evidences"):
            return "\n".join(f"evidences/{p.relative_to(paths.evidence_root).as_posix()}" for p in pub.iter_evidence_files(paths.evidence_root))
        if args == ("status", "--short"):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(pub, "git", fake_git)
    monkeypatch.setattr(pub.wp4, "_git", fake_git)
    freeze = pub.freeze_archive(paths)
    plan = pub.build_plan(paths, freeze)
    pub.build_candidate(paths, plan)
    verification = pub.verify_candidate(paths, freeze, plan)

    assert verification["ok"] is True
    assert plan["disposition_counts"]["report_facing"] == 2
    assert plan["disposition_counts"]["runtime_relocated"] == 1
    ids = {row["permanent_file_id"] for row in plan["files"]}
    assert ids == {"EIEL-EV-000001", "EIEL-EV-000002", "EIEL-EV-000003"}
    report_rows = [row for row in plan["files"] if row["archive_disposition"] == "report_facing"]
    assert all(str(row["new_path"]).startswith("evidences/000_programme-methods/") for row in report_rows)
    assert not list(paths.candidate_root.rglob("*.csv"))
    redirect = json.loads((paths.candidate_root / "990_legacy-redirects/archive-publication-v1-map.json").read_text())
    assert len(redirect["files"]) == 3


def test_archive_publication_refuses_publish_without_clean_worktree(tmp_path: Path, monkeypatch) -> None:
    paths = _paths(tmp_path)

    def fake_git(repo_root: Path, *args: str, check: bool = True) -> str:
        if args == ("status", "--short"):
            return " M evidences/x.md"
        raise AssertionError(args)

    monkeypatch.setattr(pub, "git", fake_git)

    try:
        pub.publish_candidate(paths, {"ok": True})
    except pub.PublicationError as exc:
        assert "worktree must be clean" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("publish did not fail closed")


def _post_publication_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    evidence = repo / "evidences"
    redirect_root = evidence / "990_legacy-redirects"
    incoming = evidence / "900_incoming"
    incoming.mkdir(parents=True)
    redirect_root.mkdir(parents=True)
    (redirect_root / "archive-publication-v1-map.json").write_text(
        json.dumps(
            {
                "schema_version": "eiel_evidence_archive_publication_v1_legacy_redirect_map",
                "policy": "test",
                "files": [
                    {
                        "archive_disposition": "report_facing",
                        "new_path": "evidences/000_programme-methods/EIEL-EV-000001--old.md",
                        "old_evidence_relative_path": "OLD/old.md",
                        "old_path": "evidences/OLD/old.md",
                        "permanent_file_id": "EIEL-EV-000001",
                        "publication_folder": "000_programme-methods",
                        "sha256": "a" * 64,
                        "size_bytes": 1,
                        "tracked": True,
                    },
                    {
                        "archive_disposition": "runtime_relocated",
                        "new_path": None,
                        "old_evidence_relative_path": "OLD/raw.csv",
                        "old_path": "evidences/OLD/raw.csv",
                        "permanent_file_id": "EIEL-EV-000003",
                        "publication_folder": "990_legacy-redirects",
                        "runtime_path": "/runtime/EIEL-EV-000003--raw.csv",
                        "sha256": "b" * 64,
                        "size_bytes": 1,
                        "tracked": True,
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (redirect_root / "archive-publication-v1-summary.json").write_text(
        json.dumps(
            {
                "schema_version": "eiel_evidence_archive_publication_v1_candidate_summary",
                "file_count": 2,
                "disposition_counts": {"report_facing": 1, "runtime_relocated": 1},
                "folder_counts": {"000_programme-methods": 1},
                "archive_1_0_folder_counts": {"programme-methods": 1},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return repo


def _fake_post_publication_git(monkeypatch: pytest.MonkeyPatch, tracked: set[str], *, dirty: bool = False) -> None:
    def fake_git(repo_root: Path, *args: str, check: bool = True) -> str:
        if args == ("rev-parse", "HEAD"):
            return "c" * 40
        if args == ("status", "--short"):
            return " M unrelated" if dirty else ""
        if len(args) == 3 and args[:2] == ("ls-files", "--error-unmatch"):
            if args[2] in tracked:
                return args[2]
            raise pub.PublicationError("not tracked")
        raise AssertionError(args)

    monkeypatch.setattr(pub, "git", fake_git)


def test_post_publication_accession_moves_one_tracked_report_facing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _post_publication_repo(tmp_path)
    source = repo / "evidences/900_incoming/New Finding.md"
    source.write_text("# New Finding\n", encoding="utf-8")
    _fake_post_publication_git(monkeypatch, {"evidences/900_incoming/New Finding.md"})

    receipt = pub.accession_post_publication_file(
        repo_root=repo,
        source=Path("evidences/900_incoming/New Finding.md"),
        shelf="100_engine-route-qualification",
        expected_head="c" * 40,
    )

    assert receipt["ok"] is True
    assert receipt["permanent_file_id"] == "EIEL-EV-000004"
    assert receipt["destination_path"] == (
        "evidences/100_engine-route-qualification/"
        "EIEL-EV-000004--new-finding.md"
    )
    assert not source.exists()
    destination = repo / receipt["destination_path"]
    assert destination.read_text(encoding="utf-8") == "# New Finding\n"
    redirect = json.loads((repo / "evidences/990_legacy-redirects/archive-publication-v1-map.json").read_text())
    assert redirect["files"][-1]["old_path"] == "evidences/900_incoming/New Finding.md"
    assert redirect["files"][-1]["new_path"] == receipt["destination_path"]
    assert redirect["files"][-1]["authority_effect"] == "none"
    assert redirect["files"][-1]["claims_altered"] is False
    summary = json.loads((repo / "evidences/990_legacy-redirects/archive-publication-v1-summary.json").read_text())
    assert summary["file_count"] == 3
    assert summary["disposition_counts"]["report_facing"] == 2
    assert summary["folder_counts"]["100_engine-route-qualification"] == 1
    assert summary["archive_1_0_folder_counts"]["engine-route-qualification"] == 1


def test_post_publication_accession_detects_unmapped_filesystem_ids_before_move(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _post_publication_repo(tmp_path)
    existing = repo / "evidences/010_archive-indexes/EIEL-EV-000006--unmapped.md"
    existing.parent.mkdir(parents=True)
    existing.write_text("# Existing\n", encoding="utf-8")
    source = repo / "evidences/900_incoming/New Finding.md"
    source.write_text("# New Finding\n", encoding="utf-8")
    _fake_post_publication_git(monkeypatch, {"evidences/900_incoming/New Finding.md"})

    redirect = json.loads(
        (repo / "evidences/990_legacy-redirects/archive-publication-v1-map.json").read_text()
    )
    assert pub.next_permanent_evidence_id(redirect, repo / "evidences") == "EIEL-EV-000007"
    with pytest.raises(pub.PublicationError, match="missing from the redirect map"):
        pub.accession_post_publication_file(
            repo_root=repo,
            source=Path("evidences/900_incoming/New Finding.md"),
            shelf="100_engine-route-qualification",
            expected_head="c" * 40,
        )
    assert source.is_file()


def test_post_publication_accession_rejects_bad_shelf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _post_publication_repo(tmp_path)
    source = repo / "evidences/900_incoming/New Finding.md"
    source.write_text("# New Finding\n", encoding="utf-8")
    _fake_post_publication_git(monkeypatch, {"evidences/900_incoming/New Finding.md"})

    with pytest.raises(pub.PublicationError, match="bad or unapproved"):
        pub.accession_post_publication_file(
            repo_root=repo,
            source=Path("evidences/900_incoming/New Finding.md"),
            shelf="990_legacy-redirects",
            expected_head="c" * 40,
        )


def test_post_publication_accession_rejects_head_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _post_publication_repo(tmp_path)
    source = repo / "evidences/900_incoming/New Finding.md"
    source.write_text("# New Finding\n", encoding="utf-8")
    _fake_post_publication_git(monkeypatch, {"evidences/900_incoming/New Finding.md"})

    with pytest.raises(pub.PublicationError, match="observed HEAD"):
        pub.accession_post_publication_file(
            repo_root=repo,
            source=Path("evidences/900_incoming/New Finding.md"),
            shelf="100_engine-route-qualification",
            expected_head="d" * 40,
        )


def test_post_publication_accession_rejects_dirty_worktree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _post_publication_repo(tmp_path)
    source = repo / "evidences/900_incoming/New Finding.md"
    source.write_text("# New Finding\n", encoding="utf-8")
    _fake_post_publication_git(monkeypatch, {"evidences/900_incoming/New Finding.md"}, dirty=True)

    with pytest.raises(pub.PublicationError, match="worktree must be clean"):
        pub.accession_post_publication_file(
            repo_root=repo,
            source=Path("evidences/900_incoming/New Finding.md"),
            shelf="100_engine-route-qualification",
            expected_head="c" * 40,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("duplicate", "already exists in archive redirect map"),
        ("missing", "missing source"),
        ("untracked", "not tracked"),
        ("symlink", "symlinks are forbidden"),
        ("collision", "destination already exists"),
    ],
)
def test_post_publication_accession_rejects_unsafe_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    repo = _post_publication_repo(tmp_path)
    source = repo / "evidences/900_incoming/New Finding.md"
    tracked = {"evidences/900_incoming/New Finding.md"}
    if case == "duplicate":
        redirect_path = repo / "evidences/990_legacy-redirects/archive-publication-v1-map.json"
        redirect = json.loads(redirect_path.read_text())
        redirect["files"].append(
            {
                "old_path": "evidences/900_incoming/New Finding.md",
                "new_path": "evidences/100_engine-route-qualification/EIEL-EV-000004--new-finding.md",
                "permanent_file_id": "EIEL-EV-000004",
            }
        )
        redirect_path.write_text(json.dumps(redirect, indent=2, sort_keys=True) + "\n")
        source.write_text("# New Finding\n", encoding="utf-8")
    elif case == "missing":
        pass
    elif case == "untracked":
        source.write_text("# New Finding\n", encoding="utf-8")
        tracked = set()
    elif case == "symlink":
        target = repo / "target.md"
        target.write_text("# Target\n", encoding="utf-8")
        source.symlink_to(target)
    elif case == "collision":
        source.write_text("# New Finding\n", encoding="utf-8")
        collision = repo / "evidences/100_engine-route-qualification/EIEL-EV-000004--new-finding.md"
        collision.mkdir(parents=True)
    _fake_post_publication_git(monkeypatch, tracked)

    with pytest.raises(pub.PublicationError, match=message):
        pub.accession_post_publication_file(
            repo_root=repo,
            source=Path("evidences/900_incoming/New Finding.md"),
            shelf="100_engine-route-qualification",
            expected_head="c" * 40,
        )
