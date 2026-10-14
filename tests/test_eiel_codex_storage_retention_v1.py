from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from tools.scripts import eiel_codex_storage_retention as retention


def _session(home: Path, *, session_id: str) -> Path:
    path = home / "sessions" / "2026" / "01" / "01" / f"rollout-2026-01-01T00-00-00-{session_id}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {"session_id": session_id, "id": session_id},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_retention_dry_run_is_fail_closed_and_config_bounded(tmp_path: Path) -> None:
    home = tmp_path / "codex"
    home.mkdir()
    (home / "config.toml").write_text('model = "gpt-5.6-luna"\n', encoding="utf-8")
    session_id = "01a00000-0000-7000-8000-000000000000"
    source = _session(home, session_id=session_id)
    os.utime(source, (0, 0))
    report = retention.build_report(
        home=home,
        age_days=30,
        archive_root=tmp_path / "cold",
        apply=False,
        max_history_bytes=1024,
        ensure_config=True,
    )
    assert report["history_config"]["would_set_max_bytes"] == 1024
    assert report["candidate_count"] == 1
    assert report["actions"][0]["status"] == "eligible"
    assert source.is_file()


def test_retention_uses_supported_delete_after_verified_copy(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "codex"
    session_id = "01a00000-0000-7000-8000-000000000001"
    source = _session(home, session_id=session_id)
    row = retention._session_rows(home)[0]
    monkeypatch.setattr(
        retention.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "", ""),
    )
    result = retention._archive_and_delete_session(
        row,
        archive_root=tmp_path / "cold",
        apply=True,
    )
    assert result["status"] == "deleted_via_supported_cli"
    assert Path(result["archive_path"]).read_bytes() == source.read_bytes()
    assert result["delete_command"] == ["codex", "delete", "--force", session_id]


def test_retention_skips_rollout_that_disappears_during_archive(tmp_path: Path) -> None:
    home = tmp_path / "codex"
    session_id = "01a00000-0000-7000-8000-000000000002"
    source = _session(home, session_id=session_id)
    row = retention._session_rows(home)[0]
    source.unlink()

    result = retention._archive_and_delete_session(
        row,
        archive_root=tmp_path / "cold",
        apply=True,
    )

    assert result["status"] == "skipped_source_missing"
    assert "disappeared" in result["reason"]
    assert not any(path.is_file() for path in (tmp_path / "cold").rglob("*"))


def test_retention_skips_missing_source_when_archive_already_exists(tmp_path: Path) -> None:
    home = tmp_path / "codex"
    session_id = "01a00000-0000-7000-8000-000000000003"
    source = _session(home, session_id=session_id)
    row = retention._session_rows(home)[0]
    archive = tmp_path / "cold" / source.parent.name / source.name
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"previous-copy")
    source.unlink()

    result = retention._archive_and_delete_session(
        row,
        archive_root=tmp_path / "cold",
        apply=True,
    )

    assert result["status"] == "skipped_source_missing"
    assert archive.read_bytes() == b"previous-copy"
