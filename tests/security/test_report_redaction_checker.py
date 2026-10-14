from __future__ import annotations

from pathlib import Path

from tools.bin.check_report_redaction import scan_path, scan_text


def test_scan_text_flags_local_paths() -> None:
    findings = scan_text("Internal path: /home/synthetic-user/causyl/runtime/live/report.md")

    assert findings == ["/home/synthetic-user/causyl/runtime/live/report.md"]


def test_scan_path_reports_only_files_with_findings(tmp_path: Path) -> None:
    clean = tmp_path / "clean.md"
    dirty = tmp_path / "dirty.md"
    clean.write_text("relative artifact: evidences/04-theme-4-cognition/report.md\n", encoding="utf-8")
    dirty.write_text("publishable report must not include C:\\Users\\User\\Downloads\\raw.txt\n", encoding="utf-8")

    findings = scan_path(tmp_path)

    assert findings == [(dirty, ["C:\\Users\\User\\Downloads\\raw.txt"])]
