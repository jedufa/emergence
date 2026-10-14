#!/usr/bin/env python3
"""Scan reports for local workstation paths before publication."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


LOCAL_PATH_PATTERNS = (
    re.compile(r"/home/[A-Za-z0-9_.-]+/[^\s)`'\"]*"),
    re.compile(r"/mnt/c/Users/[A-Za-z0-9_.-]+/[^\s)`'\"]*"),
    re.compile(r"[A-Za-z]:\\Users\\[A-Za-z0-9_.-]+\\[^\s)`'\"]*"),
)


def iter_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in {".md", ".txt", ".html", ".json", ".yaml", ".yml"}
    )


def scan_text(text: str) -> list[str]:
    findings: list[str] = []
    for pattern in LOCAL_PATH_PATTERNS:
        findings.extend(match.group(0) for match in pattern.finditer(text))
    return sorted(set(findings))


def scan_path(path: Path) -> list[tuple[Path, list[str]]]:
    results: list[tuple[Path, list[str]]] = []
    for file_path in iter_files(path):
        findings = scan_text(file_path.read_text(encoding="utf-8", errors="replace"))
        if findings:
            results.append((file_path, findings))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Flag local absolute paths in publishable reports.")
    parser.add_argument("path", type=Path, help="Report file or directory to scan.")
    parser.add_argument("--fail-on-findings", action="store_true", help="Exit non-zero when local paths are found.")
    args = parser.parse_args()

    findings = scan_path(args.path)
    for file_path, paths in findings:
        for local_path in paths:
            print(f"{file_path}: local_path={local_path}")
    print(f"scanned={len(iter_files(args.path))} files_with_findings={len(findings)}")
    return 1 if findings and args.fail_on_findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
