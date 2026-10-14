#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "eiel_dirty_tree_ownership_v1"


def _git(repo: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def _status_entries(scope: Path) -> list[tuple[str, str]]:
    records = _git(scope, "status", "--porcelain=v2", "-z", "--untracked-files=all", "--", ".").split(b"\0")
    entries: list[tuple[str, str]] = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        kind = record[:1]
        if kind == b"1":
            fields = record.split(b" ", 8)
            status = fields[1].decode("ascii")
            raw_path = fields[8]
        elif kind == b"2":
            fields = record.split(b" ", 9)
            status = fields[1].decode("ascii")
            raw_path = fields[9]
            index += 1  # The next NUL record is the original rename/copy path.
        elif kind == b"u":
            fields = record.split(b" ", 10)
            status = fields[1].decode("ascii")
            raw_path = fields[10]
        elif kind in {b"?", b"!"}:
            status = kind.decode("ascii")
            raw_path = record[2:]
        else:
            raise RuntimeError(f"unsupported git porcelain-v2 record: {record[:80]!r}")
        entries.append((status, os.fsdecode(raw_path)))
    return sorted(entries, key=lambda item: item[1])


def _hash_path(repo: Path, relative_path: str) -> dict[str, Any]:
    path = repo / relative_path
    if path.is_symlink():
        payload = os.readlink(path).encode("utf-8", errors="surrogateescape")
        return {
            "kind": "symlink",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    if path.is_file():
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        return {"kind": "file", "bytes": size, "sha256": digest.hexdigest()}
    if not path.exists():
        return {"kind": "missing", "bytes": 0, "sha256": None}
    return {"kind": "directory", "bytes": 0, "sha256": None}


def _classification(path: str, scope_path: str, preserve_globs: list[str]) -> str:
    if any(
        fnmatch.fnmatchcase(candidate, pattern)
        for pattern in preserve_globs
        for candidate in (path, scope_path)
    ):
        return "preserved_unrelated"
    return "flat_final_closeout_owned"


def build_manifest(repo: Path, preserve_globs: list[str]) -> dict[str, Any]:
    scope = repo.resolve()
    git_root = Path(_git(scope, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    scope_prefix = scope.relative_to(git_root)
    rows = []
    for status, relative_path in _status_entries(scope):
        relative = Path(relative_path)
        try:
            scope_path = str(relative.relative_to(scope_prefix))
        except ValueError:
            scope_path = relative_path
        rows.append(
            {
                "path": relative_path,
                "scope_path": scope_path,
                "status": status,
                "classification": _classification(relative_path, scope_path, preserve_globs),
                **_hash_path(git_root, relative_path),
            }
        )
    canonical_rows = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path_set = json.dumps([row["path"] for row in rows], separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo": str(scope),
        "git_root": str(git_root),
        "branch": _git(scope, "branch", "--show-current").decode().strip(),
        "head": _git(scope, "rev-parse", "HEAD").decode().strip(),
        "preserve_globs": preserve_globs,
        "status_path_count": len(rows),
        "owned_path_count": sum(row["classification"] == "flat_final_closeout_owned" for row in rows),
        "preserved_unrelated_path_count": sum(row["classification"] == "preserved_unrelated" for row in rows),
        "path_set_sha256": hashlib.sha256(path_set).hexdigest(),
        "dirty_tree_digest": hashlib.sha256(canonical_rows).hexdigest(),
        "entries": rows,
    }


def _comparison_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "repo": manifest.get("repo"),
        "git_root": manifest.get("git_root"),
        "branch": manifest.get("branch"),
        "head": manifest.get("head"),
        "preserve_globs": manifest.get("preserve_globs"),
        "status_path_count": manifest.get("status_path_count"),
        "path_set_sha256": manifest.get("path_set_sha256"),
        "dirty_tree_digest": manifest.get("dirty_tree_digest"),
        "entries": manifest.get("entries"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate or verify an EIEL dirty-tree ownership manifest")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--preserve-glob", action="append", default=[])
    args = parser.parse_args()

    if bool(args.output) == bool(args.verify):
        parser.error("exactly one of --output or --verify is required")
    preserve_globs = list(args.preserve_glob)
    current = build_manifest(args.repo, preserve_globs)
    if args.verify:
        expected = json.loads(args.verify.expanduser().resolve().read_text(encoding="utf-8"))
        if _comparison_payload(current) != _comparison_payload(expected):
            print(json.dumps({"ok": False, "reason": "dirty_tree_manifest_mismatch", "current": current}, indent=2))
            return 1
        print(json.dumps({"ok": True, "dirty_tree_digest": current["dirty_tree_digest"]}, indent=2))
        return 0

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(output), "dirty_tree_digest": current["dirty_tree_digest"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
