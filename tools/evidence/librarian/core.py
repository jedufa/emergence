"""Read-only EIEL evidence librarian core.

This module intentionally depends only on the standard library.  It scans files
and ZIP metadata as inert data, writes generated views to an explicit output
root, and keeps historical evidence immutable.
"""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

import argparse
import csv
import hashlib
import html
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "eiel_evidence_librarian_audit_v1"
CATALOG_SCHEMA_VERSION = "eiel_evidence_catalog_generated_v1"
BUILD_SCHEMA_VERSION = "eiel_catalog_build_v1"
NORMALIZED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
DEFAULT_SIZE_LIMIT_FOR_SECRET_SCAN = 2_000_000
ZIP_BOMB_RATIO_LIMIT = 1000.0
ZIP_BOMB_EXPANSION_LIMIT = 1_000_000_000


class LibrarianError(RuntimeError):
    """Raised for fail-closed librarian errors."""


@dataclass(frozen=True)
class AuditConfig:
    repo_root: Path
    evidence_root: Path
    output_root: Path
    docs_root: Path | None = None
    deterministic: bool = True
    stability_delay_seconds: float = 0.0
    include_sha256: bool = True
    max_secret_scan_bytes: int = DEFAULT_SIZE_LIMIT_FOR_SECRET_SCAN

    def resolved(self) -> "AuditConfig":
        repo = self.repo_root.resolve()
        evidence = _resolve_under(repo, self.evidence_root)
        output = self.output_root.resolve()
        docs = _resolve_under(repo, self.docs_root) if self.docs_root else repo / "docs"
        return AuditConfig(
            repo_root=repo,
            evidence_root=evidence,
            output_root=output,
            docs_root=docs,
            deterministic=self.deterministic,
            stability_delay_seconds=self.stability_delay_seconds,
            include_sha256=self.include_sha256,
            max_secret_scan_bytes=self.max_secret_scan_bytes,
        )


def utc_now_iso(*, deterministic: bool) -> str:
    if deterministic:
        return "1970-01-01T00:00:00Z"
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def canonical_json_bytes(data: Any) -> bytes:
    return (
        json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, data: Any, *, pretty: bool = True) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if pretty:
        text = json.dumps(data, indent=2, sort_keys=True) + "\n"
        path.write_text(text, encoding="utf-8")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    raw = canonical_json_bytes(data)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_under(repo_root: Path, value: Path | None) -> Path:
    if value is None:
        return repo_root
    path = value if value.is_absolute() else repo_root / value
    return path.resolve()


def rel_to(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def git_output(repo_root: Path, args: list[str], *, check: bool = False) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode != 0:
        raise LibrarianError(result.stderr.strip() or "git command failed")
    return result.stdout


def git_path_sets(repo_root: Path, evidence_root: Path) -> dict[str, set[str]]:
    rel_root = rel_to(repo_root, evidence_root)

    def _z(args: list[str]) -> set[str]:
        text = git_output(repo_root, args)
        return {item for item in text.split("\0") if item}

    tracked = _z(["ls-files", "-z", "--", rel_root])
    untracked = _z(["ls-files", "-z", "--others", "--exclude-standard", "--", rel_root])
    ignored = _z(
        ["ls-files", "-z", "--others", "-i", "--exclude-standard", "--", rel_root]
    )
    return {"tracked": tracked, "untracked": untracked, "ignored": ignored}


def tracking_state(rel_repo_path: str, path_sets: Mapping[str, set[str]]) -> str:
    if rel_repo_path in path_sets["tracked"]:
        return "tracked"
    if rel_repo_path in path_sets["ignored"]:
        return "ignored"
    if rel_repo_path in path_sets["untracked"]:
        return "untracked"
    return "untracked_or_external"


def classify_extension(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".evidence.zip"):
        return "evidence_zip"
    if name.endswith(".zip"):
        return "zip"
    if name.endswith(".sha256"):
        return "sha256"
    suffix = path.suffix.lower().lstrip(".")
    return suffix or "no_extension"


def stable_hash_file(
    path: Path, *, stability_delay_seconds: float = 0.0
) -> tuple[str | None, str]:
    before = path.stat()
    if stability_delay_seconds > 0:
        time.sleep(stability_delay_seconds)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    before_tuple = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_mode,
    )
    after_tuple = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_mode,
    )
    if before_tuple != after_tuple:
        return None, "in_progress_or_unstable"
    return digest.hexdigest(), "stable"


def file_inventory(config: AuditConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = config.resolved()
    path_sets = git_path_sets(cfg.repo_root, cfg.evidence_root)
    files: list[dict[str, Any]] = []
    dirs: list[dict[str, Any]] = []
    for current, dirnames, filenames in os.walk(cfg.evidence_root):
        dirnames.sort()
        filenames.sort()
        current_path = Path(current)
        dir_stat = current_path.stat()
        dirs.append(
            {
                "relative_path": rel_to(cfg.evidence_root, current_path)
                if current_path != cfg.evidence_root
                else ".",
                "depth": 0
                if current_path == cfg.evidence_root
                else len(current_path.relative_to(cfg.evidence_root).parts),
                "dir_count": len(dirnames),
                "file_count": len(filenames),
                "mode_octal": oct(stat.S_IMODE(dir_stat.st_mode)),
                "mtime_observed_ns": dir_stat.st_mtime_ns,
            }
        )
        for name in filenames:
            path = current_path / name
            st = path.lstat()
            rel_evidence = rel_to(cfg.evidence_root, path)
            rel_repo = rel_to(cfg.repo_root, path)
            digest: str | None = None
            stability = "not_hashed"
            if cfg.include_sha256 and stat.S_ISREG(st.st_mode):
                digest, stability = stable_hash_file(
                    path, stability_delay_seconds=cfg.stability_delay_seconds
                )
            files.append(
                {
                    "relative_path": rel_evidence,
                    "repo_relative_path": rel_repo,
                    "basename": path.name,
                    "extension_class": classify_extension(path),
                    "size_bytes": st.st_size,
                    "mode_octal": oct(stat.S_IMODE(st.st_mode)),
                    "file_type": "symlink"
                    if stat.S_ISLNK(st.st_mode)
                    else "regular"
                    if stat.S_ISREG(st.st_mode)
                    else "other",
                    "mtime_observed_ns": st.st_mtime_ns,
                    "git_tracking_state": tracking_state(rel_repo, path_sets),
                    "sha256": digest,
                    "stability_state": stability,
                }
            )
    return files, dirs


def type_summary(files: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in files:
        key = str(row["extension_class"])
        item = grouped.setdefault(key, {"extension_class": key, "count": 0, "size_bytes": 0})
        item["count"] += 1
        item["size_bytes"] += int(row["size_bytes"])
    return sorted(grouped.values(), key=lambda item: item["extension_class"])


def duplicate_groups(files: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_hash: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in files:
        digest = row.get("sha256")
        if isinstance(digest, str):
            by_hash[digest].append(row)
    out = []
    for digest, rows in by_hash.items():
        if len(rows) < 2:
            continue
        sizes = {int(row["size_bytes"]) for row in rows}
        out.append(
            {
                "sha256": digest,
                "count": len(rows),
                "size_bytes_each": sorted(sizes),
                "paths": sorted(str(row["relative_path"]) for row in rows),
            }
        )
    return sorted(out, key=lambda item: (-item["count"], item["sha256"]))


def md_zip_pairing(files: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    md = set()
    zips = set()
    for row in files:
        rel = str(row["relative_path"])
        if rel.lower().endswith(".md"):
            md.add(rel[:-3])
        if rel.lower().endswith(".evidence.zip"):
            zips.add(rel[: -len(".evidence.zip")])
    return {
        "paired_exact_stems": sorted(md & zips),
        "markdown_without_evidence_zip": sorted(md - zips),
        "evidence_zip_without_markdown": sorted(zips - md),
        "paired_exact_count": len(md & zips),
        "markdown_without_evidence_zip_count": len(md - zips),
        "evidence_zip_without_markdown_count": len(zips - md),
    }


def parse_sha256_sidecar(text: str) -> str | None:
    first = text.strip().splitlines()[0] if text.strip().splitlines() else ""
    token = first.split()[0] if first.split() else ""
    if re.fullmatch(r"[0-9a-fA-F]{64}", token):
        return token.lower()
    return None


def checksum_report(evidence_root: Path, files: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    by_rel = {str(row["relative_path"]): row for row in files}
    for rel, row in sorted(by_rel.items()):
        if not rel.lower().endswith(".zip"):
            continue
        sidecar_rel = rel + ".sha256"
        sidecar = evidence_root / sidecar_rel
        expected = None
        if sidecar.is_file():
            expected = parse_sha256_sidecar(sidecar.read_text(encoding="utf-8", errors="replace"))
        actual = row.get("sha256")
        rows.append(
            {
                "zip_path": rel,
                "sidecar_path": sidecar_rel if sidecar.is_file() else None,
                "has_adjacent_sidecar": sidecar.is_file(),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "verification_state": "verified"
                if expected and actual == expected
                else "missing_sidecar"
                if not sidecar.is_file()
                else "invalid_sidecar"
                if expected is None
                else "mismatch",
            }
        )
    return rows


def validate_rel_path(name: str) -> list[str]:
    issues = []
    raw = name.replace("\\", "/")
    lowered = raw.lower()
    if not name or name != raw:
        issues.append("non_posix_or_empty_path")
    if raw.startswith("/") or PurePosixPath(raw).is_absolute():
        issues.append("absolute_path")
    if re.match(r"^[a-zA-Z]:[/\\]", name) or raw.startswith("//"):
        issues.append("windows_drive_or_unc_path")
    parts = PurePosixPath(raw).parts
    if any(part in {"", ".", ".."} for part in parts) or ".." in parts:
        issues.append("path_traversal_or_non_normal")
    if len(raw) > 240:
        issues.append("excessive_path_length")
    return sorted(set(issues))


def _zip_mode(info: zipfile.ZipInfo) -> int:
    return (info.external_attr >> 16) & 0xFFFF


def inspect_zip_safety(path: Path) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    members: list[dict[str, Any]] = []
    names: list[str] = []
    try:
        with zipfile.ZipFile(path) as archive:
            bad_crc = archive.testzip()
            if bad_crc is not None:
                issues.append(
                    {
                        "severity": "critical",
                        "category": "zip_crc_failure",
                        "member": bad_crc,
                        "observation": "zipfile.testzip reported first bad member",
                    }
                )
            for info in archive.infolist():
                names.append(info.filename)
                mode = _zip_mode(info)
                path_issues = validate_rel_path(info.filename)
                for issue in path_issues:
                    issues.append(
                        {
                            "severity": "critical"
                            if issue in {"absolute_path", "path_traversal_or_non_normal"}
                            else "high",
                            "category": f"zip_{issue}",
                            "member": info.filename,
                            "observation": "unsafe ZIP member path",
                        }
                    )
                if info.flag_bits & 0x1:
                    issues.append(
                        {
                            "severity": "high",
                            "category": "zip_encrypted_member",
                            "member": info.filename,
                            "observation": "encrypted member cannot be inspected deterministically",
                        }
                    )
                member_type = stat.S_IFMT(mode)
                if stat.S_ISLNK(mode):
                    issues.append(
                        {
                            "severity": "high",
                            "category": "zip_symlink_member",
                            "member": info.filename,
                            "observation": "symlink member is unsafe for direct extraction",
                        }
                    )
                elif member_type and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                    issues.append(
                        {
                            "severity": "high",
                            "category": "zip_special_member",
                            "member": info.filename,
                            "observation": f"unsupported mode {oct(mode)}",
                        }
                    )
                ratio = (
                    float(info.file_size) / float(info.compress_size)
                    if info.compress_size
                    else float("inf")
                    if info.file_size
                    else 0.0
                )
                if (
                    info.file_size > ZIP_BOMB_EXPANSION_LIMIT
                    or ratio > ZIP_BOMB_RATIO_LIMIT
                ):
                    issues.append(
                        {
                            "severity": "high",
                            "category": "zip_unreasonable_expansion",
                            "member": info.filename,
                            "observation": {
                                "file_size": info.file_size,
                                "compressed_size": info.compress_size,
                                "ratio": ratio,
                            },
                        }
                    )
                members.append(
                    {
                        "path": info.filename,
                        "crc": f"{info.CRC:08x}",
                        "compressed_size": info.compress_size,
                        "file_size": info.file_size,
                        "mode_octal": oct(mode) if mode else None,
                        "sha256": None,
                    }
                )
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        issues.append(
            {
                "severity": "critical",
                "category": "zip_unreadable",
                "member": None,
                "observation": f"{type(exc).__name__}: {exc}",
            }
        )
    duplicate_names = sorted(name for name, count in Counter(names).items() if count > 1)
    for name in duplicate_names:
        issues.append(
            {
                "severity": "high",
                "category": "zip_duplicate_member_name",
                "member": name,
                "observation": "duplicate member name",
            }
        )
    for collision_type, normalizer in (
        ("case_collision", lambda value: value.casefold()),
        ("unicode_nfc_collision", lambda value: unicodedata.normalize("NFC", value)),
    ):
        seen: dict[str, str] = {}
        for name in names:
            key = normalizer(name)
            previous = seen.get(key)
            if previous is not None and previous != name:
                issues.append(
                    {
                        "severity": "high",
                        "category": f"zip_{collision_type}",
                        "member": name,
                        "observation": {"first": previous, "second": name},
                    }
                )
            seen.setdefault(key, name)
    return {
        "zip_path": str(path),
        "member_count": len(members),
        "members": sorted(members, key=lambda item: item["path"]),
        "issues": sorted(
            issues,
            key=lambda item: (item["severity"], item["category"], str(item["member"])),
        ),
        "safety_state": "unsafe" if issues else "safe",
    }


def zip_report(evidence_root: Path, files: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in sorted(files, key=lambda item: str(item["relative_path"])):
        rel = str(row["relative_path"])
        if rel.lower().endswith(".zip"):
            report = inspect_zip_safety(evidence_root / rel)
            report["zip_path"] = rel
            out.append(report)
    return out


def json_parse_report(evidence_root: Path, files: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in sorted(files, key=lambda item: str(item["relative_path"])):
        rel = str(row["relative_path"])
        if not rel.lower().endswith(".json"):
            continue
        try:
            value = json.loads((evidence_root / rel).read_text(encoding="utf-8"))
            state = "parsed"
            shape = "object" if isinstance(value, dict) else "array" if isinstance(value, list) else type(value).__name__
            top_keys = sorted(value) if isinstance(value, dict) else []
            error = None
        except Exception as exc:
            state = "failed"
            shape = None
            top_keys = []
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "path": rel,
                "parse_state": state,
                "top_level_shape": shape,
                "top_level_keys": top_keys,
                "error": error,
            }
        )
    return rows


def manifest_shape_summary(json_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    signatures: dict[str, dict[str, Any]] = {}
    manifest_rows = [
        row
        for row in json_rows
        if "manifest" in str(row["path"]).lower() and row["parse_state"] == "parsed"
    ]
    for row in manifest_rows:
        signature = "|".join(str(key) for key in row.get("top_level_keys", []))
        item = signatures.setdefault(signature, {"count": 0, "example_paths": []})
        item["count"] += 1
        if len(item["example_paths"]) < 5:
            item["example_paths"].append(row["path"])
    return {
        "manifest_named_json_count": len(manifest_rows),
        "top_level_signature_count": len(signatures),
        "signatures": [
            {"signature": key, **value}
            for key, value in sorted(signatures.items(), key=lambda item: item[0])
        ],
    }


def docs_evidence_comparison(
    repo_root: Path, docs_root: Path, evidence_root: Path, evidence_files: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    evidence_md: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in evidence_files:
        if str(row["relative_path"]).lower().endswith(".md"):
            evidence_md[str(row["basename"])].append(row)
    docs_md: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if docs_root.exists():
        for path in sorted(docs_root.rglob("*.md")):
            if not path.is_file():
                continue
            docs_md[path.name].append(
                {
                    "repo_relative_path": rel_to(repo_root, path),
                    "sha256": file_sha256(path),
                }
            )
    shared = sorted(set(evidence_md) & set(docs_md))
    rows = []
    for basename in shared:
        evidence_hashes = {
            str(row.get("sha256"))
            for row in evidence_md[basename]
            if isinstance(row.get("sha256"), str)
        }
        docs_hashes = {str(row["sha256"]) for row in docs_md[basename]}
        rows.append(
            {
                "basename": basename,
                "evidence_paths": sorted(str(row["relative_path"]) for row in evidence_md[basename]),
                "docs_paths": sorted(str(row["repo_relative_path"]) for row in docs_md[basename]),
                "has_exact_content_match": bool(evidence_hashes & docs_hashes),
                "evidence_sha256": sorted(evidence_hashes),
                "docs_sha256": sorted(docs_hashes),
            }
        )
    return {
        "shared_markdown_basename_count": len(shared),
        "exact_content_match_count": sum(1 for row in rows if row["has_exact_content_match"]),
        "divergent_same_basename_count": sum(
            1 for row in rows if not row["has_exact_content_match"]
        ),
        "rows": rows,
    }


def iter_text_files(root: Path, *, max_bytes: int) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.stat().st_size > max_bytes:
            continue
        if path.suffix.lower() in {".md", ".json", ".txt", ".csv", ".sha256", ".html"}:
            yield path


def consumer_reference_report(repo_root: Path) -> dict[str, Any]:
    rows = []
    for path in iter_text_files(repo_root, max_bytes=DEFAULT_SIZE_LIMIT_FOR_SECRET_SCAN):
        rel = rel_to(repo_root, path)
        if rel.startswith("evidences/"):
            continue
        try:
            for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if "evidences/" in line:
                    rows.append(
                        {
                            "path": rel,
                            "line": lineno,
                            "text_fragment": line.strip()[:240],
                        }
                    )
        except OSError:
            continue
    return {
        "referencing_file_count": len({row["path"] for row in rows}),
        "reference_count": len(rows),
        "rows": rows,
    }


MD_LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")


def broken_link_report(evidence_root: Path, files: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for row in files:
        rel = str(row["relative_path"])
        if not rel.lower().endswith(".md"):
            continue
        source = evidence_root / rel
        try:
            text = source.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in MD_LINK_RE.finditer(text):
            target = match.group(1).split("#", 1)[0].strip()
            if (
                not target
                or "://" in target
                or target.startswith("#")
                or target.startswith("mailto:")
            ):
                continue
            candidate = (source.parent / target).resolve()
            try:
                candidate.relative_to(evidence_root.resolve())
            except ValueError:
                state = "outside_evidence_root"
            else:
                state = "ok" if candidate.exists() else "missing"
            if state != "ok":
                rows.append({"source": rel, "target": target, "state": state})
    return {"broken_or_external_link_count": len(rows), "rows": rows}


RAW_LOCATOR_RE = re.compile(
    r"(?P<path>" + re.escape(public_runtime_path()) + r"[^\s)'\",]+)"
)


def raw_locator_report(evidence_root: Path, files: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for row in files:
        rel = str(row["relative_path"])
        if not rel.lower().endswith((".md", ".json")):
            continue
        path = evidence_root / rel
        if path.stat().st_size > DEFAULT_SIZE_LIMIT_FOR_SECRET_SCAN:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in RAW_LOCATOR_RE.finditer(text):
            locator = match.group("path").rstrip(".,;")
            rows.append(
                {
                    "source": rel,
                    "locator": locator,
                    "exists_now": Path(locator).exists(),
                }
            )
    return {
        "locator_count": len(rows),
        "missing_locator_count": sum(1 for row in rows if not row["exists_now"]),
        "rows": rows,
    }


SECRET_PATTERNS = (
    ("possible_private_key", re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----")),
    ("possible_token_assignment", re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{16,}")),
)


def secret_scan_report(evidence_root: Path, files: Iterable[Mapping[str, Any]], *, max_bytes: int) -> dict[str, Any]:
    rows = []
    for row in files:
        rel = str(row["relative_path"])
        if not rel.lower().endswith((".md", ".json", ".txt", ".csv", ".env")):
            continue
        path = evidence_root / rel
        if path.stat().st_size > max_bytes:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, pattern in SECRET_PATTERNS:
            if pattern.search(text):
                rows.append(
                    {
                        "path": rel,
                        "category": name,
                        "observation": "pattern matched; historical file not modified",
                    }
                )
    return {"finding_count": len(rows), "rows": rows}


def root_entry_inventory(evidence_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(evidence_root.iterdir(), key=lambda item: item.name):
        st = path.lstat()
        rows.append(
            {
                "name": path.name,
                "entry_type": "directory" if path.is_dir() else "file" if path.is_file() else "other",
                "size_bytes": st.st_size,
                "mode_octal": oct(stat.S_IMODE(st.st_mode)),
            }
        )
    return rows


def tree_digest(files: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(files, key=lambda item: str(item["relative_path"])):
        sha = row.get("sha256")
        if not isinstance(sha, str):
            continue
        line = f"{row['relative_path']}\0{row['size_bytes']}\0{sha}\n"
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def issue_ledger(
    *,
    files: list[dict[str, Any]],
    zip_rows: list[dict[str, Any]],
    checksum_rows: list[dict[str, Any]],
    docs_compare: Mapping[str, Any],
    broken_links: Mapping[str, Any],
    raw_locators: Mapping[str, Any],
    secret_scan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    def add(
        severity: str,
        category: str,
        paths: list[str],
        evidence: Any,
        rule: str,
    ) -> None:
        basis = canonical_json_bytes({"category": category, "paths": paths, "rule": rule})
        issue_id = "eiel-issue-" + hashlib.sha256(basis).hexdigest()[:16]
        issues.append(
            {
                "schema_version": "eiel_issue_v1",
                "issue_id": issue_id,
                "severity": severity,
                "category": category,
                "paths": paths,
                "evidence": evidence,
                "detection_rule": rule,
                "first_observed": "scan_observation",
                "last_observed": "scan_observation",
                "owner": "unassigned",
                "disposition": "legacy_observed_unresolved",
                "exception_expiry": None,
                "remediation_state": "not_started",
            }
        )

    for row in files:
        if row["stability_state"] == "in_progress_or_unstable":
            add(
                "high",
                "in_progress_or_unstable_file",
                [row["relative_path"]],
                {"size_bytes": row["size_bytes"]},
                "stat before hash differed from stat after hash",
            )
        if int(row["size_bytes"]) > 1_000_000 and row["extension_class"] not in {"zip", "evidence_zip"}:
            add(
                "medium",
                "bulky_report_facing_file",
                [row["relative_path"]],
                {"size_bytes": row["size_bytes"], "extension_class": row["extension_class"]},
                "non-zip evidence file exceeds 1 MiB review threshold",
            )
    for row in checksum_rows:
        if row["verification_state"] != "verified":
            add(
                "medium",
                "zip_missing_or_bad_adjacent_checksum",
                [row["zip_path"]],
                row,
                "every zip should have an adjacent verified sha256 sidecar",
            )
    for row in zip_rows:
        for issue in row["issues"]:
            add(
                issue["severity"],
                issue["category"],
                [row["zip_path"]],
                issue,
                "ZIP safety inspection without extraction",
            )
    for row in docs_compare["rows"]:
        if not row["has_exact_content_match"]:
            add(
                "high",
                "docs_evidences_divergent_same_basename",
                [*row["evidence_paths"], *row["docs_paths"]],
                row,
                "same Markdown basename exists in docs and evidences without an exact content match",
            )
    for row in broken_links["rows"]:
        add(
            "medium",
            "broken_or_external_evidence_link",
            [row["source"]],
            row,
            "Markdown link in evidence root does not resolve inside evidence root",
        )
    for row in raw_locators["rows"]:
        if not row["exists_now"]:
            add(
                "medium",
                "missing_raw_locator",
                [row["source"]],
                row,
                "raw evidence locator observed in evidence text does not exist now",
            )
    for row in secret_scan["rows"]:
        add(
            "high",
            "possible_secret_in_evidence",
            [row["path"]],
            row,
            "credential-like pattern matched; emit issue only",
        )
    return sorted(issues, key=lambda item: (item["severity"], item["category"], item["issue_id"]))


def command_receipt(config: AuditConfig) -> dict[str, Any]:
    cfg = config.resolved()
    return {
        "schema_version": "eiel_librarian_command_receipt_v1",
        "repo_root": str(cfg.repo_root),
        "evidence_root": str(cfg.evidence_root),
        "output_root": str(cfg.output_root),
        "branch": git_output(cfg.repo_root, ["branch", "--show-current"]).strip(),
        "head_commit": git_output(cfg.repo_root, ["rev-parse", "HEAD"], check=True).strip(),
        "git_status_short": git_output(cfg.repo_root, ["status", "--short"]).splitlines(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "tool_versions": {
            "git": git_output(cfg.repo_root, ["--version"]).strip(),
            "python": platform.python_version(),
            "zipfile_module": "stdlib",
        },
        "scan_configuration": {
            "deterministic": cfg.deterministic,
            "include_sha256": cfg.include_sha256,
            "stability_delay_seconds": cfg.stability_delay_seconds,
            "exclusions": [],
            "secret_scan_max_bytes": cfg.max_secret_scan_bytes,
        },
    }


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for row in rows:
            raw = canonical_json_bytes(row)
            handle.write(raw)
            digest.update(raw)
    return digest.hexdigest()


def run_audit(config: AuditConfig) -> dict[str, Any]:
    cfg = config.resolved()
    if not cfg.evidence_root.is_dir():
        raise LibrarianError(f"evidence root is not a directory: {cfg.evidence_root}")
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    files, dirs = file_inventory(cfg)
    json_rows = json_parse_report(cfg.evidence_root, files)
    zip_rows = zip_report(cfg.evidence_root, files)
    checksums = checksum_report(cfg.evidence_root, files)
    docs_compare = docs_evidence_comparison(cfg.repo_root, cfg.docs_root or cfg.repo_root / "docs", cfg.evidence_root, files)
    refs = consumer_reference_report(cfg.repo_root)
    broken = broken_link_report(cfg.evidence_root, files)
    raw = raw_locator_report(cfg.evidence_root, files)
    secrets = secret_scan_report(cfg.evidence_root, files, max_bytes=cfg.max_secret_scan_bytes)
    issues = issue_ledger(
        files=files,
        zip_rows=zip_rows,
        checksum_rows=checksums,
        docs_compare=docs_compare,
        broken_links=broken,
        raw_locators=raw,
        secret_scan=secrets,
    )
    digest = tree_digest(files)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(deterministic=cfg.deterministic),
        "repo_root": str(cfg.repo_root),
        "evidence_root": str(cfg.evidence_root),
        "head_commit": git_output(cfg.repo_root, ["rev-parse", "HEAD"], check=True).strip(),
        "file_count": len(files),
        "directory_count": len(dirs),
        "tree_digest_sha256": digest,
        "type_summary": type_summary(files),
        "issue_counts": issue_counts(issues),
        "zip_count": sum(1 for row in files if str(row["relative_path"]).lower().endswith(".zip")),
        "json_count": sum(1 for row in files if str(row["relative_path"]).lower().endswith(".json")),
    }
    outputs = {
        "command_receipt.json": command_receipt(cfg),
        "file_inventory.jsonl": files,
        "directory_inventory.json": dirs,
        "root_entry_inventory.json": root_entry_inventory(cfg.evidence_root),
        "type_count_size_summary.json": type_summary(files),
        "duplicate_groups.json": duplicate_groups(files),
        "markdown_zip_pairing.json": md_zip_pairing(files),
        "adjacent_checksum_report.json": checksums,
        "zip_safety_report.json": zip_rows,
        "json_parse_report.json": json_rows,
        "manifest_shape_summary.json": manifest_shape_summary(json_rows),
        "docs_evidences_same_basename_report.json": docs_compare,
        "evidence_path_consumer_references.json": refs,
        "broken_link_report.json": broken,
        "raw_locator_existence_report.json": raw,
        "secret_pattern_report.json": secrets,
        "issue_ledger.json": issues,
        "baseline_manifest.json": summary,
    }
    output_hashes: dict[str, str] = {}
    for name, payload in outputs.items():
        target = cfg.output_root / name
        if name.endswith(".jsonl"):
            output_hashes[name] = write_jsonl(target, payload)
        else:
            output_hashes[name] = write_json(target, payload)
    tree_path = cfg.output_root / "tree_digest.txt"
    tree_path.write_text(digest + "\n", encoding="utf-8")
    output_hashes["tree_digest.txt"] = file_sha256(tree_path)
    catalog = build_catalog(
        repo_root=cfg.repo_root,
        audit_output_root=cfg.output_root,
        output_root=cfg.output_root / "generated",
        deterministic=cfg.deterministic,
    )
    migration = build_migration_plan(issues)
    output_hashes["migration_plan_candidate.json"] = write_json(
        cfg.output_root / "migration_plan_candidate.json", migration
    )
    authority_queue = authority_drift_queue(issues)
    output_hashes["authority_drift_curation_queue.json"] = write_json(
        cfg.output_root / "authority_drift_curation_queue.json", authority_queue
    )
    summary["outputs"] = output_hashes
    summary["catalog"] = catalog
    write_json(cfg.output_root / "baseline_manifest.json", summary)
    return summary


def issue_counts(issues: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    by_severity: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    for issue in issues:
        by_severity[str(issue["severity"])] += 1
        by_category[str(issue["category"])] += 1
    return {
        "by_severity": dict(sorted(by_severity.items())),
        "by_category": dict(sorted(by_category.items())),
        "total": sum(by_severity.values()),
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_catalog(
    *,
    repo_root: Path,
    audit_output_root: Path,
    output_root: Path,
    deterministic: bool = True,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    baseline = load_json(audit_output_root / "baseline_manifest.json")
    issues = load_json(audit_output_root / "issue_ledger.json")
    files = []
    with (audit_output_root / "file_inventory.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                files.append(json.loads(line))
    issue_by_path: dict[str, list[str]] = defaultdict(list)
    for issue in issues:
        for path in issue.get("paths", []):
            issue_by_path[path].append(issue["issue_id"])
    artifact_entries = []
    for row in files:
        artifact_id = "artifact-" + hashlib.sha256(
            canonical_json_bytes(
                {
                    "path": row["relative_path"],
                    "sha256": row.get("sha256"),
                    "size_bytes": row["size_bytes"],
                }
            )
        ).hexdigest()[:24]
        artifact_entries.append(
            {
                "artifact_id": artifact_id,
                "path": row["relative_path"],
                "artifact_role": "legacy_physical_file_inventory",
                "media_type": row["extension_class"],
                "size_bytes": row["size_bytes"],
                "sha256": row.get("sha256"),
                "integrity_state": row["stability_state"],
                "git_tracking_state": row["git_tracking_state"],
                "issue_ids": sorted(issue_by_path.get(row["relative_path"], [])),
            }
        )
    catalog = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "generated_at": utc_now_iso(deterministic=deterministic),
        "source_commit": git_output(repo_root, ["rev-parse", "HEAD"], check=True).strip(),
        "baseline_tree_digest_sha256": baseline["tree_digest_sha256"],
        "status_axes": [
            "scientific_decision_status",
            "authority_currentness_status",
            "integrity_status",
            "completeness_status",
            "raw_asset_availability_status",
            "reproducibility_status",
            "publication_release_status",
            "retention_status",
        ],
        "authority_policy": {
            "confidence_values": ["explicit", "unknown"],
            "inference_allowed": False,
        },
        "records": [],
        "claims": [],
        "authorities": [],
        "raw_assets": [],
        "artifact_inventory": sorted(artifact_entries, key=lambda item: item["path"]),
        "issues": sorted(issues, key=lambda item: item["issue_id"]),
        "issue_counts": issue_counts(issues),
    }
    catalog_path = output_root / "EIEL_Evidence_Catalog.json"
    catalog_hash = write_json(catalog_path, catalog, pretty=False)
    html_path = output_root / "EIEL_Evidence_Catalog.html"
    html_hash = write_html_catalog(html_path, catalog)
    issues_path = output_root / "EIEL_Evidence_Issues.json"
    issues_hash = write_json(issues_path, issues, pretty=False)
    build_receipt = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "source_commit": catalog["source_commit"],
        "librarian_code_commit": catalog["source_commit"],
        "accession_source_set_digest": baseline["tree_digest_sha256"],
        "build_time": utc_now_iso(deterministic=deterministic),
        "deterministic_mode": deterministic,
        "tool_versions": {"python": platform.python_version()},
        "issue_counts": catalog["issue_counts"],
        "output_hashes": {
            "EIEL_Evidence_Catalog.json": catalog_hash,
            "EIEL_Evidence_Catalog.html": html_hash,
            "EIEL_Evidence_Issues.json": issues_hash,
        },
        "build_status": "success",
    }
    build_hash = write_json(output_root / "EIEL_Evidence_Catalog_Build.json", build_receipt, pretty=False)
    return {
        "catalog_path": str(catalog_path),
        "catalog_sha256": catalog_hash,
        "html_path": str(html_path),
        "html_sha256": html_hash,
        "issues_path": str(issues_path),
        "issues_sha256": issues_hash,
        "build_receipt_path": str(output_root / "EIEL_Evidence_Catalog_Build.json"),
        "build_receipt_sha256": build_hash,
    }


def write_html_catalog(path: Path, catalog: Mapping[str, Any]) -> str:
    rows = catalog["artifact_inventory"]
    issue_counts = catalog["issue_counts"]
    body_rows = []
    for row in rows[:1000]:
        issue_ids = ", ".join(row["issue_ids"])
        body_rows.append(
            "<tr>"
            f"<td>{html.escape(row['path'])}</td>"
            f"<td>{html.escape(str(row['media_type']))}</td>"
            f"<td>{int(row['size_bytes'])}</td>"
            f"<td>{html.escape(str(row['integrity_state']))}</td>"
            f"<td>{html.escape(issue_ids)}</td>"
            "</tr>"
        )
    text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>EIEL Evidence Catalog Candidate</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #20242a; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
th, td {{ border: 1px solid #d4d7dc; padding: 0.35rem 0.5rem; text-align: left; vertical-align: top; }}
th {{ background: #eef1f5; position: sticky; top: 0; }}
.meta {{ display: grid; grid-template-columns: repeat(3, minmax(12rem, 1fr)); gap: 0.75rem; margin-bottom: 1rem; }}
.box {{ border: 1px solid #d4d7dc; padding: 0.75rem; border-radius: 6px; }}
</style>
</head>
<body>
<h1>EIEL Evidence Catalog Candidate</h1>
<p>This generated view is not an authority source. It reports file inventory, integrity, availability, and unresolved issues from the librarian baseline.</p>
<div class="meta">
<div class="box"><strong>Source commit</strong><br>{html.escape(str(catalog['source_commit']))}</div>
<div class="box"><strong>Tree digest</strong><br>{html.escape(str(catalog['baseline_tree_digest_sha256']))}</div>
<div class="box"><strong>Issue count</strong><br>{html.escape(str(issue_counts['total']))}</div>
</div>
<h2>Issue Counts By Severity</h2>
<pre>{html.escape(json.dumps(issue_counts['by_severity'], indent=2, sort_keys=True))}</pre>
<h2>Artifact Inventory Preview</h2>
<p>Showing first {min(1000, len(rows))} of {len(rows)} physical files. Use the JSON catalog for the complete deterministic view.</p>
<table>
<thead><tr><th>Path</th><th>Type</th><th>Bytes</th><th>Integrity</th><th>Issues</th></tr></thead>
<tbody>
{''.join(body_rows)}
</tbody>
</table>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return file_sha256(path)


def build_migration_plan(issues: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "eiel_evidence_migration_plan_candidate_v1",
        "migration_executed": False,
        "policy": "plan_only_no_moves_no_deletes_no_renames",
        "work_packages": [
            "curate authority drift and divergent same-basename reports",
            "add librarian-controlled intake for new evidence",
            "move only separately authorized raw/bulky payloads out of report-facing evidence",
            "normalize archive packaging only by creating new superseding packages",
        ],
        "issue_ids": sorted(issue["issue_id"] for issue in issues),
    }


def authority_drift_queue(issues: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    categories = {
        "docs_evidences_divergent_same_basename",
        "missing_raw_locator",
        "zip_path_traversal_or_non_normal",
    }
    rows = [
        issue
        for issue in issues
        if issue.get("category") in categories
        or "authority" in str(issue.get("category", ""))
    ]
    return {
        "schema_version": "eiel_authority_drift_curation_queue_v1",
        "curation_executed": False,
        "rows": sorted(rows, key=lambda item: item["issue_id"]),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def check_no_regression(
    baseline_issue_path: Path, current_issue_path: Path, *, strict: bool = True
) -> dict[str, Any]:
    baseline = load_json(baseline_issue_path)
    current = load_json(current_issue_path)
    base_counts = issue_counts(baseline)
    current_counts = issue_counts(current)
    regressions = []
    for category, count in current_counts["by_category"].items():
        if count > base_counts["by_category"].get(category, 0):
            regressions.append(
                {
                    "category": category,
                    "baseline_count": base_counts["by_category"].get(category, 0),
                    "current_count": count,
                }
            )
    return {
        "schema_version": "eiel_librarian_no_regression_check_v1",
        "strict": strict,
        "baseline_counts": base_counts,
        "current_counts": current_counts,
        "regressions": regressions,
        "status": "failed" if strict and regressions else "passed",
    }


def deterministic_zip(output_zip: Path, files: list[Path], *, root: Path) -> str:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    with zipfile.ZipFile(
        output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
            rel = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(rel, NORMALIZED_ZIP_TIMESTAMP)
            info.external_attr = 0o100644 << 16
            data = path.read_bytes()
            archive.writestr(info, data)
            manifest_rows.append(
                {"path": rel, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            )
        manifest_data = canonical_json_bytes(
            {
                "schema_version": "eiel_deterministic_zip_member_manifest_v1",
                "members": manifest_rows,
            }
        )
        info = zipfile.ZipInfo("MANIFEST.member-sha256.json", NORMALIZED_ZIP_TIMESTAMP)
        info.external_attr = 0o100644 << 16
        archive.writestr(info, manifest_data)
    return file_sha256(output_zip)


def package_audit_supplement(audit_root: Path, output_zip: Path) -> dict[str, Any]:
    files = [path for path in audit_root.rglob("*") if path.is_file()]
    digest = deterministic_zip(output_zip, files, root=audit_root)
    sidecar = output_zip.with_suffix(output_zip.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {output_zip.name}\n", encoding="utf-8")
    return {
        "zip_path": str(output_zip),
        "zip_sha256": digest,
        "sidecar_path": str(sidecar),
        "zip_safety": inspect_zip_safety(output_zip),
    }


def copy_prompt_to_trash(path: Path) -> bool:
    trash = Path.home() / ".local/share/Trash/files"
    trash.mkdir(parents=True, exist_ok=True)
    target = trash / path.name
    if target.exists():
        target = trash / f"{path.stem}-{int(time.time())}{path.suffix}"
    shutil.move(str(path), str(target))
    return target.exists()
