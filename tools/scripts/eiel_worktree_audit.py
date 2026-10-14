#!/usr/bin/env python3
"""Read-only, fail-closed Git worktree and local-branch inventory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "eiel_worktree_audit_v1"
DISPOSITIONS = {
    "preserve_active",
    "preserve_dirty",
    "preserve_uncontained",
    "preserve_ambiguous",
    "candidate_only_after_explicit_authority",
}
_CACHE_COMPONENTS = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}


class WorktreeAuditError(RuntimeError):
    """The inventory cannot be produced without guessing."""


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise WorktreeAuditError(f"git {' '.join(args)} failed: {detail}")
    return completed


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_cache_path(value: str) -> bool:
    return any(component in _CACHE_COMPONENTS for component in Path(value).parts)


def _status_summary(worktree: Path) -> dict[str, Any]:
    completed = _git(
        worktree,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignored=matching",
    )
    records = completed.stdout.split("\0")
    tracked: list[str] = []
    untracked: list[str] = []
    ignored: list[str] = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        if len(record) < 3:
            raise WorktreeAuditError(f"unsupported git status record in {worktree}: {record!r}")
        status = record[:2]
        path = record[3:]
        if status == "??":
            untracked.append(path)
        elif status == "!!":
            ignored.append(path)
        else:
            tracked.append(path)
            if "R" in status or "C" in status:
                index += 1
    non_cache_ignored = [path for path in ignored if not _is_cache_path(path)]
    return {
        "tracked_count": len(tracked),
        "untracked_count": len(untracked),
        "ignored_count": len(ignored),
        "non_cache_ignored_count": len(non_cache_ignored),
        "tracked_paths": sorted(tracked),
        "untracked_paths": sorted(untracked),
        "non_cache_ignored_paths": sorted(non_cache_ignored),
        "dirty": bool(tracked or untracked or non_cache_ignored),
    }


def _parse_worktrees(repo: Path) -> list[dict[str, Any]]:
    text = _git(repo, "worktree", "list", "--porcelain").stdout
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for line in [*text.splitlines(), ""]:
        if not line:
            if current:
                if "path" not in current or "head" not in current:
                    raise WorktreeAuditError("malformed git worktree porcelain output")
                rows.append(current)
                current = {}
            continue
        key, _, value = line.partition(" ")
        if key == "worktree":
            current["path"] = value
        elif key == "HEAD":
            current["head"] = value
        elif key == "branch":
            current["branch"] = value.removeprefix("refs/heads/")
        elif key == "detached":
            current["detached"] = True
        elif key == "bare":
            current["bare"] = True
        elif key == "locked":
            current["locked_reason"] = value
        elif key == "prunable":
            current["prunable_reason"] = value
    return rows


def _branch_rows(repo: Path) -> list[dict[str, str]]:
    marker = "\x1f"
    output = _git(
        repo,
        "for-each-ref",
        f"--format=%(refname:short){marker}%(objectname){marker}%(upstream:short)",
        "refs/heads",
    ).stdout
    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        fields = line.split(marker)
        if len(fields) != 3:
            raise WorktreeAuditError("malformed git for-each-ref output")
        rows.append({"branch": fields[0], "head": fields[1], "upstream": fields[2]})
    return sorted(rows, key=lambda row: row["branch"])


def _commit_relation(repo: Path, *, head: str, frontier: str) -> dict[str, Any]:
    contained = _git(repo, "merge-base", "--is-ancestor", head, frontier, check=False)
    if contained.returncode not in {0, 1}:
        raise WorktreeAuditError(contained.stderr.strip() or "merge-base failed")
    unique = int(_git(repo, "rev-list", "--count", f"{frontier}..{head}").stdout.strip())
    behind = int(_git(repo, "rev-list", "--count", f"{head}..{frontier}").stdout.strip())
    return {
        "contained_in_frontier": contained.returncode == 0,
        "unique_commits_vs_frontier": unique,
        "frontier_commits_not_in_head": behind,
    }


def _ownership(branch: str | None) -> dict[str, str]:
    if branch == "main":
        return {"likely_track": "main", "confidence": "high", "basis": "canonical_main_ref"}
    if branch and "/" in branch:
        prefix, topic = branch.split("/", 1)
        if prefix in {"integration", "work", "staging", "docs", "documentation"}:
            return {
                "likely_track": topic,
                "confidence": "low",
                "basis": "branch_name_only_not_authority",
            }
    return {"likely_track": "unknown", "confidence": "none", "basis": "no_authority_evidence"}


def collect_process_evidence(
    worktree_paths: Sequence[Path],
    *,
    proc_root: Path = Path("/proc"),
    max_fd_links: int = 200_000,
) -> dict[str, Any]:
    roots = [path.resolve() for path in worktree_paths]
    matches: dict[str, list[dict[str, Any]]] = {str(path): [] for path in roots}
    permission_errors = 0
    fd_links = 0
    if not proc_root.is_dir():
        return {
            "state": "unavailable",
            "permission_errors": 0,
            "fd_links_scanned": 0,
            "matches_by_worktree": matches,
        }
    truncated = False
    for pid_path in sorted(proc_root.iterdir(), key=lambda path: path.name):
        if not pid_path.name.isdigit():
            continue
        pid = int(pid_path.name)
        try:
            command = (pid_path / "comm").read_text(encoding="utf-8", errors="replace").strip()
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            permission_errors += 1
            command = ""
        links: list[tuple[str, Path]] = []
        try:
            links.append(("cwd", (pid_path / "cwd").resolve(strict=True)))
        except (FileNotFoundError, ProcessLookupError):
            pass
        except PermissionError:
            permission_errors += 1
        fd_root = pid_path / "fd"
        try:
            fd_entries = list(fd_root.iterdir())
        except (FileNotFoundError, ProcessLookupError):
            fd_entries = []
        except PermissionError:
            permission_errors += 1
            fd_entries = []
        for fd_path in fd_entries:
            if fd_links >= max_fd_links:
                truncated = True
                break
            fd_links += 1
            try:
                links.append(("open_file", fd_path.resolve(strict=True)))
            except PermissionError:
                permission_errors += 1
            except (FileNotFoundError, ProcessLookupError, OSError):
                continue
        for kind, target in links:
            for root in roots:
                if _inside(target, root):
                    row = {"pid": pid, "command": command, "kind": kind, "target": str(target)}
                    if row not in matches[str(root)]:
                        matches[str(root)].append(row)
        if truncated:
            break
    for rows in matches.values():
        rows.sort(key=lambda row: (row["pid"], row["kind"], row["target"]))
    state = "complete"
    if permission_errors or truncated:
        state = "partial"
    return {
        "state": state,
        "permission_errors": permission_errors,
        "truncated": truncated,
        "fd_links_scanned": fd_links,
        "matches_by_worktree": matches,
    }


def _disposition(
    *,
    is_frontier: bool,
    detached: bool,
    status: Mapping[str, Any],
    relation: Mapping[str, Any],
    process_matches: Sequence[Mapping[str, Any]],
    process_state: str,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if is_frontier:
        return "preserve_active", ["worktree_or_branch_is_the_named_frontier"]
    if process_matches:
        return "preserve_active", ["process_cwd_or_open_file_references_worktree"]
    if bool(status.get("dirty")):
        return "preserve_dirty", ["tracked_untracked_or_non_cache_ignored_content_present"]
    if not bool(relation.get("contained_in_frontier")) or int(
        relation.get("unique_commits_vs_frontier", 0)
    ):
        return "preserve_uncontained", ["head_has_commits_not_contained_in_frontier"]
    if detached:
        reasons.append("detached_worktree_requires_explicit_ownership_evidence")
    if process_state != "complete":
        reasons.append("process_or_open_file_collection_not_complete")
    if reasons:
        return "preserve_ambiguous", reasons
    return "candidate_only_after_explicit_authority", [
        "mechanical_checks_passed_but_ownership_and_disposition_authority_are_external"
    ]


def build_inventory(
    repo: Path,
    *,
    frontier: str,
    process_snapshot: Mapping[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    repo = Path(_git(repo, "rev-parse", "--show-toplevel").stdout.strip()).resolve()
    common_raw = _git(repo, "rev-parse", "--git-common-dir").stdout.strip()
    common_dir = Path(common_raw)
    if not common_dir.is_absolute():
        common_dir = (repo / common_dir).resolve()
    frontier_head = _git(repo, "rev-parse", "--verify", f"{frontier}^{{commit}}").stdout.strip()
    source_head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    raw_worktrees = _parse_worktrees(repo)
    worktree_paths = [Path(row["path"]).resolve() for row in raw_worktrees if not row.get("bare")]
    processes = dict(process_snapshot or collect_process_evidence(worktree_paths))
    process_state = str(processes.get("state") or "unavailable")
    matches = processes.get("matches_by_worktree")
    if not isinstance(matches, Mapping):
        matches = {}

    worktrees: list[dict[str, Any]] = []
    checked_out: dict[str, str] = {}
    for raw in raw_worktrees:
        path = Path(raw["path"]).resolve()
        branch = raw.get("branch")
        detached = bool(raw.get("detached"))
        if branch:
            checked_out[str(branch)] = str(path)
        status = (
            {"dirty": True, "error": "bare_worktree"}
            if raw.get("bare")
            else _status_summary(path)
        )
        relation = _commit_relation(repo, head=str(raw["head"]), frontier=frontier_head)
        process_rows = list(matches.get(str(path), []))
        disposition, reasons = _disposition(
            is_frontier=str(raw["head"]) == frontier_head and branch == frontier,
            detached=detached,
            status=status,
            relation=relation,
            process_matches=process_rows,
            process_state=process_state,
        )
        worktrees.append(
            {
                "path": str(path),
                "branch": branch,
                "detached": detached,
                "head": str(raw["head"]),
                "locked_reason": raw.get("locked_reason"),
                "prunable_reason": raw.get("prunable_reason"),
                "status": status,
                "relation_to_frontier": relation,
                "ownership": _ownership(str(branch) if branch else None),
                "process_evidence": process_rows,
                "disposition": disposition,
                "reasons": reasons,
            }
        )
    worktrees.sort(key=lambda row: row["path"])

    branches: list[dict[str, Any]] = []
    worktree_dispositions = {
        row.get("branch"): (row["disposition"], row["reasons"])
        for row in worktrees
        if row.get("branch")
    }
    for raw in _branch_rows(repo):
        relation = _commit_relation(repo, head=raw["head"], frontier=frontier_head)
        if raw["branch"] in worktree_dispositions:
            disposition, reasons = worktree_dispositions[raw["branch"]]
        else:
            disposition, reasons = _disposition(
                is_frontier=raw["branch"] == frontier,
                detached=False,
                status={"dirty": False},
                relation=relation,
                process_matches=[],
                process_state="complete",
            )
        branches.append(
            {
                **raw,
                "checked_out_path": checked_out.get(raw["branch"]),
                "relation_to_frontier": relation,
                "ownership": _ownership(raw["branch"]),
                "disposition": disposition,
                "reasons": reasons,
            }
        )

    core = {
        "schema_version": SCHEMA_VERSION,
        "repo_root": str(repo),
        "git_common_dir": str(common_dir),
        "source_head": source_head,
        "frontier": {"ref": frontier, "head": frontier_head},
        "process_collection": {
            key: value for key, value in processes.items() if key != "matches_by_worktree"
        },
        "worktrees": worktrees,
        "branches": branches,
    }
    inventory_digest = hashlib.sha256(_canonical_bytes(core)).hexdigest()
    result = {
        **core,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "inventory_sha256": inventory_digest,
    }
    observed = {
        row["disposition"] for group in (worktrees, branches) for row in group
    }
    if not observed.issubset(DISPOSITIONS):
        raise WorktreeAuditError("inventory produced an unsupported disposition")
    return result


def render_markdown(inventory: Mapping[str, Any]) -> str:
    lines = [
        "# EIEL Worktree Hygiene Audit",
        "",
        f"- Repository: `{inventory['repo_root']}`",
        f"- Source HEAD: `{inventory['source_head']}`",
        f"- Frontier: `{inventory['frontier']['ref']}` at `{inventory['frontier']['head']}`",
        f"- Inventory SHA-256: `{inventory['inventory_sha256']}`",
        f"- Process collection: `{inventory['process_collection'].get('state')}`",
        "",
        "This report is read-only. Candidate status is not deletion authority.",
        "",
        "## Worktrees",
        "",
        "| Path | Branch/commit | Dirty | Unique commits | Process refs | Disposition | Reasons |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in inventory["worktrees"]:
        identity = row.get("branch") or f"detached:{str(row['head'])[:12]}"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['path']}`",
                    f"`{identity}`",
                    "yes" if row["status"].get("dirty") else "no",
                    str(row["relation_to_frontier"]["unique_commits_vs_frontier"]),
                    str(len(row["process_evidence"])),
                    f"`{row['disposition']}`",
                    "; ".join(row["reasons"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Local Branches",
            "",
            "| Branch | HEAD | Checked out | Unique commits | Contained | Disposition |",
            "| --- | --- | --- | ---: | --- | --- |",
        ]
    )
    for row in inventory["branches"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['branch']}`",
                    f"`{str(row['head'])[:12]}`",
                    f"`{row['checked_out_path']}`" if row.get("checked_out_path") else "no",
                    str(row["relation_to_frontier"]["unique_commits_vs_frontier"]),
                    "yes" if row["relation_to_frontier"]["contained_in_frontier"] else "no",
                    f"`{row['disposition']}`",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _write_explicit(path: Path, text: str) -> None:
    path = path.expanduser().resolve()
    if not path.parent.is_dir():
        raise WorktreeAuditError(f"output parent does not exist: {path.parent}")
    if path.is_symlink() or path.is_dir():
        raise WorktreeAuditError(f"unsafe output path: {path}")
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect", help="inventory without mutating Git")
    inspect_parser.add_argument("--repo", type=Path, default=Path.cwd())
    inspect_parser.add_argument("--frontier", required=True)
    inspect_parser.add_argument("--json-output", type=Path)
    inspect_parser.add_argument("--markdown-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        inventory = build_inventory(args.repo, frontier=args.frontier)
        json_text = json.dumps(inventory, indent=2, sort_keys=True) + "\n"
        if args.json_output is not None:
            _write_explicit(args.json_output, json_text)
        if args.markdown_output is not None:
            _write_explicit(args.markdown_output, render_markdown(inventory))
        if args.json_output is None:
            sys.stdout.write(json_text)
        elif args.markdown_output is None:
            sys.stdout.write(json.dumps({"ok": True, "inventory_sha256": inventory["inventory_sha256"]}) + "\n")
    except (OSError, ValueError, WorktreeAuditError) as exc:
        sys.stderr.write(f"eiel_worktree_audit: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
