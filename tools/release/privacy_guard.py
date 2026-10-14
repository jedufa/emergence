"""Fail closed on unreviewed public-source content; emits locations, never values."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

PATTERNS = {
    "account_path": re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+(?:/|\b)"),
    "operator_mount": re.compile(r"/(?:data[0-9]*|volume[0-9]+)/(?:causyl|_permanent|emergence)[^\s]*"),
    "personal_email": re.compile(r"[A-Za-z0-9_.+\-]+@(?:gmail|hotmail|outlook|yahoo|protonmail)\.com", re.I),
    "private_ipv4": re.compile(r"(?<![\d.])(?:192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)(?![\d.])"),
    "conversation_url": re.compile(r"https://chatgpt\.com/(?:g|c)/[^\s\"'<>]+"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    "provider_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|AKIA[A-Z0-9]{16}|sk-(?:proj-)?[A-Za-z0-9_-]{30,})\b"),
    "private_host_url": re.compile(r"(?:https?|ssh)://[^/\s\"']+\.local(?:[:/]|\b)"),
}
FORBIDDEN_SUFFIXES = {".zip", ".tar", ".gz", ".xz", ".pdf", ".docx", ".xlsx", ".png", ".jpg", ".jpeg", ".so", ".dll", ".dylib", ".pyc", ".o", ".sqlite", ".db", ".env"}
ALLOWED_IDENTITY = re.compile(r"(?:\d+\+)?[A-Za-z0-9-]+@users\.noreply\.github\.com\Z")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_findings(name: str, data: bytes) -> list[dict]:
    digest = sha(data)
    findings = []
    if Path(name).suffix.lower() in FORBIDDEN_SUFFIXES or b"\0" in data or len(data) > 2 * 1024**2:
        findings.append({"path": name, "line": 0, "rule": "unreviewed_binary_or_large_file", "sha256": digest})
    try:
        text = data.decode("utf-8")
    except UnicodeError:
        return findings + [{"path": name, "line": 0, "rule": "non_utf8", "sha256": digest}]
    for line, value in enumerate(text.splitlines(), 1):
        for rule, pattern in PATTERNS.items():
            if pattern.search(value):
                findings.append({"path": name, "line": line, "rule": rule, "sha256": digest})
    return findings


def inspect(root: Path, *, check_history: bool = True) -> dict:
    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)
    exceptions = json.loads((root / "tools/release/privacy-exceptions.json").read_text())
    allowed = {(x["path"], x["line"], x["rule"], x["sha256"]) for x in exceptions["findings"] if x.get("reason")}
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())
    expected = {x["path"]: x["sha256"] for x in manifest["files"]}
    tracked = [p for p in git("ls-files", "-z").decode().split("\0") if p]
    failures = []
    reviewed = 0
    seen = set()
    for name in tracked:
        path = root / name
        if path.is_symlink() or not path.is_file():
            failures.append({"path": name, "rule": "non_regular_file"}); continue
        data = path.read_bytes()
        if name != "SOURCE_MANIFEST.json" and expected.get(name) != sha(data):
            failures.append({"path": name, "rule": "manifest_mismatch"})
        for f in content_findings(name, data):
            key = (f["path"], f["line"], f["rule"], f["sha256"])
            if key in allowed:
                reviewed += 1; seen.add(key)
            else:
                failures.append(f)
    if set(expected) != set(tracked) - {"SOURCE_MANIFEST.json"}:
        failures.append({"rule": "manifest_membership_mismatch"})
    refs = git("for-each-ref", "--format=%(refname)").decode().splitlines()
    if any(not r.startswith(("refs/heads/", "refs/remotes/origin/", "refs/tags/")) for r in refs):
        failures.append({"rule": "unexpected_ref_namespace"})
    commits = []
    if check_history:
        commits = git("rev-list", "--all").decode().splitlines()
        for commit in commits:
            emails = git("show", "-s", "--format=%ae%n%ce", commit).decode().splitlines()
            if not all(ALLOWED_IDENTITY.fullmatch(e) for e in emails):
                failures.append({"rule": "commit_identity", "commit": commit})
            # Inspect every historical blob, including files removed from HEAD.
            for entry in git("ls-tree", "-r", "-z", commit).split(b"\0"):
                if not entry: continue
                meta, raw_name = entry.split(b"\t", 1)
                mode, kind, oid = meta.split()
                name = raw_name.decode()
                if mode not in {b"100644", b"100755"} or kind != b"blob":
                    failures.append({"path": name, "rule": "symlink_or_submodule"}); continue
                for f in content_findings(name, git("cat-file", "blob", oid.decode())):
                    key = (f["path"], f["line"], f["rule"], f["sha256"])
                    if key not in allowed: failures.append({**f, "commit": commit})
                    else: seen.add(key)
    if allowed - seen:
        failures.append({"rule": "stale_privacy_exceptions"})
    return {"status": "PASS" if not failures else "FAIL", "files": len(tracked), "commits": len(commits), "reviewed_exact_exceptions": reviewed, "findings": failures}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--working-tree-only", action="store_true")
    args = parser.parse_args()
    report = inspect(args.root.resolve(), check_history=not args.working_tree_only)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
