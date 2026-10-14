#!/usr/bin/env python3
"""
tools.verify_syntax

Repo sanity checker (syntax + line endings), with optional interactive fixes.

Adds:
- Colored terminal output (TTY-aware, CI-safe)
- --no-color option
"""

from __future__ import annotations

import argparse
import compileall
import os
import sys
import tempfile
import zipfile
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple


# ============================================================
# Terminal Colors (TTY safe)
# ============================================================


def _find_repo_root(start: Path) -> Path:
    """Find repo root by walking upward from `start` until a repo marker is found.

    This is robust against moving scripts into subpackages (e.g. tools/_scripts).
    Repo markers: .git, pyproject.toml, setup.cfg, requirements.txt.
    Falls back to current working directory if no marker is found.
    """
    start = start.resolve()
    if start.is_file():
        cur = start.parent
    else:
        cur = start
    markers = ('.git', 'pyproject.toml', 'setup.cfg', 'requirements.txt')
    for p in (cur, *cur.parents):
        for m in markers:
            if (p / m).exists():
                return p
    # Last resort: allow running from repo root via cwd
    cwd = Path.cwd().resolve()
    for p in (cwd, *cwd.parents):
        for m in markers:
            if (p / m).exists():
                return p
    return cwd

class TermColor:
    ENABLE = sys.stdout.isatty()

    RED = "\033[31m" if ENABLE else ""
    GREEN = "\033[32m" if ENABLE else ""
    YELLOW = "\033[33m" if ENABLE else ""
    CYAN = "\033[36m" if ENABLE else ""
    RESET = "\033[0m" if ENABLE else ""


def red(s: str) -> str:
    return f"{TermColor.RED}{s}{TermColor.RESET}"


def green(s: str) -> str:
    return f"{TermColor.GREEN}{s}{TermColor.RESET}"


def yellow(s: str) -> str:
    return f"{TermColor.YELLOW}{s}{TermColor.RESET}"


def cyan(s: str) -> str:
    return f"{TermColor.CYAN}{s}{TermColor.RESET}"


# ============================================================
# Helpers
# ============================================================

def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _which(cmd: str) -> Optional[str]:
    from shutil import which
    return which(cmd)


def _prompt_yes_no(question: str, default_no: bool = True) -> bool:
    if not _is_interactive():
        return False

    suffix = " [y/N]: " if default_no else " [Y/n]: "
    while True:
        try:
            ans = input(yellow(question + suffix)).strip().lower()
        except EOFError:
            return False

        if ans == "":
            return not default_no
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False

        print("Please answer y or n.")


def _detect_crlf_bytes(data: bytes) -> bool:
    return (b"\r\n" in data) or (b"\r" in data)


def _normalize_to_lf_bytes(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _check_and_maybe_fix_line_endings(path: Path, *, auto_fix: bool) -> Tuple[bool, bool]:
    try:
        raw = path.read_bytes()
    except Exception as e:
        sys.stderr.write(red(f"[verify_syntax] ERROR reading {path}: {e}\n"))
        return (False, False)

    if not _detect_crlf_bytes(raw):
        return (True, False)

    sys.stderr.write(yellow(f"[verify_syntax] CRLF/CR detected: {path}\n"))

    if auto_fix:
        new_raw = _normalize_to_lf_bytes(raw)
        if new_raw != raw:
            path.write_bytes(new_raw)
            sys.stderr.write(green(f"[verify_syntax] Fixed line endings (LF) for: {path}\n"))
            return (True, True)
        return (True, False)

    if _prompt_yes_no(f"Fix line endings (convert to LF) for {path}?"):
        new_raw = _normalize_to_lf_bytes(raw)
        if new_raw != raw:
            path.write_bytes(new_raw)
            sys.stderr.write(green(f"[verify_syntax] Fixed line endings (LF) for: {path}\n"))
            return (True, True)
        return (True, False)

    sys.stderr.write(red(f"[verify_syntax] Not fixed: {path}\n"))
    return (False, False)


# ============================================================
# Python checks
# ============================================================

def _compile_path(path: Path) -> bool:
    ok = True
    if path.is_dir():
        ok = compileall.compile_dir(str(path), quiet=1)
    elif path.is_file() and path.suffix == ".py":
        ok = compileall.compile_file(str(path), quiet=1)

    if not ok:
        sys.stderr.write(red(f"[verify_syntax] Python syntax FAILED: {path}\n"))
    return bool(ok)


# ============================================================
# Shell checks
# ============================================================

def _check_shell_file(path: Path) -> bool:
    ok = True

    result = subprocess.run(
        ["bash", "-n", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(red(f"[verify_syntax] bash -n FAILED: {path}\n"))
        sys.stderr.write(result.stderr)
        ok = False

    if _which("shellcheck"):
        sc = subprocess.run(
            ["shellcheck", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if sc.returncode != 0:
            sys.stderr.write(yellow(f"[verify_syntax] shellcheck warnings/errors: {path}\n"))
            sys.stderr.write(sc.stdout)
            ok = False

    return ok


def _gather_targets(t: Path) -> List[Path]:
    if t.is_file():
        return [t]
    if t.is_dir():
        out: List[Path] = []
        out.extend(t.rglob("*.py"))
        out.extend(t.rglob("*.sh"))
        seen = set()
        uniq: List[Path] = []
        for p in out:
            rp = str(p.resolve())
            if rp in seen:
                continue
            seen.add(rp)
            uniq.append(p)
        return uniq
    return []


def _check_line_endings_in_targets(target: Path, *, auto_fix: bool) -> bool:
    ok = True
    for p in _gather_targets(target):
        if p.suffix not in (".py", ".sh"):
            continue
        o, _changed = _check_and_maybe_fix_line_endings(p, auto_fix=auto_fix)
        if not o:
            ok = False
    return ok


def _check_shell_in_dir(path: Path) -> bool:
    ok = True
    for p in path.rglob("*.sh"):
        if not _check_shell_file(p):
            ok = False
    return ok


# ============================================================
# ZIP handling
# ============================================================

def _check_zip(zip_path: Path) -> bool:
    with tempfile.TemporaryDirectory(prefix="verify_syntax_") as td:
        root = Path(td)
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(root))

        ok_le = _check_line_endings_in_targets(root, auto_fix=False)
        ok_py = compileall.compile_dir(str(root), quiet=1)
        ok_sh = _check_shell_in_dir(root)

        return bool(ok_le) and bool(ok_py) and bool(ok_sh)


# ============================================================
# CLI
# ============================================================

def _default_targets(repo_root: Path) -> List[Path]:
    return [
        repo_root / "tools",
        repo_root / "scripts",
        repo_root / "engine",
        repo_root / "experiments",
        repo_root / "emergence",
    ]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--file", action="append", default=[], help="File/dir/zip to check (repeatable).")
    p.add_argument("--fix", action="store_true", help="Auto-fix CRLF/CR line endings.")
    p.add_argument("--no-color", action="store_true", help="Disable colored output.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    if args.no_color:
        TermColor.ENABLE = False
        TermColor.RED = TermColor.GREEN = TermColor.YELLOW = TermColor.CYAN = TermColor.RESET = ""

    overall_ok = True

    if args.file:
        targets = [Path(x).expanduser() for x in args.file]
    else:
        repo_root = _find_repo_root(Path(__file__).resolve())
        targets = _default_targets(repo_root)

    for t in targets:
        if not t.exists():
            continue

        if t.suffix == ".zip":
            ok = _check_zip(t)
        else:
            ok_le = _check_line_endings_in_targets(t, auto_fix=bool(args.fix))
            ok_py = _compile_path(t)
            ok_sh = _check_shell_in_dir(t) if t.is_dir() else (_check_shell_file(t) if t.suffix == ".sh" else True)
            ok = bool(ok_le) and bool(ok_py) and bool(ok_sh)

        print(f"[verify_syntax] {cyan(str(t))} {green('OK') if ok else red('FAILED')}")

        if not ok:
            overall_ok = False

    if not overall_ok:
        if _is_interactive() and not args.fix:
            sys.stderr.write(yellow("[verify_syntax] Tip: re-run with --fix to auto-normalize CRLF/CR to LF.\n"))
        return 2

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        if os.environ.get("EMERGENCE_TRACEBACK") == "1":
            raise
        sys.stderr.write(red(f"[verify_syntax] ERROR: {e}\n"))
        raise SystemExit(2)
