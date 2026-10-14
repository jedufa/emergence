#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path


DEFAULT_SHARED_ENV = Path.home() / ".venvs" / "emergence-py314-cuda"
MINIMAL_PACKAGES = ("-e", ".", "Markdown", "PyYAML", "requests")


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _repo_root(path: Path) -> Path:
    out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=path, text=True)
    return Path(out.strip()).resolve()


def _ensure_env(shared_env: Path) -> None:
    python = shared_env / "bin" / "python"
    if python.exists():
        return
    shared_env.parent.mkdir(parents=True, exist_ok=True)
    venv.EnvBuilder(with_pip=True).create(shared_env)
    _run([str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])


def _link(repo: Path, shared_env: Path) -> str:
    target = repo / ".venv"
    desired = str(shared_env)
    if target.is_symlink():
        current = os.readlink(target)
        if Path(current).expanduser().resolve() == shared_env.resolve():
            return f"ok: {repo}"
        target.unlink()
    elif target.exists():
        raise SystemExit(f"refusing to replace non-symlink venv: {target}")
    target.symlink_to(shared_env, target_is_directory=True)
    return f"linked: {repo}"


def _worktree_roots(repo: Path) -> list[Path]:
    out = subprocess.check_output(["git", "worktree", "list", "--porcelain"], cwd=repo, text=True)
    roots: list[Path] = []
    for line in out.splitlines():
        if line.startswith("worktree "):
            roots.append(Path(line.removeprefix("worktree ")).resolve())
    return roots


def main() -> int:
    parser = argparse.ArgumentParser(description="Attach Emergence checkouts to one shared virtual environment.")
    parser.add_argument("--shared-env", type=Path, default=DEFAULT_SHARED_ENV)
    parser.add_argument("--all-worktrees", action="store_true", help="link every registered worktree for this repo")
    parser.add_argument(
        "--install",
        choices=("none", "minimal", "full"),
        default="none",
        help="optional package install profile; full uses requirements.txt and may install large CUDA/Torch wheels",
    )
    args = parser.parse_args()

    repo = _repo_root(Path.cwd())
    shared_env = args.shared_env.expanduser().resolve()
    _ensure_env(shared_env)

    roots = _worktree_roots(repo) if args.all_worktrees else [repo]
    for root in roots:
        print(_link(root, shared_env))

    python = shared_env / "bin" / "python"
    if args.install == "minimal":
        _run([str(python), "-m", "pip", "install", *MINIMAL_PACKAGES], cwd=repo)
    elif args.install == "full":
        _run([str(python), "-m", "pip", "install", "-r", "requirements.txt"], cwd=repo)

    print(f"shared_env={shared_env}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
