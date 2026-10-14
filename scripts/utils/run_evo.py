"""
scripts.utils.run_evo

Test/automation runner helpers for Emergence.

Why this file exists
- Your nox sessions are meant to be a thin launcher: `python -m experiments.runs.smoke` etc.
- Those run scripts need a *stable* API to:
    * find repo root (so relative paths behave)
    * run `python -m engine.main ...`
    * optionally archive the latest run folder into a .zip
    * optionally mirror artifacts to Windows (WSL -> /mnt/d/...).

This module is that stable API.

Ctrl-C behavior
- First Ctrl-C: forward SIGINT to the child process group, wait for clean shutdown.
- Second Ctrl-C: force kill (SIGKILL) the child process group.

Compatibility
- Linux/WSL
- Python 3.10+
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import shutil
import signal
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RunResult:
    """
    Result of launching the simulation process.

    Attributes:
        returncode: Process return code (0 on success).
        cmd: The exact command that was executed.
        interrupted: True if Ctrl-C was caught and forwarded to the simulation.
    """

    returncode: int
    cmd: List[str]
    interrupted: bool = False


@dataclass(frozen=True)
class PostProcessOptions:
    """
    Optional post-processing for a run launched via run_main().

    This replaces the legacy bash wrapper behaviors
    while keeping activation explicit and test-friendly.

    Attributes:
        capture_console: If True, write the child process console output into
            each produced run directory as "console.txt".
        archive: If True, zip each produced run directory into "<run_id>.zip"
            alongside the directory.
        delete_run_dir: If True, delete each run directory after successful
            archiving. Only applies when archive=True.
        push_windows: If True, mirror produced top-level .zip artifacts to the
            directory pointed to by WIN_RUNS_DIR.
        on_failure: If True, perform archive/push even when the simulation
            return code is non-zero. Useful for debugging failures.
    """
    capture_console: bool = True
    archive: bool = False
    delete_run_dir: bool = True
    push_windows: bool = False
    on_failure: bool = False



# -----------------------------
# Repo / path helpers
# -----------------------------

def _repo_root() -> Path:
    """
    Resolve repository root (best-effort).

    Returns:
        Path: The directory that contains this file's parent 'scripts' folder.
    """
    here = Path(__file__).resolve()
    # scripts/utils/run_evo.py -> scripts -> repo root
    return here.parents[2]


def get_repo_root() -> Path:
    """
    Public repo-root helper (stable API used by run scripts).

    Returns:
        Path: Repository root directory.
    """
    return _repo_root()


def _default_main_module() -> str:
    """
    Default python module entrypoint for the simulation.

    Returns:
        str: Module name usable with `python -m <module>`.
    """
    # Canonical module during migration; delegates to engine.main.
    return "engine.main"


def _print(msg: str) -> None:
    """
    Print helper that flushes immediately (important for nox/CI output).

    Args:
        msg: Message to print.
    """
    print(msg, flush=True)


# -----------------------------
# Process runner
# -----------------------------

def run_evo(args: Sequence[str], *, python: Optional[str] = None, module: Optional[str] = None, post: Optional[PostProcessOptions] = None) -> RunResult:
    """
    Convenience wrapper used by many test runners.

    Args:
        args: List of CLI args to pass to the simulation (e.g. ["--gen","10","--seed","42"]).
        python: Optional path to python executable. Defaults to current interpreter.
        module: Optional module name. Defaults to "engine.main".

    Returns:
        RunResult: Return code + command.
    """
    return run_main(list(args), python=python, module=module, post=post)


def _extract_arg_value(args: Sequence[str], key: str) -> Optional[str]:
    """
    Extract the value for a CLI flag from an argument list.

    Supports both:
      --key value
      --key=value

    Args:
        args: Argument sequence.
        key: Flag name including leading dashes, e.g. "--runs_dir".

    Returns:
        Optional[str]: The value if present, else None.
    """
    if key in args:
        i = list(args).index(key)
        if i + 1 < len(args):
            return str(args[i + 1])
        return None

    prefix = key + "="
    for a in args:
        if str(a).startswith(prefix):
            return str(a)[len(prefix):]
    return None


def _extract_runs_dir(args: Sequence[str]) -> Optional[Path]:
    """
    Best-effort extraction of the runs directory from CLI args.

    Args:
        args: Argument sequence passed to the simulation.

    Returns:
        Optional[Path]: Parsed runs_dir path if present.
    """
    v = _extract_arg_value(args, "--runs_dir")
    if not v:
        return None
    try:
        return Path(v).expanduser()
    except Exception:
        return None


def _tee_child_output_to_console_and_file(proc: subprocess.Popen, log_file) -> None:
    """
    Stream the child's combined stdout/stderr to both the parent console and a file.

    This keeps nox output live (so you see progress) while also preserving the
    exact console text for later zipping into the run directory.

    Args:
        proc: The running subprocess with stdout=PIPE.
        log_file: Open file handle to write console output into.
    """
    assert proc.stdout is not None
    for line in iter(proc.stdout.readline, ""):
        # Write to file first, then to console.
        try:
            log_file.write(line)
            log_file.flush()
        except Exception:
            # Don't let logging failures kill the run.
            pass
        try:
            sys.stdout.write(line)
            sys.stdout.flush()
        except Exception:
            pass


def run_main(args: List[str], *, python: Optional[str] = None, module: Optional[str] = None, post: Optional[PostProcessOptions] = None) -> RunResult:
    """
    Launch the simulation as a child process and wait for completion.

    This function is designed to play nicely with Ctrl-C:
    - It does NOT crash the parent immediately on KeyboardInterrupt.
    - It forwards SIGINT to the child, then waits for the child to exit.
    - If Ctrl-C is pressed again, it force-kills the child.

    IMPORTANT: Console capture
    - When run via nox, we often invoke `python -m engine.main ...` directly
      (not via the removed legacy wrapper).
    - To keep run artifacts self-contained, we capture combined stdout/stderr into
      a temp file, then place it into the most recent run directory as console.txt.

    Args:
        args: CLI args to pass to the simulation.
        python: Optional python executable path (defaults to sys.executable).
        module: Optional python module entrypoint (defaults to "engine.main").

    Returns:
        RunResult: Return code + command.
    """
    py = python or sys.executable
    mod = module or _default_main_module()

    # Ensure we run from repo root so relative paths behave predictably.
    root = _repo_root()
    cmd = [py, "-m", mod] + list(args)

    runs_dir = _extract_runs_dir(args)

    # Snapshot existing run directories so we can attribute artifacts to *this* invocation.
    pre_run_dirs: Optional[set[str]] = None
    if runs_dir is not None and runs_dir.exists():
        try:
            pre_run_dirs = {p.name for p in _iter_candidate_run_dirs(runs_dir)}
        except Exception:
            pre_run_dirs = None

    # Temp console log (we move/copy it into the run directory after completion).
    tmp_dir = (runs_dir if runs_dir else (root / "runs"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_console = tmp_dir / f"_tmp_console__pid{os.getpid()}__{int(time.time())}.txt"

    # Put child in its own process group/session so we can signal it reliably.
    # start_new_session=True => setsid() on POSIX.
    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Pump output in a thread so Ctrl-C handling remains responsive.
    import threading
    with open(tmp_console, "w", encoding="utf-8", errors="replace") as f:
        pump = threading.Thread(target=_tee_child_output_to_console_and_file, args=(proc, f), daemon=True)
        pump.start()

        interrupt_count = 0
        rc: Optional[int] = None

        while True:
            try:
                rc = proc.wait()
                break
            except KeyboardInterrupt:
                interrupt_count += 1

                if interrupt_count == 1:
                    _print(
                        "[interrupt] Ctrl-C caught in launcher; forwarding SIGINT to simulation and waiting for clean shutdown..."
                    )
                    try:
                        # Send SIGINT to the process group (child + its children).
                        os.killpg(proc.pid, signal.SIGINT)
                    except ProcessLookupError:
                        # Child already exited.
                        pass
                    # Loop back to wait.
                    continue

                _print("[interrupt] Second Ctrl-C: force-killing simulation (SIGKILL).")
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                # Re-raise so outer layers (nox) still mark as interrupted, but we attempted cleanup.
                raise
        # Ensure output thread drains.
        pump.join(timeout=2.0)

    # Place temp console log into produced run dir(s) if possible.
    produced_run_dirs: List[Path] = []
    try:
        if runs_dir is not None and runs_dir.exists():
            try:
                post_run_names = {p.name for p in _iter_candidate_run_dirs(runs_dir)}
            except Exception:
                post_run_names = set()

            if pre_run_dirs is not None:
                new_names = sorted(post_run_names - pre_run_dirs)
                produced_run_dirs = [runs_dir / n for n in new_names if (runs_dir / n).is_dir()]
            # Fallback: at least attach to "latest" if we cannot attribute precisely.
            if not produced_run_dirs:
                latest = _latest_run_dir(runs_dir)
                if latest is not None:
                    produced_run_dirs = [latest]
    except Exception:
        produced_run_dirs = []

    # Default post-processing options
    post_opts = post or PostProcessOptions()

    # Attach console.txt
    try:
        if post_opts.capture_console and produced_run_dirs:
            for i, rd in enumerate(produced_run_dirs):
                dst = rd / "console.txt"
                # Only overwrite if missing or empty (so manual runs can preserve a richer log).
                if (not dst.exists()) or (dst.stat().st_size == 0):
                    if i == 0:
                        shutil.move(str(tmp_console), str(dst))
                    else:
                        shutil.copy2(str(produced_run_dirs[0] / "console.txt"), str(dst))
            # If we moved into first run dir, tmp_console no longer exists.
            if tmp_console.exists():
                tmp_console.unlink(missing_ok=True)
        else:
            tmp_console.unlink(missing_ok=True)
    except Exception:
        try:
            tmp_console.unlink(missing_ok=True)
        except Exception:
            pass

    # Archive/delete/push are owned here (Plan A) when enabled.
    #
    # IMPORTANT (Ctrl-C / graceful interrupt)
    # If the run was interrupted, the simulation is expected to have written an
    # interrupt checkpoint inside the run directory so it can be resumed.
    # Therefore we MUST NOT delete (or even zip+delete) the run directory on an
    # interrupted run, even if the process exits with return code 0.
    try:
        interrupted = (interrupt_count > 0)
        # Never post-process interrupted runs. Keeping the run directory intact is
        # more important than saving disk space.
        if not interrupted:
            should_post = (rc == 0) or bool(post_opts.on_failure)
            if should_post and post_opts.archive and produced_run_dirs:
                for rd in produced_run_dirs:
                    archive_run_dir(rd, delete_dir=bool(post_opts.delete_run_dir))
            if should_post and post_opts.push_windows:
                push_runs_to_windows(verbose=False)
    except Exception:
        # Post-processing must never fail the run itself.
        pass

    return RunResult(returncode=int(rc if rc is not None else 1), cmd=cmd, interrupted=(interrupt_count > 0))



def run_many(runs: Iterable[Sequence[str]], *, python: Optional[str] = None, module: Optional[str] = None) -> int:
    """
    Run multiple simulations sequentially.

    Args:
        runs: Iterable of arg-lists. Each element is passed to run_main().
        python: Optional python executable path.
        module: Optional python module entrypoint.

    Returns:
        int: 0 if all runs succeeded, else the first non-zero return code.
    """
    for args in runs:
        result = run_main(list(args), python=python, module=module)
        # If a run was interrupted, bubble up a conventional SIGINT-like exit code.
        if getattr(result, "interrupted", False):
            return 130
        if result.returncode != 0:
            return result.returncode
    return 0


# -----------------------------
# Run artifact helpers
# -----------------------------

def _iter_candidate_run_dirs(runs_dir: Path) -> Iterable[Path]:
    """
    Yield run directory candidates under runs_dir.

    We intentionally ignore:
    - files (.zip etc.)
    - the batches/ folder
    - hidden folders

    Args:
        runs_dir: The runs directory.

    Yields:
        Path objects of candidate run directories.
    """
    if not runs_dir.exists():
        return
    for p in runs_dir.iterdir():
        if not p.is_dir():
            continue
        if p.name.startswith("."):
            continue
        if p.name == "batches":
            continue
        yield p


def _latest_run_dir(runs_dir: Path) -> Optional[Path]:
    """
    Pick the most recently modified run directory.

    Args:
        runs_dir: The runs directory.

    Returns:
        Path | None: The newest run directory, or None if none exist.
    """
    candidates = list(_iter_candidate_run_dirs(runs_dir))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def archive_latest_run(runs_dir: Path, *, delete_dir: bool = True) -> Optional[Path]:
    """
    Zip the most recently modified run directory under runs_dir.

    The zip is created next to the run directory:
        runs/<run_id>/  ->  runs/<run_id>.zip

    Args:
        runs_dir: Directory that contains per-run folders.
        delete_dir: If True, delete the run directory after successful zipping.

    Returns:
        Path | None: Path to the created zip, or None if no run dir exists.
    """
    runs_dir = Path(runs_dir)
    latest = _latest_run_dir(runs_dir)
    if latest is None:
        _print(f"[archive] No run directories found under: {runs_dir}")
        return None

    zip_path = runs_dir / f"{latest.name}.zip"

    _print(f"[archive] Zipping latest run: {latest} -> {zip_path}")

    # Create zip (deterministic enough for our use; not bit-identical across OS).
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in latest.rglob("*"):
            if file_path.is_dir():
                continue
            arcname = file_path.relative_to(latest)
            zf.write(file_path, arcname=str(Path(latest.name) / arcname))

    if delete_dir:
        _print(f"[archive] Deleting run directory: {latest}")
        shutil.rmtree(latest, ignore_errors=False)

    return zip_path




def archive_run_dir(run_dir: Path, *, delete_dir: bool = True) -> Optional[Path]:
    """
    Zip a specific run directory into "<run_dir>.zip".

    This is the explicit, non-guessy sibling of archive_latest_run().
    It exists so the launcher can own post-processing (Plan A) without relying on
    "latest modified" heuristics.

    Args:
        run_dir: The run directory to archive.
        delete_dir: If True, delete the directory after a successful zip.

    Returns:
        Optional[Path]: Path to the created zip (or None if run_dir invalid).
    """
    if run_dir is None:
        return None
    run_dir = Path(run_dir)
    if (not run_dir.exists()) or (not run_dir.is_dir()):
        return None

    zip_path = run_dir.with_suffix(".zip")

    # Create zip
    try:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in run_dir.rglob("*"):
                if p.is_dir():
                    continue
                rel = p.relative_to(run_dir)
                zf.write(p, arcname=str(rel))
        # Basic integrity check
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                raise RuntimeError(f"Zip integrity check failed at: {bad}")
    except Exception:
        # If we partially created a zip, remove it to avoid confusing later steps.
        try:
            zip_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise

    if delete_dir:
        shutil.rmtree(run_dir, ignore_errors=True)

    return zip_path

def push_runs_to_windows(*, verbose: bool = True) -> None:
    """
    Mirror run artifacts to a Windows-accessible directory (WSL typical).

    Controlled by env var:
        WIN_RUNS_DIR=/mnt/d/_Projects/Emergence/runs

    Behavior (minimal + safe):
    - Copy top-level .zip files in runs/ into WIN_RUNS_DIR
    - Copy runs/batches/** manifests (small text/json/md) into WIN_RUNS_DIR/batches

    This is intentionally conservative to avoid syncing giant raw folders.

    Args:
        verbose: If True, print what is happening.
    """
    dst = os.environ.get("WIN_RUNS_DIR", "").strip()
    if not dst:
        if verbose:
            _print("[push] WIN_RUNS_DIR not set; skipping Windows mirroring.")
        return

    repo_root = get_repo_root()
    runs_dir = repo_root / "runs"
    dst_dir = Path(dst).expanduser()

    dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy top-level zips
    copied = 0
    if runs_dir.exists():
        for p in runs_dir.iterdir():
            if p.is_file() and p.suffix.lower() == ".zip":
                shutil.copy2(p, dst_dir / p.name)
                copied += 1

    # Copy batches manifests
    batches_src = runs_dir / "batches"
    batches_dst = dst_dir / "batches"
    if batches_src.exists():
        for p in batches_src.rglob("*"):
            if p.is_dir():
                continue
            # Only small-ish manifest formats
            if p.suffix.lower() not in {".json", ".txt", ".md", ".csv", ".yaml", ".yml"}:
                continue
            rel = p.relative_to(batches_src)
            out = batches_dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)
            copied += 1

    if verbose:
        _print(f"[push] Mirrored {copied} file(s) to: {dst_dir}")
