#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import shutil
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "eiel_parallel_campaign_execution_v1"
LIVE_MANIFEST_SCHEMA_VERSION = "eiel_parallel_campaign_live_manifest_v1"
CAMPAIGN_RUN_INDEX_SCHEMA_VERSION = "eiel_campaign_run_index_v1"
CAMPAIGN_RUN_INDEX_FILENAME = "campaign_run_index_v1.json"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slug(text: str, limit: int = 48) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum():
            out.append(ch.lower())
        elif ch in ("-", "_", "."):
            out.append(ch.lower())
        else:
            out.append("_")
    compact = "".join(out).strip("._-")
    while "__" in compact:
        compact = compact.replace("__", "_")
    return (compact or "run")[: int(limit)]


def _sanitize_cmd(cmd: list[str]) -> list[str]:
    # Keep commands reproducible without shell quoting ambiguity. This wrapper
    # does not accept secrets, but centralizing this keeps future additions safer.
    return [str(part) for part in cmd]


def _cmd_text(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def _tmux_safe_name(text: str, limit: int = 48) -> str:
    return _slug(text, limit).replace(".", "_")


def _child_log_paths(logs_dir: Path, *, index: int, label: str, run_id: str) -> tuple[Path, Path]:
    run_slug = _slug(label or run_id, 36)
    return (
        logs_dir / f"child_{int(index):04d}_{run_slug}.stdout.txt",
        logs_dir / f"child_{int(index):04d}_{run_slug}.stderr.txt",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_command_log(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _launch_progress_ui(
    *,
    args: argparse.Namespace,
    runs_root: Path,
    command_log: Path,
) -> dict[str, Any]:
    mode = str(getattr(args, "progress_ui", "auto") or "auto").strip().lower()
    payload: dict[str, Any] = {
        "requested_mode": mode,
        "launched": False,
        "kind": "",
        "session": "",
        "attach_command": "",
        "command": [],
        "error": "",
    }
    if mode == "off":
        payload["kind"] = "off"
        return payload
    if mode not in {"auto", "tmux"}:
        payload["kind"] = "invalid"
        payload["error"] = f"unsupported progress UI mode: {mode}"
        return payload
    tmux_bin = shutil.which("tmux")
    if not tmux_bin:
        payload["kind"] = "tmux"
        payload["error"] = "tmux not found; run eiel_campaign_tui.py manually with --runs-root"
        return payload
    session = str(getattr(args, "progress_ui_session", "") or "").strip()
    if not session:
        session = "eiel_" + _tmux_safe_name(f"{getattr(args, 'batch_tag', 'campaign')}_{runs_root.name}", 42)
    tui_script = _project_root() / "tools" / "scripts" / "eiel_campaign_tui.py"
    refresh_s = max(0.1, float(getattr(args, "progress_ui_refresh_s", 1.0) or 1.0))
    tui_cmd = [
        sys.executable,
        str(tui_script),
        "--runs-root",
        str(runs_root),
        "--refresh-s",
        f"{refresh_s:g}",
    ]
    tmux_cmd = [tmux_bin, "new-session", "-d", "-s", session, _cmd_text(tui_cmd)]
    payload.update(
        {
            "kind": "tmux",
            "session": session,
            "attach_command": f"tmux attach -t {shlex.quote(session)}",
            "command": _sanitize_cmd(tui_cmd),
        }
    )
    try:
        cp = subprocess.run(tmux_cmd, text=True, capture_output=True, check=False)
    except Exception as exc:
        payload["error"] = str(exc)
        return payload
    if cp.returncode == 0:
        payload["launched"] = True
        return payload
    stderr = (cp.stderr or "").strip()
    if "duplicate session" in stderr.lower():
        payload["launched"] = True
        payload["error"] = "tmux session already existed; reusing attach command"
        return payload
    payload["error"] = stderr or (cp.stdout or "").strip() or f"tmux exited with {cp.returncode}"
    return payload


def _run_capture(
    cmd: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    affinity_cpu_set: tuple[int, ...] = (),
) -> tuple[int, float]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    preexec_fn = None
    if affinity_cpu_set and hasattr(os, "sched_setaffinity"):
        cpu_set = set(int(cpu) for cpu in affinity_cpu_set)

        def _apply_affinity() -> None:
            os.sched_setaffinity(0, cpu_set)

        preexec_fn = _apply_affinity
    with stdout_path.open("w", encoding="utf-8", errors="replace") as out, stderr_path.open(
        "w", encoding="utf-8", errors="replace"
    ) as err:
        cp = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            stdout=out,
            stderr=err,
            check=False,
            preexec_fn=preexec_fn,
        )
    return int(cp.returncode), time.perf_counter() - started


def _path_state(path: Path) -> dict[str, Any]:
    try:
        st = path.stat()
    except FileNotFoundError:
        return {"exists": False, "is_file": False, "size": 0, "mtime_ns": 0}
    except OSError as exc:
        return {"exists": False, "is_file": False, "size": 0, "mtime_ns": 0, "error": str(exc)}
    return {
        "exists": True,
        "is_file": path.is_file(),
        "size": int(st.st_size),
        "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))),
    }


def _same_path_state(before: dict[str, Any], after: dict[str, Any]) -> bool:
    keys = ("exists", "is_file", "size", "mtime_ns")
    return all(before.get(key) == after.get(key) for key in keys)


def _canonical_path_text(path: Path | str) -> str:
    try:
        return str(Path(path).expanduser().resolve())
    except Exception:
        return str(path)


def _find_latest_summary_full(root: Path) -> Path | None:
    matches = sorted(root.rglob("summary.json"), key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
    return matches[-1] if matches else None


def _find_latest_summary(root: Path, index_entry: dict[str, Any] | None = None, counters: dict[str, int] | None = None) -> Path | None:
    if index_entry:
        summary_raw = str(index_entry.get("summary_path") or "").strip()
        if summary_raw:
            summary_path = Path(summary_raw)
            before = index_entry.get("summary_state", {}) if isinstance(index_entry.get("summary_state"), dict) else {}
            collection_raw = str(index_entry.get("summary_collection_path") or "").strip()
            collection_before = (
                index_entry.get("summary_collection_state", {})
                if isinstance(index_entry.get("summary_collection_state"), dict)
                else {}
            )
            after = _path_state(summary_path)
            collection_after = _path_state(Path(collection_raw)) if collection_raw else {}
            if (
                _same_path_state(before, after)
                and collection_raw
                and _same_path_state(collection_before, collection_after)
            ):
                if counters is not None:
                    counters["campaign_index_entries_reused"] = counters.get("campaign_index_entries_reused", 0) + 1
                    counters["campaign_index_full_scan_fallbacks"] = counters.get("campaign_index_full_scan_fallbacks", 0)
                return summary_path
            if counters is not None:
                counters["campaign_index_entries_refreshed"] = counters.get("campaign_index_entries_refreshed", 0) + 1
    if counters is not None:
        counters["campaign_index_full_scan_fallbacks"] = counters.get("campaign_index_full_scan_fallbacks", 0) + 1
    return _find_latest_summary_full(root)


def _load_matrix(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception as exc:
        return {"_load_error": str(exc)}


def _checkpoint_summary(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {
            "checkpoint_bundles_v2_exists": False,
            "checkpoint_bundle_file_count": 0,
            "checkpoints_index_exists": False,
        }
    bundle_dir = run_dir / "checkpoint_bundles_v2"
    files = [p for p in bundle_dir.rglob("*") if p.is_file()] if bundle_dir.exists() else []
    return {
        "checkpoint_bundles_v2_exists": bundle_dir.exists(),
        "checkpoint_bundle_file_count": len(files),
        "checkpoints_index_exists": (run_dir / "viz" / "checkpoints_index.json").exists(),
    }


def _artifact_summary(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {
            "run_dir_exists": False,
            "metrics_exists": False,
            "manifest_exists": False,
            "run_config_exists": False,
            **_checkpoint_summary(None),
        }
    return {
        "run_dir_exists": run_dir.exists(),
        "metrics_exists": (run_dir / "metrics.csv").exists(),
        "manifest_exists": (run_dir / "manifest.json").exists(),
        "run_config_exists": (run_dir / "run_config.csv").exists(),
        **_checkpoint_summary(run_dir),
    }


def _metrics_final_summary(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {"metrics_row_count": 0}
    metrics = run_dir / "metrics.csv"
    if not metrics.exists():
        return {"metrics_row_count": 0}
    try:
        csv.field_size_limit(sys.maxsize)
    except Exception:
        pass
    final: dict[str, str] = {}
    row_count = 0
    try:
        with metrics.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            for row in reader:
                if fieldnames and any(row.get(name) is None for name in fieldnames):
                    continue
                final = dict(row)
                row_count += 1
    except Exception as exc:
        return {"metrics_row_count": 0, "metrics_error": str(exc)}
    fields = [
        "alive",
        "deaths",
        "successful_births",
        "failed_births",
        "lineage_active_count",
        "lineage_top1_alive_share",
        "lineage_effective_alive_lineages",
    ]
    return {"metrics_row_count": row_count, "final_metrics": {name: final.get(name, "") for name in fields}}


def _campaign_index_counters() -> dict[str, int]:
    return {
        "campaign_index_entries_loaded": 0,
        "campaign_index_entries_reused": 0,
        "campaign_index_entries_refreshed": 0,
        "campaign_index_entries_added": 0,
        "campaign_index_entries_deleted": 0,
        "campaign_index_entries_repaired": 0,
        "campaign_index_full_scan_fallbacks": 0,
        "campaign_index_corruption_rebuilds": 0,
        "campaign_index_errors": 0,
    }


def _load_campaign_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    if payload.get("schema_version") != CAMPAIGN_RUN_INDEX_SCHEMA_VERSION:
        return {}
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return {}
    return payload


def _campaign_index_entry_from_result(row: dict[str, Any]) -> dict[str, Any]:
    summary_path = Path(str(row.get("summary_path") or ""))
    run_dir = Path(str(row.get("run_dir") or ""))
    metrics_path = run_dir / "metrics.csv" if str(run_dir) else Path("")
    summary_collection_path: Path | None = None
    if str(row.get("summary_path") or "") and len(summary_path.parents) >= 2:
        summary_collection_path = summary_path.parent.parent
    return {
        "index": int(row.get("index", 0) or 0),
        "label": str(row.get("label") or ""),
        "run_id": str(row.get("run_id") or ""),
        "registry": str(row.get("registry") or ""),
        "child_root": str(row.get("child_root") or ""),
        "summary_path": str(summary_path) if str(summary_path) else "",
        "summary_collection_path": str(summary_collection_path) if summary_collection_path else "",
        "run_dir": str(run_dir) if str(run_dir) else "",
        "metrics_path": str(metrics_path) if str(run_dir) else "",
        "status": str(row.get("status") or ""),
        "returncode": int(row.get("returncode", 0) or 0),
        "summary_state": _path_state(summary_path) if str(summary_path) else {},
        "summary_collection_state": _path_state(summary_collection_path) if summary_collection_path else {},
        "run_dir_state": _path_state(run_dir) if str(run_dir) else {},
        "metrics_state": _path_state(metrics_path) if str(run_dir) else {},
    }


def _result_success(row: dict[str, Any]) -> bool:
    try:
        returncode = int(row.get("returncode", 1))
    except Exception:
        returncode = 1
    return returncode == 0 and str(row.get("status") or "") in {"ok", "complete"}


def _build_campaign_index_payload(
    *,
    runs_root: Path,
    args: argparse.Namespace,
    results: list[dict[str, Any]],
    counters: dict[str, int] | None = None,
) -> dict[str, Any]:
    entries = [_campaign_index_entry_from_result(row) for row in sorted(results, key=lambda item: int(item.get("index", 0)))]
    payload = {
        "schema_version": CAMPAIGN_RUN_INDEX_SCHEMA_VERSION,
        "campaign_schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "runs_root": str(runs_root),
        "batch_tag": str(args.batch_tag),
        "registry": str(getattr(args, "registry", "") or ""),
        "case_mode": bool(getattr(args, "case", [])),
        "planned_runs": len(entries),
        "entries": entries,
        "counters": dict(counters or _campaign_index_counters()),
        "source_of_truth": "filesystem_child_artifacts",
        "disposable": True,
    }
    return payload


def _write_campaign_index(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _capture_command(cmd: list[str], *, timeout: int = 15) -> dict[str, Any]:
    try:
        cp = subprocess.run(cmd, text=True, capture_output=True, timeout=int(timeout), check=False)
        return {
            "command": _sanitize_cmd(cmd),
            "returncode": int(cp.returncode),
            "stdout": cp.stdout.strip(),
            "stderr": cp.stderr.strip(),
        }
    except Exception as exc:  # pragma: no cover - host dependent
        return {"command": _sanitize_cmd(cmd), "error": str(exc)}


def _capture_monitoring(root: Path, label: str) -> dict[str, Any]:
    snapshot_dir = root / "monitoring"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    commands = {
        "nvidia_smi": [
            "nvidia-smi",
            "--query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ],
        "free_h": ["free", "-h"],
        "lscpu": ["lscpu"],
        "df_runs_root": ["df", "-T", str(root)],
    }
    rows: dict[str, Any] = {"label": label, "captured_at_utc": _utc_now(), "commands": {}}
    for name, cmd in commands.items():
        payload = _capture_command(cmd)
        rows["commands"][name] = payload
        (snapshot_dir / f"{label}_{name}.txt").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    rows["snapshot_dir"] = str(snapshot_dir)
    return rows


def _current_affinity() -> tuple[int, ...]:
    if hasattr(os, "sched_getaffinity"):
        try:
            return tuple(sorted(int(cpu) for cpu in os.sched_getaffinity(0)))
        except Exception:
            pass
    count = os.cpu_count() or 1
    return tuple(range(int(count)))


def _parse_cpu_set(text: str) -> tuple[int, ...]:
    cpus: set[int] = set()
    for part in str(text or "").split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_raw, end_raw = item.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                raise ValueError(f"invalid descending CPU range: {item}")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(item))
    if not cpus:
        raise ValueError("CPU set must not be empty")
    if min(cpus) < 0:
        raise ValueError("CPU ids must be non-negative")
    return tuple(sorted(cpus))


def _lscpu_e() -> str:
    payload = _capture_command(["lscpu", "-e=CPU,CORE,SOCKET,NODE"], timeout=5)
    return str(payload.get("stdout") or payload.get("stderr") or "")


def _resolve_worker_cpu_sets(
    *,
    mode: str,
    workers: int,
    manual_sets: list[str],
    available_cpus: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    selected_mode = str(mode or "none").strip().lower()
    worker_count = max(1, int(workers))
    cpus = tuple(sorted(int(cpu) for cpu in (available_cpus if available_cpus is not None else _current_affinity())))
    payload: dict[str, Any] = {
        "mode": selected_mode,
        "available_cpus": list(cpus),
        "worker_cpu_sets": [[] for _ in range(worker_count)],
        "applied": False,
        "warning": "",
        "lscpu_e": _lscpu_e(),
    }
    if selected_mode == "none":
        return payload
    if not cpus:
        payload["warning"] = "no available CPUs detected; affinity disabled"
        payload["mode"] = "none"
        return payload
    if selected_mode == "manual":
        if len(manual_sets) != worker_count:
            raise ValueError("--worker-cpu-sets must provide exactly one CPU set per worker in manual mode")
        resolved = [_parse_cpu_set(raw) for raw in manual_sets]
    elif selected_mode == "compact":
        width = max(1, int(math.floor(len(cpus) / worker_count)))
        resolved = []
        for idx in range(worker_count):
            start = idx * width
            chunk = cpus[start : start + width]
            if not chunk:
                chunk = (cpus[idx % len(cpus)],)
            resolved.append(tuple(chunk))
    elif selected_mode == "spread":
        half = max(1, int(math.ceil(len(cpus) / 2)))
        resolved = []
        if worker_count * 2 <= len(cpus):
            for idx in range(worker_count):
                first = cpus[idx % len(cpus)]
                second = cpus[(idx + half) % len(cpus)]
                worker_set = tuple(sorted({first, second}))
                resolved.append(worker_set)
        else:
            interleaved = []
            for left, right in zip(cpus[:half], cpus[half:]):
                interleaved.extend([left, right])
            interleaved.extend(cpu for cpu in cpus if cpu not in interleaved)
            for idx in range(worker_count):
                resolved.append((interleaved[idx % len(interleaved)],))
    else:
        raise ValueError("--worker-affinity-mode must be one of none, compact, spread, manual")
    available = set(cpus)
    for worker_set in resolved:
        outside = [cpu for cpu in worker_set if cpu not in available]
        if outside:
            raise ValueError(f"CPU set {list(worker_set)} includes CPUs outside current affinity {list(cpus)}")
    payload["worker_cpu_sets"] = [list(worker_set) for worker_set in resolved]
    payload["applied"] = bool(resolved)
    return payload


def _monitor_loop(root: Path, stop: threading.Event, interval_s: float, out: list[dict[str, Any]]) -> None:
    sample = 0
    while not stop.is_set():
        out.append(_capture_monitoring(root, f"during_{sample:03d}"))
        sample += 1
        stop.wait(max(1.0, float(interval_s)))


@dataclass(frozen=True)
class ChildSpec:
    index: int
    label: str
    run_id: str
    registry: str
    child_root: str
    batch_tag: str
    knobs: tuple[str, ...]
    performance_profile: str
    campaign_output_profile: str
    allow_large_output: bool
    allow_generation_checkpointing: bool
    perf: bool
    overwrite: bool
    command_log: str
    logs_dir: str
    campaign_index_entry: dict[str, Any] | None = None
    worker_slot: int = 0
    affinity_mode: str = "none"
    affinity_cpu_set: tuple[int, ...] = ()


def _run_child(spec: ChildSpec) -> dict[str, Any]:
    project_root = _project_root()
    child_root = Path(spec.child_root)
    logs_dir = Path(spec.logs_dir)
    stdout_path, stderr_path = _child_log_paths(logs_dir, index=spec.index, label=spec.label, run_id=spec.run_id)
    cmd = [
        sys.executable,
        "_runner/run_registry_batch.py",
        "--registry",
        spec.registry,
        "--runs_root",
        str(child_root),
        "--batch_tag",
        spec.batch_tag,
        "--only_run_id",
        spec.run_id,
    ]
    if spec.overwrite:
        cmd.append("--overwrite")
    if spec.perf:
        cmd.append("--perf")
    if spec.performance_profile:
        cmd.extend(["--performance_profile", str(spec.performance_profile)])
    if spec.campaign_output_profile:
        cmd.extend(["--campaign_output_profile", str(spec.campaign_output_profile)])
    if spec.allow_large_output:
        cmd.append("--allow_large_output")
    if spec.allow_generation_checkpointing:
        cmd.append("--allow_generation_checkpointing")
    for knob in spec.knobs:
        cmd.extend(["--knob", str(knob)])
    affinity_before = _current_affinity()
    affinity_after = affinity_before
    affinity_error = ""
    affinity_applied = False
    if spec.affinity_cpu_set:
        try:
            if not hasattr(os, "sched_setaffinity"):
                raise RuntimeError("os.sched_setaffinity is unavailable")
            os.sched_setaffinity(0, set(int(cpu) for cpu in spec.affinity_cpu_set))
            affinity_after = _current_affinity()
            affinity_applied = tuple(sorted(spec.affinity_cpu_set)) == affinity_after
        except Exception as exc:
            affinity_error = str(exc)
            affinity_after = _current_affinity()
    _append_command_log(
        Path(spec.command_log),
        {
            "kind": "child_start",
            "label": spec.label,
            "run_id": spec.run_id,
            "index": int(spec.index),
            "worker_slot": int(spec.worker_slot),
            "affinity_mode": spec.affinity_mode,
            "configured_affinity": list(spec.affinity_cpu_set),
            "observed_affinity": list(affinity_after),
            "affinity_applied": bool(affinity_applied),
            "affinity_error": affinity_error,
            "timestamp_utc": _utc_now(),
            "cmd": _sanitize_cmd(cmd),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        },
    )
    rc, wall_s = _run_capture(
        cmd,
        cwd=project_root,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        affinity_cpu_set=spec.affinity_cpu_set,
    )
    lookup_counters = _campaign_index_counters()
    summary_path = _find_latest_summary(child_root, spec.campaign_index_entry, lookup_counters)
    summary = _load_json(summary_path)
    run_dir: Path | None = None
    if summary.get("runs"):
        first = summary["runs"][0] if isinstance(summary["runs"], list) else {}
        run_dir_raw = str(first.get("run_dir") or "").strip()
        run_dir = Path(run_dir_raw) if run_dir_raw else None
    row_status = ""
    if summary.get("runs") and isinstance(summary["runs"], list):
        row_status = str(summary["runs"][0].get("status") or "")
    result = {
        "index": int(spec.index),
        "label": spec.label,
        "run_id": spec.run_id,
        "registry": spec.registry,
        "child_root": str(child_root),
        "batch_tag": spec.batch_tag,
        "returncode": int(rc),
        "status": row_status or ("ok" if rc == 0 else "failed"),
        "wall_s": float(wall_s),
        "worker_slot": int(spec.worker_slot),
        "affinity_mode": spec.affinity_mode,
        "configured_affinity": list(spec.affinity_cpu_set),
        "observed_affinity_before": list(affinity_before),
        "observed_affinity_after": list(affinity_after),
        "affinity_applied": bool(affinity_applied),
        "affinity_error": affinity_error,
        "summary_path": str(summary_path) if summary_path else "",
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "run_dir": str(run_dir) if run_dir else "",
        "campaign_index_lookup_counters": lookup_counters,
        **_artifact_summary(run_dir),
        **_metrics_final_summary(run_dir),
    }
    _append_command_log(
        Path(spec.command_log),
        {
            "kind": "child_end",
            "label": spec.label,
            "run_id": spec.run_id,
            "index": int(spec.index),
            "worker_slot": int(spec.worker_slot),
            "affinity_mode": spec.affinity_mode,
            "configured_affinity": list(spec.affinity_cpu_set),
            "observed_affinity": list(affinity_after),
            "affinity_applied": bool(affinity_applied),
            "affinity_error": affinity_error,
            "timestamp_utc": _utc_now(),
            "returncode": int(rc),
            "wall_s": float(wall_s),
            "summary_path": result["summary_path"],
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        },
    )
    return result


def _result_from_child_artifacts(
    spec: ChildSpec,
    *,
    returncode: int,
    wall_s: float,
    index_entry: dict[str, Any] | None = None,
    previous_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    child_root = Path(spec.child_root)
    stdout_path, stderr_path = _child_log_paths(Path(spec.logs_dir), index=spec.index, label=spec.label, run_id=spec.run_id)
    lookup_counters = _campaign_index_counters()
    summary_path = _find_latest_summary(child_root, index_entry, lookup_counters)
    summary = _load_json(summary_path)
    run_dir: Path | None = None
    if summary.get("runs"):
        first = summary["runs"][0] if isinstance(summary["runs"], list) else {}
        run_dir_raw = str(first.get("run_dir") or "").strip()
        run_dir = Path(run_dir_raw) if run_dir_raw else None
    row_status = ""
    summary_returncode: int | None = None
    if summary.get("runs") and isinstance(summary["runs"], list):
        first_run = summary["runs"][0]
        row_status = str(first_run.get("status") or "")
        try:
            summary_returncode = int(first_run.get("returncode"))
        except Exception:
            summary_returncode = None
    if not row_status:
        if summary_path is None:
            row_status = "missing"
        elif summary.get("_load_error"):
            row_status = "partial"
        else:
            row_status = "ok" if int(returncode) == 0 else "failed"
    effective_returncode = int(summary_returncode) if row_status == "complete" and summary_returncode is not None else int(returncode)
    previous_result = previous_result or {}
    return {
        "index": int(spec.index),
        "label": spec.label,
        "run_id": spec.run_id,
        "registry": spec.registry,
        "child_root": str(child_root),
        "batch_tag": spec.batch_tag,
        "returncode": effective_returncode,
        "status": row_status,
        "wall_s": float(wall_s),
        "worker_slot": int(spec.worker_slot),
        "affinity_mode": spec.affinity_mode,
        "configured_affinity": list(spec.affinity_cpu_set),
        "observed_affinity_before": list(previous_result.get("observed_affinity_before", [])),
        "observed_affinity_after": list(previous_result.get("observed_affinity_after", [])),
        "affinity_applied": bool(previous_result.get("affinity_applied", False)),
        "affinity_error": str(previous_result.get("affinity_error", "")),
        "summary_path": str(summary_path) if summary_path else "",
        "stdout_path": str(previous_result.get("stdout_path") or stdout_path),
        "stderr_path": str(previous_result.get("stderr_path") or stderr_path),
        "run_dir": str(run_dir) if run_dir else "",
        "campaign_index_lookup_counters": lookup_counters,
        **_artifact_summary(run_dir),
        **_metrics_final_summary(run_dir),
    }


def _aggregate_specs_from_manifest(
    *,
    runs_root: Path,
    args: argparse.Namespace,
    command_log: Path,
    logs_dir: Path,
    previous_results: dict[str, dict[str, Any]],
) -> list[ChildSpec]:
    manifest = _load_json(runs_root / "parallel_live_manifest.json")
    children = manifest.get("children") if isinstance(manifest.get("children"), list) else []
    specs: list[ChildSpec] = []
    for fallback_index, child in enumerate(children, start=1):
        if not isinstance(child, dict):
            continue
        index = int(child.get("index", fallback_index) or fallback_index)
        run_id = str(child.get("run_id") or "").strip()
        if not run_id:
            continue
        specs.append(
            ChildSpec(
                index=index,
                label=str(child.get("label") or _slug(run_id, 24)),
                run_id=run_id,
                registry=str(child.get("registry") or manifest.get("registry") or getattr(args, "registry", "") or ""),
                child_root=str(child.get("child_root") or ""),
                batch_tag=str(child.get("batch_tag") or getattr(args, "batch_tag", "")),
                knobs=tuple(),
                performance_profile=str(getattr(args, "performance_profile", "") or ""),
                campaign_output_profile=str(getattr(args, "campaign_output_profile", "") or ""),
                allow_large_output=bool(getattr(args, "allow_large_output", False)),
                allow_generation_checkpointing=bool(getattr(args, "allow_generation_checkpointing", False)),
                perf=bool(getattr(args, "perf", False)),
                overwrite=False,
                command_log=str(command_log),
                logs_dir=str(logs_dir),
                worker_slot=int(child.get("worker_slot", (index - 1)) or 0),
                affinity_mode=str(child.get("affinity_mode") or "none"),
                affinity_cpu_set=tuple(int(cpu) for cpu in (child.get("configured_affinity") or [])),
            )
        )
    children_root = runs_root / "children"
    if specs:
        seen_roots = {_canonical_path_text(spec.child_root) for spec in specs}
        next_index = max((int(spec.index) for spec in specs), default=0) + 1
        for child_root in sorted(p for p in children_root.glob("*") if p.is_dir()):
            if _canonical_path_text(child_root) in seen_roots:
                continue
            run_id = child_root.name
            specs.append(
                ChildSpec(
                    index=next_index,
                    label=_slug(run_id, 24),
                    run_id=run_id,
                    registry=str(getattr(args, "registry", "") or ""),
                    child_root=str(child_root),
                    batch_tag=str(getattr(args, "batch_tag", "")),
                    knobs=tuple(),
                    performance_profile=str(getattr(args, "performance_profile", "") or ""),
                    campaign_output_profile=str(getattr(args, "campaign_output_profile", "") or ""),
                    allow_large_output=bool(getattr(args, "allow_large_output", False)),
                    allow_generation_checkpointing=bool(getattr(args, "allow_generation_checkpointing", False)),
                    perf=bool(getattr(args, "perf", False)),
                    overwrite=False,
                    command_log=str(command_log),
                    logs_dir=str(logs_dir),
                    worker_slot=0,
                    affinity_mode="none",
                    affinity_cpu_set=tuple(),
                )
            )
            next_index += 1
        return sorted(specs, key=lambda item: int(item.index))

    for index, child_root in enumerate(sorted(p for p in children_root.glob("*") if p.is_dir()), start=1):
        previous = next((row for row in previous_results.values() if str(row.get("child_root")) == str(child_root)), {})
        run_id = str(previous.get("run_id") or child_root.name)
        specs.append(
            ChildSpec(
                index=int(previous.get("index", index) or index),
                label=str(previous.get("label") or _slug(run_id, 24)),
                run_id=run_id,
                registry=str(previous.get("registry") or getattr(args, "registry", "") or ""),
                child_root=str(child_root),
                batch_tag=str(previous.get("batch_tag") or getattr(args, "batch_tag", "")),
                knobs=tuple(),
                performance_profile=str(getattr(args, "performance_profile", "") or ""),
                campaign_output_profile=str(getattr(args, "campaign_output_profile", "") or ""),
                allow_large_output=bool(getattr(args, "allow_large_output", False)),
                allow_generation_checkpointing=bool(getattr(args, "allow_generation_checkpointing", False)),
                perf=bool(getattr(args, "perf", False)),
                overwrite=False,
                command_log=str(command_log),
                logs_dir=str(logs_dir),
                worker_slot=int(previous.get("worker_slot", (index - 1)) or 0),
                affinity_mode=str(previous.get("affinity_mode") or "none"),
                affinity_cpu_set=tuple(int(cpu) for cpu in (previous.get("configured_affinity") or [])),
            )
        )
    return sorted(specs, key=lambda item: int(item.index))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run independent EIEL registry cells with bounded process workers")
    parser.add_argument("--registry", default="", help="Registry path accepted by _runner/run_registry_batch.py")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Explicit mixed-registry case: label=registry::run_id. May be repeated.",
    )
    parser.add_argument("--runs-root", required=True, help="Parallel campaign runtime output root")
    parser.add_argument("--batch-tag", required=True, help="Human-readable campaign batch tag")
    parser.add_argument("--workers", type=int, default=1, help="Process worker count. Default: 1")
    parser.add_argument("--only-run-id", action="append", default=[], help="Selected run_id. May be repeated")
    parser.add_argument("--run-id-file", default="", help="Optional newline-delimited selected run_id file")
    parser.add_argument("--knob", action="append", default=[], help="Extra knob passed to each child harness")
    parser.add_argument("--device", default="", choices=["", "auto", "cpu", "cuda"], help="Optional sim_cfg.device override")
    parser.add_argument(
        "--bank-device",
        "--bank_device",
        dest="bank_device",
        default="",
        choices=["", "auto", "cpu", "cuda"],
        help="Optional sim_cfg.bank_device override",
    )
    parser.add_argument("--perf", action="store_true", help="Enable perf buckets in each child harness")
    parser.add_argument(
        "--performance-profile",
        "--performance_profile",
        dest="performance_profile",
        default="",
        help="Named simulation performance profile applied to every child after generic --knob overrides",
    )
    parser.add_argument(
        "--campaign-output-profile",
        "--campaign_output_profile",
        dest="campaign_output_profile",
        default="compact_default",
        help="Named campaign output/storage profile applied to every child",
    )
    parser.add_argument(
        "--allow-large-output",
        "--allow_large_output",
        dest="allow_large_output",
        action="store_true",
        help="Acknowledge large debug/full-replay output profile risk",
    )
    parser.add_argument(
        "--allow-generation-checkpointing",
        "--allow_generation_checkpointing",
        dest="allow_generation_checkpointing",
        action="store_true",
        help="Allow per-generation checkpointing when full_replay is selected",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing isolated child roots")
    parser.add_argument("--dry-run", action="store_true", help="Plan selected rows and write summaries without child execution")
    parser.add_argument(
        "--aggregate-only",
        "--aggregate_only",
        dest="aggregate_only",
        action="store_true",
        help="Rebuild campaign summary/status from existing child artifacts without launching simulations",
    )
    parser.add_argument(
        "--status-only",
        "--status_only",
        dest="status_only",
        action="store_true",
        help="Alias for --aggregate-only for read-only campaign status refreshes",
    )
    parser.add_argument(
        "--campaign-index",
        "--campaign_index",
        dest="campaign_index",
        action="store_true",
        help="Write a disposable campaign_run_index_v1.json sidecar for repeated aggregation/status reads",
    )
    parser.add_argument(
        "--rebuild-campaign-index",
        "--rebuild_campaign_index",
        dest="rebuild_campaign_index",
        action="store_true",
        help="Ignore any existing campaign_run_index_v1.json and rebuild it from filesystem artifacts",
    )
    parser.add_argument("--capture-monitoring", action="store_true", help="Capture lightweight GPU/host snapshots")
    parser.add_argument("--monitor-interval-s", type=float, default=15.0, help="Monitoring sample interval")
    parser.add_argument(
        "--progress-ui",
        default="auto",
        choices=["auto", "tmux", "off"],
        help="Launch the live terminal campaign UI for non-dry-run campaigns. Default: auto",
    )
    parser.add_argument(
        "--progress-ui-session",
        default="",
        help="Optional tmux session name for the live terminal campaign UI",
    )
    parser.add_argument(
        "--progress-ui-refresh-s",
        type=float,
        default=1.0,
        help="Refresh interval passed to the live terminal campaign UI",
    )
    parser.add_argument(
        "--worker-affinity-mode",
        default="none",
        choices=["none", "compact", "spread", "manual"],
        help="Optional Linux CPU-affinity mode for worker subprocesses. Default: none",
    )
    parser.add_argument(
        "--worker-cpu-sets",
        nargs="*",
        default=[],
        help="Manual per-worker CPU sets, for example: --worker-cpu-sets 0,1 2,3 4,5",
    )
    return parser.parse_args(argv)


def _selected_run_ids(args: argparse.Namespace) -> list[str]:
    out = [str(x).strip() for x in (args.only_run_id or []) if str(x).strip()]
    file_arg = str(args.run_id_file or "").strip()
    if file_arg:
        path = Path(file_arg).expanduser()
        out.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    seen: set[str] = set()
    deduped: list[str] = []
    for item in out:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def _parse_cases(raw_cases: list[str]) -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    for raw in raw_cases or []:
        text = str(raw).strip()
        if not text:
            continue
        if "=" in text:
            label, body = text.split("=", 1)
            label = label.strip()
        else:
            label, body = "", text
        if "::" not in body:
            raise ValueError("--case must use label=registry::run_id")
        registry, run_id = body.split("::", 1)
        registry = registry.strip()
        run_id = run_id.strip()
        if not registry or not run_id:
            raise ValueError("--case requires non-empty registry and run_id")
        cases.append({"label": label or _slug(run_id, 24), "registry": registry, "run_id": run_id})
    return cases


def _effective_knobs(args: argparse.Namespace) -> list[str]:
    knobs = [str(x).strip() for x in (args.knob or []) if str(x).strip()]
    if str(args.device or "").strip():
        knobs.append(f"sim_cfg.device={str(args.device).strip()}")
    if str(args.bank_device or "").strip():
        knobs.append(f"sim_cfg.bank_device={str(args.bank_device).strip()}")
    return knobs


def _write_status_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "index",
        "worker_slot",
        "label",
        "run_id",
        "registry",
        "returncode",
        "status",
        "wall_s",
        "affinity_mode",
        "configured_affinity",
        "observed_affinity_after",
        "affinity_applied",
        "affinity_error",
        "run_dir",
        "metrics_exists",
        "run_config_exists",
        "manifest_exists",
        "checkpoint_bundles_v2_exists",
        "checkpoint_bundle_file_count",
        "summary_path",
        "stdout_path",
        "stderr_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(results, key=lambda item: int(item.get("index", 0))):
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_live_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    runs_root: Path,
    command_log: Path,
    summary_json: Path,
    status_csv: Path,
    specs: list[ChildSpec],
) -> None:
    children: list[dict[str, Any]] = []
    for spec in sorted(specs, key=lambda item: int(item.index)):
        stdout_path, stderr_path = _child_log_paths(
            Path(spec.logs_dir), index=spec.index, label=spec.label, run_id=spec.run_id
        )
        children.append(
            {
                "index": int(spec.index),
                "worker_slot": int(spec.worker_slot),
                "label": spec.label,
                "run_id": spec.run_id,
                "registry": spec.registry,
                "child_root": spec.child_root,
                "batch_tag": spec.batch_tag,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "affinity_mode": spec.affinity_mode,
                "configured_affinity": list(spec.affinity_cpu_set),
            }
        )
    _write_json(
        path,
        {
            "schema_version": LIVE_MANIFEST_SCHEMA_VERSION,
            "campaign_schema_version": SCHEMA_VERSION,
            "created_at_utc": _utc_now(),
            "runs_root": str(runs_root),
            "batch_tag": str(args.batch_tag),
            "workers": max(1, int(args.workers)),
            "planned_runs": len(children),
            "command_log_path": str(command_log),
            "summary_path": str(summary_json),
            "status_csv_path": str(status_csv),
            "children": children,
        },
    )


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# EIEL Parallel Campaign Summary",
        "",
        f"- schema: `{payload['schema_version']}`",
        f"- registry: `{payload['registry']}`",
        f"- runs_root: `{payload['runs_root']}`",
        f"- batch_tag: `{payload['batch_tag']}`",
        f"- workers: {payload['workers']}",
        f"- performance profile: `{payload.get('performance_profile', '') or ''}`",
        f"- status: `{payload['status']}`",
        f"- planned runs: {payload['planned_runs']}",
        f"- failures: {len(payload['failures'])}",
        f"- total wall seconds: {payload.get('total_wall_s', 0.0):.3f}",
        f"- worker affinity mode: `{payload.get('worker_affinity_mode', 'none')}`",
        "",
        "| index | worker | run_id | status | rc | wall_s | affinity | applied | metrics | checkpoint files |",
        "|---:|---:|---|---|---:|---:|---|---|---|---:|",
    ]
    for row in sorted(payload.get("results", []), key=lambda item: int(item.get("index", 0))):
        affinity = ",".join(str(cpu) for cpu in row.get("configured_affinity", []) or [])
        lines.append(
            "| {index} | {worker} | `{run_id}` | {status} | {returncode} | {wall_s:.3f} | {affinity} | {applied} | {metrics} | {ckpt} |".format(
                index=row.get("index", ""),
                worker=row.get("worker_slot", ""),
                run_id=row.get("run_id", ""),
                status=row.get("status", ""),
                returncode=row.get("returncode", ""),
                wall_s=float(row.get("wall_s", 0.0) or 0.0),
                affinity=affinity or "",
                applied="yes" if row.get("affinity_applied") else "no",
                metrics="yes" if row.get("metrics_exists") else "no",
                ckpt=int(row.get("checkpoint_bundle_file_count", 0) or 0),
            )
        )
    if payload.get("failures"):
        lines.extend(["", "## Failures", ""])
        for row in payload["failures"]:
            lines.append(f"- `{row.get('run_id', '')}` status={row.get('status')} rc={row.get('returncode')}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _base_payload(args: argparse.Namespace, runs_root: Path, knobs: list[str]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "registry": str(args.registry),
        "case_mode": bool(args.case),
        "runs_root": str(runs_root),
        "batch_tag": str(args.batch_tag),
        "workers": max(1, int(args.workers)),
        "parallel_mode": "process_pool" if max(1, int(args.workers)) > 1 else "serial_child_processes",
        "selected_run_ids_requested": _selected_run_ids(args),
        "effective_knobs": list(knobs),
        "performance_profile": str(getattr(args, "performance_profile", "") or "").strip(),
        "campaign_output_profile": str(getattr(args, "campaign_output_profile", "") or "").strip(),
        "allow_large_output": bool(getattr(args, "allow_large_output", False)),
        "allow_generation_checkpointing": bool(getattr(args, "allow_generation_checkpointing", False)),
        "perf": bool(args.perf),
        "worker_affinity_mode": str(getattr(args, "worker_affinity_mode", "none")),
        "worker_affinity": {},
        "started_at_utc": _utc_now(),
        "ended_at_utc": "",
        "total_wall_s": 0.0,
        "planned_runs": 0,
        "plan": {},
        "results": [],
        "failures": [],
        "monitoring": [],
        "progress_ui": {},
        "status": "started",
    }


def _plan_registry(
    *,
    registry: str,
    planning_root: Path,
    logs_dir: Path,
    command_log: Path,
    batch_tag: str,
    perf: bool,
    knobs: list[str],
    performance_profile: str,
    campaign_output_profile: str,
    allow_large_output: bool,
    allow_generation_checkpointing: bool,
    log_suffix: str,
) -> dict[str, Any]:
    plan_cmd = [
        sys.executable,
        "_runner/run_registry_batch.py",
        "--registry",
        str(registry),
        "--runs_root",
        str(planning_root),
        "--batch_tag",
        f"{batch_tag}_parallel_plan_{log_suffix}",
        "--dry_run",
        "--overwrite",
    ]
    if perf:
        plan_cmd.append("--perf")
    if performance_profile:
        plan_cmd.extend(["--performance_profile", str(performance_profile)])
    if campaign_output_profile:
        plan_cmd.extend(["--campaign_output_profile", str(campaign_output_profile)])
    if allow_large_output:
        plan_cmd.append("--allow_large_output")
    if allow_generation_checkpointing:
        plan_cmd.append("--allow_generation_checkpointing")
    for knob in knobs:
        plan_cmd.extend(["--knob", knob])
    _append_command_log(
        command_log,
        {"kind": "plan_start", "timestamp_utc": _utc_now(), "registry": registry, "cmd": _sanitize_cmd(plan_cmd)},
    )
    plan_rc, plan_wall = _run_capture(
        plan_cmd,
        cwd=_project_root(),
        stdout_path=logs_dir / f"plan_{log_suffix}.stdout.txt",
        stderr_path=logs_dir / f"plan_{log_suffix}.stderr.txt",
    )
    plan_summary = _find_latest_summary(planning_root)
    plan_batch = plan_summary.parent if plan_summary else None
    matrix_path = plan_batch / "matrix.csv" if plan_batch else None
    rows = _load_matrix(matrix_path) if matrix_path and matrix_path.exists() else []
    payload = {
        "registry": registry,
        "returncode": int(plan_rc),
        "wall_s": float(plan_wall),
        "summary_path": str(plan_summary) if plan_summary else "",
        "matrix_path": str(matrix_path) if matrix_path and matrix_path.exists() else "",
        "planned_rows": len(rows),
    }
    _append_command_log(
        command_log,
        {
            "kind": "plan_end",
            "timestamp_utc": _utc_now(),
            "registry": registry,
            "returncode": int(plan_rc),
            "wall_s": float(plan_wall),
            "summary_path": payload["summary_path"],
        },
    )
    payload["_rows"] = rows
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    aggregate_mode = bool(getattr(args, "aggregate_only", False) or getattr(args, "status_only", False))
    try:
        explicit_cases = _parse_cases(list(args.case or []))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not aggregate_mode and not str(args.registry or "").strip() and not explicit_cases:
        print("--registry is required unless --case is provided", file=sys.stderr)
        return 2
    workers = max(1, int(args.workers))
    runs_root = Path(args.runs_root).expanduser().resolve()
    logs_dir = runs_root / "logs"
    children_root = runs_root / "children"
    planning_root = runs_root / "_planning"
    summary_json = runs_root / "parallel_campaign_summary.json"
    summary_md = runs_root / "parallel_campaign_summary.md"
    status_csv = runs_root / "parallel_child_status.csv"
    command_log = runs_root / "parallel_command_log.jsonl"
    runs_root.mkdir(parents=True, exist_ok=True)
    knobs = _effective_knobs(args)
    payload = _base_payload(args, runs_root, knobs)
    performance_profile = str(getattr(args, "performance_profile", "") or "").strip()
    campaign_output_profile = str(getattr(args, "campaign_output_profile", "") or "").strip()
    try:
        affinity_plan = _resolve_worker_cpu_sets(
            mode=str(args.worker_affinity_mode),
            workers=workers,
            manual_sets=list(args.worker_cpu_sets or []),
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        payload["status"] = "failed_planning"
        payload["failures"] = [{"run_id": "", "status": "invalid_affinity", "returncode": 2, "error": str(exc)}]
        payload["worker_affinity"] = {"error": str(exc)}
        payload["ended_at_utc"] = _utc_now()
        payload["total_wall_s"] = 0.0
        _write_json(summary_json, payload)
        _write_markdown(summary_md, payload)
        _write_status_csv(status_csv, payload["failures"])
        return 2
    payload["worker_affinity"] = affinity_plan
    payload["worker_affinity_mode"] = str(affinity_plan.get("mode", args.worker_affinity_mode))
    started = time.perf_counter()

    if bool(args.capture_monitoring):
        payload["monitoring"].append(_capture_monitoring(runs_root, "before"))
    campaign_index_path = runs_root / CAMPAIGN_RUN_INDEX_FILENAME
    campaign_index_counters = _campaign_index_counters()
    existing_campaign_index = (
        {}
        if bool(getattr(args, "rebuild_campaign_index", False))
        else (_load_campaign_index(campaign_index_path) if bool(args.campaign_index) else {})
    )
    existing_campaign_entries = {
        str(entry.get("run_id") or ""): entry
        for entry in existing_campaign_index.get("entries", [])
        if isinstance(entry, dict) and str(entry.get("run_id") or "")
    }
    if bool(args.campaign_index):
        if existing_campaign_index:
            campaign_index_counters["campaign_index_entries_loaded"] = len(existing_campaign_index.get("entries", []))
        elif campaign_index_path.exists():
            campaign_index_counters["campaign_index_corruption_rebuilds"] = 1
        payload["campaign_index"] = {
            "enabled": True,
            "path": str(campaign_index_path),
            "schema_version": CAMPAIGN_RUN_INDEX_SCHEMA_VERSION,
            "counters": dict(campaign_index_counters),
        }

    if aggregate_mode:
        previous_summary = _load_json(summary_json)
        previous_results = {
            str(row.get("run_id") or ""): row
            for row in previous_summary.get("results", [])
            if isinstance(row, dict) and str(row.get("run_id") or "")
        }
        specs = _aggregate_specs_from_manifest(
            runs_root=runs_root,
            args=args,
            command_log=command_log,
            logs_dir=logs_dir,
            previous_results=previous_results,
        )
        payload["aggregate_mode"] = {
            "enabled": True,
            "status_only": bool(getattr(args, "status_only", False)),
            "index_enabled": bool(getattr(args, "campaign_index", False)),
            "index_rebuild_requested": bool(getattr(args, "rebuild_campaign_index", False)),
            "source_of_truth": "filesystem_child_artifacts",
            "previous_summary_path": str(summary_json) if previous_summary else "",
        }
        payload["planned_runs"] = len(specs)
        payload["selected_run_ids"] = [spec.run_id for spec in specs]
        payload["selected_cases"] = [
            {"label": spec.label, "registry": spec.registry, "run_id": spec.run_id}
            for spec in specs
        ]
        results = []
        index_entries = {} if bool(getattr(args, "rebuild_campaign_index", False)) else existing_campaign_entries
        aggregate_started = time.perf_counter()
        for spec in specs:
            previous = previous_results.get(spec.run_id, {})
            result = _result_from_child_artifacts(
                spec,
                returncode=int(previous.get("returncode", 0) or 0),
                wall_s=float(previous.get("wall_s", 0.0) or 0.0),
                index_entry=index_entries.get(spec.run_id),
                previous_result=previous,
            )
            if bool(args.campaign_index):
                for key, value in (result.get("campaign_index_lookup_counters") or {}).items():
                    if key in campaign_index_counters:
                        campaign_index_counters[key] += int(value or 0)
            results.append(result)
        payload["results"] = sorted(results, key=lambda item: int(item.get("index", 0)))
        payload["failures"] = [
            row
            for row in payload["results"]
            if not _result_success(row)
        ]
        if bool(args.campaign_index):
            loaded_run_ids = set(index_entries.keys())
            result_run_ids = {str(row.get("run_id") or "") for row in payload["results"] if str(row.get("run_id") or "")}
            campaign_index_counters["campaign_index_entries_added"] += len(result_run_ids - loaded_run_ids)
            campaign_index_counters["campaign_index_entries_deleted"] += len(loaded_run_ids - result_run_ids)
        payload["status"] = "complete" if not payload["failures"] else "incomplete"
        payload["ended_at_utc"] = _utc_now()
        payload["total_wall_s"] = time.perf_counter() - started
        payload["mean_child_wall_s"] = (
            sum(float(row.get("wall_s", 0.0) or 0.0) for row in payload["results"]) / max(1, len(payload["results"]))
        )
        payload["aggregate_mode"]["aggregate_wall_s"] = time.perf_counter() - aggregate_started
        payload["aggregate_mode"]["runs_total"] = len(payload["results"])
        if bool(args.campaign_index):
            campaign_index_payload = _build_campaign_index_payload(
                runs_root=runs_root,
                args=args,
                results=payload["results"],
                counters=campaign_index_counters,
            )
            _write_campaign_index(campaign_index_path, campaign_index_payload)
            payload["campaign_index"]["counters"] = dict(campaign_index_counters)
        _write_json(summary_json, payload)
        _write_markdown(summary_md, payload)
        _write_status_csv(status_csv, payload["results"])
        print(
            json.dumps(
                {
                    "summary": str(summary_json),
                    "aggregate_only": True,
                    "planned_runs": len(specs),
                    "failures": len(payload["failures"]),
                    "aggregate_wall_s": payload["aggregate_mode"]["aggregate_wall_s"],
                },
                sort_keys=True,
            )
        )
        return 0 if not payload["failures"] else 2

    plan_payloads: list[dict[str, Any]] = []
    registry_rows: dict[str, dict[str, dict[str, str]]] = {}
    registries = sorted({case["registry"] for case in explicit_cases}) if explicit_cases else [str(args.registry)]
    planning_failed = False
    for idx, registry in enumerate(registries, start=1):
        suffix = f"{idx:02d}_{_slug(registry, 18)}"
        plan = _plan_registry(
            registry=registry,
            planning_root=planning_root / suffix,
            logs_dir=logs_dir,
            command_log=command_log,
            batch_tag=str(args.batch_tag),
            perf=bool(args.perf),
            knobs=knobs,
            performance_profile=performance_profile,
            campaign_output_profile=campaign_output_profile,
            allow_large_output=bool(getattr(args, "allow_large_output", False)),
            allow_generation_checkpointing=bool(getattr(args, "allow_generation_checkpointing", False)),
            log_suffix=suffix,
        )
        rows = list(plan.pop("_rows", []))
        plan_payloads.append(plan)
        registry_rows[registry] = {str(row.get("run_id") or ""): row for row in rows}
        if int(plan.get("returncode", 1)) != 0 or not plan.get("matrix_path"):
            planning_failed = True
    payload["plan"] = plan_payloads[0] if len(plan_payloads) == 1 else plan_payloads
    if planning_failed:
        payload["status"] = "failed_planning"
        payload["failures"] = [
            {
                "run_id": "",
                "registry": str(plan.get("registry", "")),
                "status": "planning_failed",
                "returncode": int(plan.get("returncode", 1)),
            }
            for plan in plan_payloads
            if int(plan.get("returncode", 1)) != 0 or not plan.get("matrix_path")
        ]
        payload["ended_at_utc"] = _utc_now()
        payload["total_wall_s"] = time.perf_counter() - started
        _write_json(summary_json, payload)
        _write_markdown(summary_md, payload)
        return 2

    if explicit_cases:
        selected_cases = explicit_cases
    else:
        requested = _selected_run_ids(args)
        first_registry = str(args.registry)
        all_rows = list(registry_rows.get(first_registry, {}).values())
        selected_run_ids = requested or [str(row.get("run_id") or "") for row in all_rows]
        selected_cases = [
            {"label": _slug(run_id, 24), "registry": first_registry, "run_id": run_id}
            for run_id in selected_run_ids
            if run_id
        ]

    if explicit_cases:
        missing = [
            f"{case['registry']}::{case['run_id']}"
            for case in selected_cases
            if case["run_id"] not in registry_rows.get(case["registry"], {})
        ]
    else:
        missing = [
            case["run_id"]
            for case in selected_cases
            if case["run_id"] not in registry_rows.get(case["registry"], {})
        ]
    if missing:
        payload["status"] = "failed_planning"
        payload["planned_runs"] = 0
        payload["invalid_run_ids"] = list(missing)
        payload["failures"] = [
            {"run_id": run_id, "status": "invalid_run_id", "returncode": 2} for run_id in missing
        ]
        payload["ended_at_utc"] = _utc_now()
        payload["total_wall_s"] = time.perf_counter() - started
        _write_json(summary_json, payload)
        _write_markdown(summary_md, payload)
        _write_status_csv(status_csv, payload["failures"])
        return 2

    payload["planned_runs"] = len(selected_cases)
    payload["selected_run_ids"] = [case["run_id"] for case in selected_cases]
    payload["selected_cases"] = list(selected_cases)

    if args.dry_run:
        payload["status"] = "dry_run"
        payload["ended_at_utc"] = _utc_now()
        payload["total_wall_s"] = time.perf_counter() - started
        if bool(args.campaign_index):
            payload["campaign_index"]["counters"] = dict(campaign_index_counters)
        _write_json(summary_json, payload)
        _write_markdown(summary_md, payload)
        _write_status_csv(status_csv, [])
        return 0

    specs: list[ChildSpec] = []
    for index, case in enumerate(selected_cases, start=1):
        run_id = case["run_id"]
        run_slug = _slug(case.get("label") or run_id, 36)
        child_root = children_root / f"{index:04d}_{run_slug}"
        if child_root.exists() and any(child_root.iterdir()) and not args.overwrite:
            payload["status"] = "failed_planning"
            payload["failures"].append(
                {"run_id": run_id, "status": "child_root_exists", "returncode": 2, "child_root": str(child_root)}
            )
            continue
        specs.append(
            ChildSpec(
                index=index,
                label=case.get("label") or run_slug,
                run_id=run_id,
                registry=str(case["registry"]),
                child_root=str(child_root),
                batch_tag=f"{args.batch_tag}_{index:04d}_{_slug(case.get('label') or run_id, 18)}",
                knobs=tuple(knobs),
                performance_profile=performance_profile,
                campaign_output_profile=campaign_output_profile,
                allow_large_output=bool(getattr(args, "allow_large_output", False)),
                allow_generation_checkpointing=bool(getattr(args, "allow_generation_checkpointing", False)),
                perf=bool(args.perf),
                overwrite=bool(args.overwrite),
                command_log=str(command_log),
                logs_dir=str(logs_dir),
                campaign_index_entry=existing_campaign_entries.get(run_id),
                worker_slot=(index - 1) % workers,
                affinity_mode=str(affinity_plan.get("mode", "none")),
                affinity_cpu_set=tuple(
                    int(cpu)
                    for cpu in affinity_plan.get("worker_cpu_sets", [[]])[(index - 1) % workers]
                ),
            )
        )
    if payload["failures"]:
        payload["ended_at_utc"] = _utc_now()
        payload["total_wall_s"] = time.perf_counter() - started
        _write_json(summary_json, payload)
        _write_markdown(summary_md, payload)
        _write_status_csv(status_csv, payload["failures"])
        return 2

    _write_live_manifest(
        runs_root / "parallel_live_manifest.json",
        args=args,
        runs_root=runs_root,
        command_log=command_log,
        summary_json=summary_json,
        status_csv=status_csv,
        specs=specs,
    )
    payload["progress_ui"] = _launch_progress_ui(args=args, runs_root=runs_root, command_log=command_log)
    _append_command_log(
        command_log,
        {
            "kind": "progress_ui_launch",
            "timestamp_utc": _utc_now(),
            **payload["progress_ui"],
        },
    )
    payload["status"] = "running"
    if bool(args.campaign_index):
        payload["campaign_index"]["counters"] = dict(campaign_index_counters)
    _write_json(summary_json, payload)
    if payload["progress_ui"].get("attach_command"):
        print(
            json.dumps(
                {
                    "progress_ui": payload["progress_ui"],
                },
                sort_keys=True,
            )
        )

    results: list[dict[str, Any]] = []
    monitor_stop: threading.Event | None = None
    monitor_thread: threading.Thread | None = None
    monitor_rows: list[dict[str, Any]] = []
    if bool(args.capture_monitoring):
        monitor_stop = threading.Event()
        monitor_thread = threading.Thread(
            target=_monitor_loop,
            args=(runs_root, monitor_stop, float(args.monitor_interval_s), monitor_rows),
            daemon=True,
        )
        monitor_thread.start()
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_child, spec): spec for spec in specs}
            for future in as_completed(futures):
                result = future.result()
                if bool(args.campaign_index):
                    for key, value in (result.get("campaign_index_lookup_counters") or {}).items():
                        if key in campaign_index_counters:
                            campaign_index_counters[key] += int(value or 0)
                results.append(result)
                payload["results"] = sorted(results, key=lambda item: int(item.get("index", 0)))
                payload["failures"] = [
                    row for row in payload["results"] if not _result_success(row)
                ]
                if len(payload["results"]) < len(specs):
                    payload["status"] = "running" if not payload["failures"] else "incomplete"
                else:
                    payload["status"] = "complete" if not payload["failures"] else "incomplete"
                payload["ended_at_utc"] = _utc_now()
                payload["total_wall_s"] = time.perf_counter() - started
                if bool(args.campaign_index):
                    campaign_index_payload = _build_campaign_index_payload(
                        runs_root=runs_root,
                        args=args,
                        results=payload["results"],
                        counters=campaign_index_counters,
                    )
                    _write_campaign_index(campaign_index_path, campaign_index_payload)
                    payload["campaign_index"]["counters"] = dict(campaign_index_counters)
                _write_json(summary_json, payload)
                _write_markdown(summary_md, payload)
                _write_status_csv(status_csv, payload["results"])
    finally:
        if monitor_stop is not None:
            monitor_stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=max(2.0, float(args.monitor_interval_s)))
        if monitor_rows:
            payload["monitoring"].extend(monitor_rows)

    if bool(args.capture_monitoring):
        payload["monitoring"].append(_capture_monitoring(runs_root, "after"))

    payload["results"] = sorted(results, key=lambda item: int(item.get("index", 0)))
    payload["failures"] = [
        row for row in payload["results"] if not _result_success(row)
    ]
    payload["status"] = "complete" if not payload["failures"] else "incomplete"
    payload["ended_at_utc"] = _utc_now()
    payload["total_wall_s"] = time.perf_counter() - started
    payload["mean_child_wall_s"] = (
        sum(float(row.get("wall_s", 0.0) or 0.0) for row in payload["results"]) / max(1, len(payload["results"]))
    )
    if bool(args.campaign_index):
        campaign_index_payload = _build_campaign_index_payload(
            runs_root=runs_root,
            args=args,
            results=payload["results"],
            counters=campaign_index_counters,
        )
        _write_campaign_index(campaign_index_path, campaign_index_payload)
        payload["campaign_index"]["counters"] = dict(campaign_index_counters)
    _write_json(summary_json, payload)
    _write_markdown(summary_md, payload)
    _write_status_csv(status_csv, payload["results"])
    print(
        json.dumps(
            {
                "summary": str(summary_json),
                "workers": workers,
                "planned_runs": len(selected_cases),
                "failures": len(payload["failures"]),
                "total_wall_s": payload["total_wall_s"],
            },
            sort_keys=True,
        )
    )
    return 0 if not payload["failures"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
