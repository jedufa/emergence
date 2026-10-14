#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

PROGRESS_RE = re.compile(
    r"\[gen\s+(?P<done>\d+)\s*/\s*(?P<total>\d+)\]\s+"
    r"elapsed\s+(?P<elapsed>-?[0-9:.]+)\s+\|\s+"
    r"avg/gen\s+(?P<avg>-?[0-9.]+)s"
    r"(?:\s+\|\s+ETA\s+(?P<eta>-?[0-9:.]+))?"
)


@dataclass(frozen=True)
class ProgressSample:
    done: int
    total: int
    elapsed_s: float | None
    avg_gen_s: float | None
    eta_s: float | None


@dataclass(frozen=True)
class ChildSnapshot:
    index: int
    worker_slot: int
    label: str
    run_id: str
    status: str
    stdout_path: Path
    stderr_path: Path
    progress: ProgressSample | None
    wall_s: float | None
    returncode: int | None
    last_update_age_s: float | None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _parse_clock_s(text: str | None) -> float | None:
    if text is None:
        return None
    value = str(text).strip()
    if not value or value.startswith("-"):
        return None
    parts = value.split(":")
    try:
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            total = hours * 3600.0 + minutes * 60.0 + seconds
        elif len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            total = minutes * 60.0 + seconds
        elif len(parts) == 1:
            total = float(parts[0])
        else:
            return None
    except ValueError:
        return None
    if not math.isfinite(total) or total < 0:
        return None
    return total


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(float(seconds)) or float(seconds) < 0:
        return "unknown"
    seconds = float(seconds)
    if seconds < 10:
        return f"00:00:{seconds:05.2f}"
    whole = int(round(seconds))
    minutes, sec = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out) or out < 0:
        return None
    return out


def parse_progress_line(line: str) -> ProgressSample | None:
    match = PROGRESS_RE.search(line)
    if not match:
        return None
    done = int(match.group("done"))
    total = int(match.group("total"))
    avg = _safe_float(match.group("avg"))
    eta_text = match.group("eta")
    eta = _parse_clock_s(eta_text)
    if eta_text is None and eta is None and avg is not None:
        eta = max(0.0, (total - done) * avg)
    return ProgressSample(
        done=done,
        total=total,
        elapsed_s=_parse_clock_s(match.group("elapsed")),
        avg_gen_s=avg,
        eta_s=eta,
    )


def latest_progress_from_text(text: str) -> ProgressSample | None:
    sample: ProgressSample | None = None
    for line in text.splitlines():
        parsed = parse_progress_line(line)
        if parsed is not None:
            sample = parsed
    return sample


def _load_command_log(
    path: Path,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    starts: dict[int, dict[str, Any]] = {}
    ends: dict[int, dict[str, Any]] = {}
    for line in _read_text(path).splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        try:
            index = int(row.get("index"))
        except (TypeError, ValueError):
            continue
        kind = str(row.get("kind") or "")
        if kind == "child_start":
            starts[index] = row
        elif kind == "child_end":
            ends[index] = row
    return starts, ends


def _load_status_csv(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except Exception:
        return {}
    out: dict[int, dict[str, str]] = {}
    for row in rows:
        try:
            out[int(row.get("index") or 0)] = row
        except ValueError:
            continue
    return out


def _path_from_child(child: dict[str, Any], key: str) -> Path:
    raw = str(child.get(key) or "").strip()
    return Path(raw) if raw else Path()


def _status_for_child(
    *,
    progress: ProgressSample | None,
    started: bool,
    end: dict[str, Any] | None,
    status_row: dict[str, str] | None,
    stdout_text: str,
    stale_after_s: float,
    last_update_age_s: float | None,
) -> str:
    if end is not None:
        try:
            rc = int(end.get("returncode"))
        except (TypeError, ValueError):
            rc = 1
        if rc != 0:
            return "failed"
        status = str((status_row or {}).get("status") or "ok").strip()
        return "complete" if status in ("", "ok") else status
    if "Run complete." in stdout_text:
        return "complete"
    if progress is not None and progress.total > 0 and progress.done >= progress.total:
        return "finishing"
    if started or progress is not None:
        if last_update_age_s is not None and last_update_age_s > max(
            1.0, stale_after_s
        ):
            return "stale"
        return "running"
    return "queued"


def load_campaign_snapshot(
    runs_root: Path, *, stale_after_s: float = 120.0
) -> dict[str, Any]:
    runs_root = runs_root.expanduser().resolve()
    manifest = _load_json(runs_root / "parallel_live_manifest.json")
    summary = _load_json(runs_root / "parallel_campaign_summary.json")
    command_log = Path(
        str(
            manifest.get("command_log_path")
            or summary.get("command_log_path")
            or runs_root / "parallel_command_log.jsonl"
        )
    )
    status_csv = Path(
        str(manifest.get("status_csv_path") or runs_root / "parallel_child_status.csv")
    )
    starts, ends = _load_command_log(command_log)
    status_rows = _load_status_csv(status_csv)
    children = list(manifest.get("children") or [])

    if not children:
        children = [
            {
                "index": row.get("index"),
                "worker_slot": row.get("worker_slot"),
                "label": row.get("label"),
                "run_id": row.get("run_id"),
                "stdout_path": row.get("stdout_path"),
                "stderr_path": row.get("stderr_path"),
            }
            for row in sorted(
                status_rows.values(), key=lambda item: int(item.get("index") or 0)
            )
        ]
    if not children:
        command_rows = {**starts, **ends}
        children = [
            {
                "index": row.get("index"),
                "worker_slot": row.get("worker_slot"),
                "label": row.get("label"),
                "run_id": row.get("run_id"),
                "stdout_path": row.get("stdout_path"),
                "stderr_path": row.get("stderr_path"),
            }
            for row in sorted(
                command_rows.values(), key=lambda item: int(item.get("index") or 0)
            )
        ]

    now = time.time()
    snapshots: list[ChildSnapshot] = []
    for child in children:
        try:
            index = int(child.get("index") or 0)
        except (TypeError, ValueError):
            continue
        try:
            worker_slot = int(child.get("worker_slot") or 0)
        except (TypeError, ValueError):
            worker_slot = 0
        stdout_path = _path_from_child(child, "stdout_path")
        stderr_path = _path_from_child(child, "stderr_path")
        stdout_text = _read_text(stdout_path)
        progress = latest_progress_from_text(stdout_text)
        last_age: float | None = None
        if stdout_path.exists():
            try:
                last_age = max(0.0, now - stdout_path.stat().st_mtime)
            except OSError:
                last_age = None
        end = ends.get(index)
        row = status_rows.get(index)
        status = _status_for_child(
            progress=progress,
            started=index in starts,
            end=end,
            status_row=row,
            stdout_text=stdout_text,
            stale_after_s=stale_after_s,
            last_update_age_s=last_age,
        )
        wall_s = _safe_float((end or row or {}).get("wall_s"))
        returncode: int | None = None
        raw_rc = (end or row or {}).get("returncode")
        if raw_rc not in (None, ""):
            try:
                returncode = int(raw_rc)
            except (TypeError, ValueError):
                returncode = None
        snapshots.append(
            ChildSnapshot(
                index=index,
                worker_slot=worker_slot,
                label=str(child.get("label") or ""),
                run_id=str(child.get("run_id") or ""),
                status=status,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                progress=progress,
                wall_s=wall_s,
                returncode=returncode,
                last_update_age_s=last_age,
            )
        )

    planned = int(
        manifest.get("planned_runs") or summary.get("planned_runs") or len(snapshots)
    )
    workers = int(
        manifest.get("workers")
        or summary.get("workers")
        or max([s.worker_slot for s in snapshots], default=-1) + 1
        or 1
    )
    return {
        "runs_root": str(runs_root),
        "manifest": manifest,
        "summary": summary,
        "children": sorted(snapshots, key=lambda item: item.index),
        "planned_runs": planned,
        "workers": max(1, workers),
        "overall_eta_s": _overall_eta_s(
            snapshots, planned=planned, workers=max(1, workers)
        ),
    }


def _active_remaining_s(child: ChildSnapshot) -> float | None:
    if child.status not in {"running", "stale", "finishing"}:
        return 0.0 if child.status == "complete" else None
    if child.progress is None:
        return None
    if child.progress.eta_s is not None:
        return child.progress.eta_s
    if child.progress.avg_gen_s is None:
        return None
    return max(
        0.0, (child.progress.total - child.progress.done) * child.progress.avg_gen_s
    )


def _overall_eta_s(
    children: list[ChildSnapshot], *, planned: int, workers: int
) -> float | None:
    completed_walls = [
        child.wall_s
        for child in children
        if child.status == "complete" and child.wall_s is not None
    ]
    active_remaining = [
        value
        for child in children
        for value in [_active_remaining_s(child)]
        if value is not None and child.status in {"running", "stale", "finishing"}
    ]
    active_or_terminal = {"complete", "failed", "running", "stale", "finishing"}
    queued = max(
        0, planned - sum(1 for child in children if child.status in active_or_terminal)
    )
    if completed_walls:
        queued_unit = sum(completed_walls) / len(completed_walls)
    else:
        projected_active_totals = [
            (child.progress.elapsed_s or 0.0) + remaining
            for child in children
            for remaining in [_active_remaining_s(child)]
            if child.progress is not None
            and remaining is not None
            and child.status in {"running", "stale", "finishing"}
        ]
        queued_unit = (
            (sum(projected_active_totals) / len(projected_active_totals))
            if projected_active_totals
            else None
        )
    if not active_remaining and (not queued or queued_unit is None):
        return (
            0.0
            if planned
            and sum(1 for child in children if child.status == "complete") >= planned
            else None
        )
    active_slots = max(
        1,
        min(
            workers,
            max(
                1,
                sum(
                    1
                    for child in children
                    if child.status in {"running", "stale", "finishing"}
                ),
            ),
        ),
    )
    queued_total = queued * queued_unit if queued_unit is not None else 0.0
    return (sum(active_remaining) + queued_total) / active_slots


def _bar(done: int, total: int, *, width: int = 24) -> str:
    if total <= 0:
        return "[" + " " * width + "]"
    ratio = max(0.0, min(1.0, float(done) / float(total)))
    filled = int(round(width * ratio))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _ste_dashboard_text(summary: Mapping[str, Any]) -> str | None:
    ste = summary.get("scientific_throughput_efficiency")
    if not isinstance(ste, Mapping):
        return None
    dashboard = ste.get("dashboard_payload")
    if not isinstance(dashboard, Mapping):
        dashboard = {}
    cards = dashboard.get("cards")
    if not isinstance(cards, Mapping):
        cards = {}
    resource = cards.get("resource_efficiency")
    primary = cards.get("primary_science_speed")
    scientific = cards.get("scientific_throughput") or cards.get(
        "scientific_efficiency"
    )
    qualification = cards.get("qualification_throughput")
    discovery = cards.get("scientific_discovery_efficiency")
    resource = resource if isinstance(resource, Mapping) else {}
    primary = primary if isinstance(primary, Mapping) else {}
    scientific = scientific if isinstance(scientific, Mapping) else {}
    qualification = qualification if isinstance(qualification, Mapping) else {}
    discovery = discovery if isinstance(discovery, Mapping) else {}
    worlds_rate = (
        scientific.get("worlds_completed_per_second")
        if scientific.get("worlds_completed_per_second") is not None
        else scientific.get("mean_worlds_completed_per_second")
    )
    primary_rate = primary.get("primary_world_generations_per_hour")
    qualification_rate = qualification.get("qualification_generations_per_hour")
    return (
        "STE | resource occupancy {occupancy} | primary gen/h {primary_rate} | "
        "scientific worlds/sec {worlds_rate} | qualification gen/h {qualification_rate} | "
        "evidence packages {evidence}"
    ).format(
        occupancy=resource.get("hardware_occupancy", "unknown"),
        primary_rate=primary_rate if primary_rate is not None else "unknown",
        worlds_rate=worlds_rate if worlds_rate is not None else "unknown",
        qualification_rate=(
            qualification_rate if qualification_rate is not None else "unknown"
        ),
        evidence=discovery.get("evidence_packages", "unknown"),
    )


def render_plain(snapshot: dict[str, Any]) -> str:
    children: list[ChildSnapshot] = snapshot["children"]
    planned = int(snapshot["planned_runs"])
    counts = {
        status: sum(1 for child in children if child.status == status)
        for status in sorted({c.status for c in children})
    }
    completed = sum(1 for child in children if child.status == "complete")
    failed = sum(1 for child in children if child.status == "failed")
    header = (
        f"EIEL campaign {completed}/{planned} complete | "
        f"running {counts.get('running', 0) + counts.get('stale', 0) + counts.get('finishing', 0)} | "
        f"queued {counts.get('queued', 0)} | failed {failed} | "
        f"ETA {_fmt_duration(snapshot.get('overall_eta_s'))}"
    )
    lines = [header, f"runs_root: {snapshot['runs_root']}"]
    ste_line = _ste_dashboard_text(dict(snapshot.get("summary") or {}))
    if ste_line is not None:
        lines.append(ste_line)
    for child in sorted(children, key=lambda item: (item.worker_slot, item.index)):
        progress = child.progress
        done = progress.done if progress else 0
        total = progress.total if progress else 0
        label = child.run_id or child.label or f"child_{child.index:04d}"
        lines.append(
            "w{worker:02d} #{index:04d} {status:<9} {bar} gen {done:>3}/{total:<3} "
            "elapsed {elapsed:<11} avg/gen {avg:<8} ETA {eta:<11} {label}".format(
                worker=child.worker_slot,
                index=child.index,
                status=child.status,
                bar=_bar(done, total),
                done=done,
                total=total,
                elapsed=_fmt_duration(progress.elapsed_s if progress else None),
                avg=(
                    f"{progress.avg_gen_s:.2f}s"
                    if progress and progress.avg_gen_s is not None
                    else "unknown"
                ),
                eta=_fmt_duration(progress.eta_s if progress else None),
                label=label[:72],
            )
        )
    return "\n".join(lines)


def render_rich(snapshot: dict[str, Any]):
    from rich import box
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    children: list[ChildSnapshot] = snapshot["children"]
    planned = int(snapshot["planned_runs"])
    completed = sum(1 for child in children if child.status == "complete")
    failed = sum(1 for child in children if child.status == "failed")
    running = sum(
        1 for child in children if child.status in {"running", "stale", "finishing"}
    )
    queued = sum(1 for child in children if child.status == "queued")
    table = Table(box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("Worker", justify="right", no_wrap=True)
    table.add_column("Run")
    table.add_column("Status", no_wrap=True)
    table.add_column("Progress", no_wrap=True)
    table.add_column("Gen", justify="right", no_wrap=True)
    table.add_column("Elapsed", justify="right", no_wrap=True)
    table.add_column("Avg/gen", justify="right", no_wrap=True)
    table.add_column("ETA", justify="right", no_wrap=True)
    status_styles = {
        "complete": "green",
        "failed": "red",
        "running": "cyan",
        "stale": "yellow",
        "finishing": "blue",
        "queued": "dim",
    }
    for child in sorted(children, key=lambda item: (item.worker_slot, item.index)):
        progress = child.progress
        done = progress.done if progress else 0
        total = progress.total if progress else 0
        style = status_styles.get(child.status, "")
        table.add_row(
            f"{child.worker_slot:02d}",
            (child.run_id or child.label or f"child_{child.index:04d}")[:48],
            Text(child.status, style=style),
            _bar(done, total),
            f"{done}/{total}" if total else "-",
            _fmt_duration(progress.elapsed_s if progress else None),
            (
                f"{progress.avg_gen_s:.2f}s"
                if progress and progress.avg_gen_s is not None
                else "unknown"
            ),
            _fmt_duration(progress.eta_s if progress else None),
        )
    ste_line = _ste_dashboard_text(dict(snapshot.get("summary") or {}))
    if ste_line is not None:
        table.caption = ste_line
    title = (
        f"EIEL campaign | {completed}/{planned} complete | running {running} | "
        f"queued {queued} | failed {failed} | ETA {_fmt_duration(snapshot.get('overall_eta_s'))}"
    )
    return Panel(table, title=title, subtitle=str(snapshot["runs_root"]), expand=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor an EIEL parallel campaign in the terminal"
    )
    parser.add_argument(
        "--runs-root", required=True, help="Parallel campaign runtime root"
    )
    parser.add_argument(
        "--refresh-s", type=float, default=1.0, help="Refresh interval for live mode"
    )
    parser.add_argument(
        "--stale-after-s",
        type=float,
        default=120.0,
        help="Mark running rows stale after no stdout updates",
    )
    parser.add_argument(
        "--once", action="store_true", help="Print one snapshot and exit"
    )
    parser.add_argument("--no-color", action="store_true", help="Use plain text output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    runs_root = Path(args.runs_root)
    if args.once or args.no_color:
        print(
            render_plain(
                load_campaign_snapshot(
                    runs_root, stale_after_s=float(args.stale_after_s)
                )
            )
        )
        return 0
    try:
        from rich.console import Console
        from rich.live import Live
    except Exception as exc:
        print(
            f"Rich is unavailable ({exc}); use --once or --no-color for plain output."
        )
        return 2
    console = Console()
    with Live(
        console=console,
        refresh_per_second=max(1.0, 1.0 / max(0.1, float(args.refresh_s))),
    ) as live:
        while True:
            live.update(
                render_rich(
                    load_campaign_snapshot(
                        runs_root, stale_after_s=float(args.stale_after_s)
                    )
                )
            )
            time.sleep(max(0.1, float(args.refresh_s)))


if __name__ == "__main__":
    raise SystemExit(main())
