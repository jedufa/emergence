"""Standalone Emergence WebUI server.

This module intentionally serves only the Emergence-facing WebUI surface.  It
does not import the historical Causyl lab-assistant, scheduler, mobile, ledger,
or patch-admin control plane.
"""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

import argparse
import csv
import datetime as dt
import gzip
import html
import json
import mimetypes
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
import posixpath
import re
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Iterable
from urllib.parse import parse_qs, unquote, urlparse

from infra.webui import server_chronicle
from infra.webui import canopy_library
from infra.webui import server_live_simulations


REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_ROOT = Path(__file__).resolve().parent / "static"
DEFAULT_RUNTIME_ROOT = Path(os.environ.get("EIEL_RUNTIME_ROOT") or public_runtime_path(''))
LEGACY_RUNTIME_ROOT = Path(public_runtime_path(''))
COLD_STORAGE_ROOT = Path("/mnt/Projects/_Emergence-cold-storage")
COLD_CHRONICLE_ARCHIVE_ROOT = (
    COLD_STORAGE_ROOT
    / "runtime_cleanup_20260801_non_option_d"
    / "data"
    / "chronicle_history"
)
DEFAULT_RUNS_ROOT = Path(os.environ.get("EIEL_RUNS_ROOT") or DEFAULT_RUNTIME_ROOT / "live")
DEFAULT_SLOT_ROOT = Path(os.environ.get("EIEL_SLOT_TELEMETRY_ROOT") or DEFAULT_RUNTIME_ROOT / "live" / "simulation-slots")
DEFAULT_ARCHIVE_ROOT = Path(os.environ.get("EIEL_CHRONICLE_ARCHIVE_ROOT") or DEFAULT_RUNTIME_ROOT / "chronicle_history")
MAX_TEXT_TAIL_BYTES = 1_000_000
RUN_MARKER_FILES = (
    "run_status.json",
    "status.json",
    "run_meta.json",
    "meta.json",
    "manifest.json",
    "summary.json",
    "metrics.csv",
    "run_config.csv",
    "stdout.log",
)
BLOCKED_STATIC_NAMES = {
    "agents.html",
    "agents.js",
    "mobile.html",
    "mobile.js",
    "mobile_backlog.html",
    "mobile_backlog.js",
    "mobile_update.html",
    "schedules.html",
    "schedules.js",
}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        if path.suffix == ".gz":
            raw = gzip.decompress(path.read_bytes()).decode("utf-8", errors="replace")
        else:
            raw = path.read_text(encoding="utf-8")
        value = json.loads(raw)
    except Exception:
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _read_text_tail(path: Path, max_bytes: int = 65536) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(max(0, size - max_bytes))
            return handle.read(max_bytes).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except Exception:
        return False
    return True


def _mime(path: Path) -> str:
    return mimetypes.guess_type(str(path))[0] or "application/octet-stream"


def _first_number(*values: Any) -> int | None:
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            number = int(float(str(value)))
        except Exception:
            continue
        return number
    return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            number = float(str(value))
        except Exception:
            continue
        if number == number and abs(number) != float("inf"):
            return number
    return None


def _status_from_files(run_dir: Path, docs: list[dict[str, Any]]) -> str:
    for doc in docs:
        for key in ("status", "state", "lifecycle", "run_status"):
            value = str(doc.get(key) or "").strip().upper()
            if value:
                if value in {"SUCCEEDED", "SUCCESS", "DONE", "COMPLETED"}:
                    return "COMPLETE"
                if value in {"ERROR"}:
                    return "FAILED"
                return value
    if (run_dir / "COMPLETE").exists() or (run_dir / "complete").exists():
        return "COMPLETE"
    if (run_dir / "FAILED").exists() or (run_dir / "failed").exists():
        return "FAILED"
    if any((run_dir / name).is_file() for name in ("stdout.log", "stderr.log", "run.log")):
        return "UNKNOWN"
    return "UNKNOWN"


def _metrics_path(run_dir: Path) -> Path | None:
    for rel in ("metrics.csv", "payload/metrics.csv", "metrics/metrics.csv"):
        path = run_dir / rel
        if path.is_file():
            return path
    return None


def _csv_tail_row(path: Path) -> tuple[list[str], dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            row: dict[str, str] = {}
            for row in reader:
                pass
            return list(reader.fieldnames or []), dict(row or {})
    except Exception:
        return [], {}


def _run_json_docs(run_dir: Path) -> list[dict[str, Any]]:
    docs = []
    for rel in (
        "run_status.json",
        "status.json",
        "run_meta.json",
        "meta.json",
        "manifest.json",
        "summary.json",
        "payload/run_status.json",
        "payload/status.json",
        "payload/manifest.json",
        "payload/summary.json",
    ):
        path = run_dir / rel
        if path.is_file():
            docs.append(_read_json(path))
    return docs


def _run_key(root: Path, run_dir: Path) -> str:
    try:
        return run_dir.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return run_dir.name


def _run_row(root: Path, run_dir: Path) -> dict[str, Any]:
    docs = _run_json_docs(run_dir)
    merged: dict[str, Any] = {}
    for doc in docs:
        merged.update(doc)
    header, last_metrics = _csv_tail_row(_metrics_path(run_dir) or run_dir / "__missing__")
    rk = _run_key(root, run_dir)
    current_gen = _first_number(
        merged.get("current_generation"),
        merged.get("generation"),
        merged.get("completed_generation"),
        last_metrics.get("generation"),
        last_metrics.get("gen"),
    )
    target_gen = _first_number(
        merged.get("target_generation"),
        merged.get("generations"),
        merged.get("max_generation"),
        merged.get("n_generations"),
    )
    epi = _first_float(
        merged.get("epi"),
        merged.get("EPI"),
        last_metrics.get("epi"),
        last_metrics.get("EPI"),
        last_metrics.get("evolvability_pressure_index"),
    )
    av = _first_float(
        merged.get("av"),
        merged.get("AV"),
        last_metrics.get("av"),
        last_metrics.get("AV"),
        last_metrics.get("adaptive_value"),
    )
    tags = merged.get("tags")
    label = str(
        merged.get("label")
        or merged.get("run_label")
        or merged.get("campaign_id")
        or merged.get("scenario_id")
        or ""
    )
    return {
        "run_key": rk,
        "run_id": str(merged.get("run_id") or run_dir.name),
        "display_id": str(merged.get("display_id") or merged.get("run_id") or run_dir.name),
        "label": label,
        "status": _status_from_files(run_dir, docs),
        "determinism": str(merged.get("determinism") or merged.get("determinism_status") or "UNKNOWN").upper(),
        "strict_mode": bool(merged.get("strict_mode") or merged.get("strict")),
        "last_verified_generation": _first_number(merged.get("last_verified_generation"), merged.get("verified_generation")),
        "current_generation": current_gen,
        "target_generation": target_gen,
        "epi": epi,
        "av": av,
        "tags": tags if isinstance(tags, list) else [],
        "parent_run_key": str(merged.get("parent_run_key") or ""),
        "lineage": merged.get("lineage") if isinstance(merged.get("lineage"), dict) else {},
        "notes_excerpt": str(merged.get("notes_excerpt") or ""),
        "run_root": str(run_dir),
        "modified_utc": dt.datetime.fromtimestamp(run_dir.stat().st_mtime, dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "metrics_header": header,
    }


def _looks_like_run_dir(path: Path) -> bool:
    if any((path / marker).is_file() for marker in RUN_MARKER_FILES):
        return True
    if (path / "chronicle").is_dir():
        return True
    if (path / "payload" / "metrics.csv").is_file():
        return True
    return False


def discover_runs(root: Path, scan_depth: int) -> list[dict[str, Any]]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        path, depth = stack.pop()
        try:
            if path != root and _looks_like_run_dir(path):
                rows.append(_run_row(root, path))
                continue
            if depth >= scan_depth:
                continue
            with os.scandir(path) as entries:
                for entry in entries:
                    if entry.name.startswith("."):
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        stack.append((Path(entry.path), depth + 1))
        except Exception:
            continue
    rows.sort(key=lambda row: str(row.get("modified_utc") or ""), reverse=True)
    return rows


def _chronicle_run_row(row: dict[str, Any]) -> dict[str, Any]:
    run_key = str(row.get("run_key") or "")
    generation_end = _first_number(row.get("generation_end"))
    status = str(row.get("status") or "UNKNOWN").upper()
    if status == "READY":
        status = "COMPLETE" if row.get("finalized") else "UNKNOWN"
    return {
        "run_key": run_key,
        "run_id": str(row.get("run_id") or row.get("archive_run_id") or run_key),
        "display_id": str(row.get("run_id") or row.get("world_id") or row.get("archive_run_id") or run_key),
        "label": str(row.get("campaign_id") or row.get("profile") or ""),
        "status": status,
        "determinism": "UNKNOWN",
        "strict_mode": False,
        "last_verified_generation": None,
        "current_generation": generation_end,
        "target_generation": generation_end,
        "epi": None,
        "av": None,
        "tags": ["chronicle"],
        "parent_run_key": "",
        "lineage": {},
        "notes_excerpt": str(row.get("source_completeness") or ""),
        "run_root": str(row.get("run_root") or ""),
        "source_kind": "chronicle",
        "archived": str(row.get("storage_kind") or "") == "zip",
        "modified_utc": _utc_now(),
    }


def webui_runs(cfg: SimpleNamespace) -> list[dict[str, Any]]:
    rows = discover_runs(cfg.runs_root, cfg.scan_depth)
    by_key = {str(row.get("run_key") or ""): row for row in rows}
    try:
        for chronicle_row in server_chronicle.discover_runs(cfg):
            if str(chronicle_row.get("status") or "") != "ready":
                continue
            projected = _chronicle_run_row(dict(chronicle_row))
            key = str(projected.get("run_key") or "")
            if key and key not in by_key:
                by_key[key] = projected
    except Exception:
        pass
    out = list(by_key.values())
    out.sort(key=lambda row: (str(row.get("status") or "") != "RUNNING", str(row.get("modified_utc") or "")), reverse=True)
    return out


def _resolve_run_path(cfg: SimpleNamespace, run_key: str) -> Path:
    key = PurePosixPath(unquote(str(run_key or "")).lstrip("/"))
    if ".." in key.parts:
        raise ValueError("run key may not contain '..'")
    candidate = (cfg.runs_root / Path(*key.parts)).resolve()
    if not _inside(candidate, cfg.runs_root):
        raise ValueError("run key escapes runs root")
    return candidate


def _run_points(run_dir: Path, *, metric: str | None = None, after_gen: int = -1, limit: int = 1000) -> tuple[list[str], list[dict[str, Any]]]:
    path = _metrics_path(run_dir)
    if path is None:
        return [], []
    points: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or [])
            for row in reader:
                gen = _first_number(row.get("generation"), row.get("gen"))
                if gen is None or gen <= after_gen:
                    continue
                rec: dict[str, Any] = {"generation": gen}
                if metric:
                    value = _first_float(row.get(metric))
                    if value is not None:
                        rec["value"] = value
                else:
                    epi = _first_float(row.get("epi"), row.get("EPI"), row.get("evolvability_pressure_index"))
                    av = _first_float(row.get("av"), row.get("AV"), row.get("adaptive_value"))
                    if epi is not None:
                        rec["epi"] = epi
                    if av is not None:
                        rec["av"] = av
                if len(rec) > 1:
                    points.append(rec)
        return header, points[-max(1, min(limit, 10000)) :]
    except Exception:
        return [], []


def _builder_harnesses() -> list[dict[str, Any]]:
    roots = [
        REPO_ROOT / "experiments" / "harnesses" / "generated",
        REPO_ROOT / "_spec" / "registry" / "v1" / "generated",
    ]
    rows: list[dict[str, Any]] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py" if root.name == "generated" and root.parent.name == "harnesses" else "*.yaml")):
            rel = path.relative_to(REPO_ROOT).as_posix()
            module = ""
            if path.suffix == ".py":
                module = rel[:-3].replace("/", ".")
            rows.append(
                {
                    "name": path.stem,
                    "module": module,
                    "path": rel,
                    "registry_path": rel if path.suffix in {".yaml", ".yml"} else "",
                    "description": "",
                    "modified_utc": dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).isoformat().replace("+00:00", "Z"),
                }
            )
    return rows[:200]


def _builder_registries() -> list[dict[str, Any]]:
    root = REPO_ROOT / "_spec" / "registry"
    if not root.is_dir():
        return []
    rows = []
    for path in sorted(root.rglob("*.yaml")) + sorted(root.rglob("*.yml")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        rows.append({"path": rel, "name": path.stem, "description": "", "modified_utc": dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).isoformat().replace("+00:00", "Z")})
    return rows[:500]


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


class EmergenceWebUIHandler(BaseHTTPRequestHandler):
    server_version = "EmergenceWebUI/1.0"

    @property
    def cfg(self) -> SimpleNamespace:
        return self.server.cfg  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (_utc_now(), fmt % args))

    def _send(self, status: HTTPStatus | int, body: bytes, content_type: str, *, headers: dict[str, str] | None = None) -> None:
        self.send_response(int(status))
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store" if content_type.startswith("application/json") else "public, max-age=60")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: HTTPStatus | int, payload: Any) -> None:
        self._send(status, _json_bytes(payload), "application/json; charset=utf-8")

    def _serve_file(self, path: Path) -> None:
        resolved = path.resolve()
        if path.name in BLOCKED_STATIC_NAMES:
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain; charset=utf-8")
            return
        allowed_roots = [STATIC_ROOT.resolve(), (REPO_ROOT / "docs").resolve()]
        if not any(_inside(resolved, root) for root in allowed_roots) or not resolved.is_file():
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain; charset=utf-8")
            return
        self._send(HTTPStatus.OK, resolved.read_bytes(), _mime(resolved))

    def _serve_docs(self, prefix: str, path: str) -> bool:
        if path == prefix.rstrip("/"):
            self.send_response(HTTPStatus.MOVED_PERMANENTLY)
            self.send_header("Location", prefix)
            self.end_headers()
            return True
        if not path.startswith(prefix):
            return False
        rel = unquote(path[len(prefix) :])
        doc_root = (REPO_ROOT / "docs").resolve()
        candidate = (doc_root / rel).resolve()
        if not _inside(candidate, doc_root):
            self._send(HTTPStatus.BAD_REQUEST, b"bad docs path", "text/plain; charset=utf-8")
            return True
        if candidate.is_dir():
            index = candidate / "index.html"
            if index.is_file():
                self._serve_file(index)
                return True
            rows = []
            if rel:
                rows.append('<li><a href="../">..</a></li>')
            for child in sorted(candidate.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                href = posixpath.join(path.rstrip("/") + "/", child.name + ("/" if child.is_dir() else ""))
                rows.append(f'<li><a href="{html.escape(href)}">{html.escape(child.name)}{" /" if child.is_dir() else ""}</a></li>')
            body = f"<!doctype html><title>Emergence Docs</title><h1>Emergence Docs</h1><ul>{''.join(rows)}</ul>".encode("utf-8")
            self._send(HTTPStatus.OK, body, "text/html; charset=utf-8")
            return True
        if candidate.is_file():
            self._serve_file(candidate)
            return True
        self._send(HTTPStatus.NOT_FOUND, b"docs not found", "text/plain; charset=utf-8")
        return True

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path or "/"
        qs = parse_qs(parsed.query or "")
        if path in {"/", "/index.html"}:
            self._serve_file(STATIC_ROOT / "index.html")
            return
        if path.startswith("/static/"):
            self._serve_file(STATIC_ROOT / unquote(path[len("/static/") :]))
            return
        if path in {"/chronicle", "/chronicle/"}:
            self._serve_file(STATIC_ROOT / "chronicle.html")
            return
        if self._serve_docs("/docs/", path):
            return
        if self._serve_docs("/projects/emergence/docs/", path):
            return
        if path == "/canopy":
            location = "/canopy/" + (f"?{parsed.query}" if parsed.query else "")
            self.send_response(HTTPStatus.MOVED_PERMANENTLY)
            self.send_header("Location", location)
            self.end_headers()
            return
        if path.startswith("/living-lineage-canopy") or path.startswith("/canopy"):
            try:
                canopy_path = path
                if path.startswith("/canopy"):
                    canopy_path = "/living-lineage-canopy" + path.removeprefix("/canopy")
                body, content_type = server_live_simulations.canopy_asset(canopy_path)
                self._send(HTTPStatus.OK, body, content_type)
            except Exception as exc:
                self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": str(exc)})
            return
        if path == "/api/health":
            self._json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "utc": _utc_now(),
                    "entity": "emergence",
                    "repo_root": str(REPO_ROOT),
                    "runs_root": str(self.cfg.runs_root),
                    "runtime_root": str(self.cfg.runtime_root),
                },
            )
            return
        if path == "/api/runs":
            self._json(HTTPStatus.OK, {"schema_version": "emergence_webui_runs_v1", "runs": webui_runs(self.cfg)})
            return
        if path == "/api/canopy/playbacks":
            self._json(
                HTTPStatus.OK,
                {
                    "schema_version": "eiel_canopy_playback_library_v1",
                    "playbacks": canopy_library.discover(self.cfg.canopy_library_roots),
                },
            )
            return
        if path == "/api/canopy/playback":
            try:
                body, content_type = canopy_library.artifact(
                    self.cfg.canopy_library_roots,
                    key=str((qs.get("key") or [""])[0]),
                    name=str((qs.get("artifact") or [""])[0]),
                )
                self._send(HTTPStatus.OK, body, content_type)
            except canopy_library.CanopyLibraryError as exc:
                self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": str(exc)})
            return
        if path == "/api/changesets":
            self._json(HTTPStatus.OK, {"schema_version": "emergence_webui_changesets_v1", "changesets": []})
            return
        if path == "/api/system/runs":
            self._json(HTTPStatus.OK, {"schema_version": "emergence_webui_system_runs_v1", "runs": webui_runs(self.cfg)})
            return
        if path == "/api/system/processes":
            self._json(HTTPStatus.OK, {"schema_version": "emergence_webui_processes_v1", "processes": []})
            return
        if path == "/api/builder/config":
            self._json(
                HTTPStatus.OK,
                {
                    "schema_version": "builder_config_v1",
                    "git_head": _git_head(),
                    "entrypoints": [
                        {"label": "Registry case", "module": "_runner.run_registry_case"},
                        {"label": "Registry batch", "module": "_runner.run_registry_batch"},
                    ],
                    "knob_keys": [],
                    "builder": {
                        "runs_root_default": str(self.cfg.runs_root),
                        "harness_module_prefix": "experiments.harnesses.generated",
                        "run_job_module": "tools.manifest.run_job",
                        "manifest_out_dir": "_artifacts/manifests/generated",
                    },
                },
            )
            return
        if path == "/api/builder/registries":
            self._json(HTTPStatus.OK, {"schema_version": "builder_registries_v1", "registries": _builder_registries()})
            return
        if path == "/api/builder/templates":
            self._json(HTTPStatus.OK, {"schema_version": "builder_templates_list_v1", "templates": []})
            return
        if path == "/api/builder/harnesses":
            self._json(HTTPStatus.OK, {"schema_version": "builder_harnesses_v1", "harnesses": _builder_harnesses(), "recent_jobs": []})
            return
        if path == "/api/builder/job_status":
            self._json(HTTPStatus.NOT_FOUND, {"schema_version": "builder_job_status_v1", "terminal": True, "lifecycle": "UNKNOWN", "error": "job not found"})
            return
        if path == "/api/builder/job_log":
            self._json(HTTPStatus.OK, {"schema_version": "builder_job_log_v1", "text": ""})
            return
        if path.startswith("/api/chronicle/"):
            status, payload = server_chronicle.handle_get_request(cfg=self.cfg, path=path, qs=qs)
            self._json(status, payload)
            return
        if path == "/api/emergence/live_simulations":
            self._json(HTTPStatus.OK, server_live_simulations.snapshot(root=self.cfg.slot_telemetry_root))
            return
        if path == "/api/emergence/live_simulations/chronicle":
            try:
                payload = server_live_simulations.chronicle_snapshot(
                    cfg=self.cfg,
                    device=str((qs.get("device") or [""])[0]),
                    position=int((qs.get("position") or ["0"])[0]),
                    root=self.cfg.slot_telemetry_root,
                )
                self._json(HTTPStatus.OK, payload)
            except Exception as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return
        if path.startswith("/api/emergence/live_simulations/observatory"):
            status, payload, content_type = server_live_simulations.observatory_response(path=path, root=self.cfg.slot_telemetry_root)
            if isinstance(payload, bytes):
                self._send(status, payload, content_type)
            else:
                self._json(status, payload)
            return
        if path == "/api/emergence/pressure_search/campaigns":
            from tools import pressure_search

            self._json(HTTPStatus.OK, {"schema_version": "emergence_pressure_search_campaigns_v1", "utc": _utc_now(), "campaigns": pressure_search.list_campaigns()})
            return
        if path == "/api/emergence/pressure_search/campaign":
            from tools import pressure_search

            self._json(HTTPStatus.OK, pressure_search.get_campaign(str((qs.get("campaign_id") or [""])[0])))
            return
        if path == "/api/run/file_tail":
            run_dir = _resolve_run_path(self.cfg, str((qs.get("key") or [""])[0]))
            which = str((qs.get("which") or ["stdout"])[0]).lower()
            name = {"stdout": "stdout.log", "stderr": "stderr.log", "log": "run.log"}.get(which, "stdout.log")
            max_bytes = min(MAX_TEXT_TAIL_BYTES, max(1, int(str((qs.get("max_bytes") or ["65536"])[0] or "65536"))))
            self._json(HTTPStatus.OK, {"schema_version": "run_file_tail_v1", "text": _read_text_tail(run_dir / name, max_bytes=max_bytes)})
            return
        if path.startswith("/api/run/"):
            self._handle_run_get(path, qs)
            return
        self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain; charset=utf-8")

    def _handle_run_get(self, path: str, qs: dict[str, list[str]]) -> None:
        parts = [unquote(p) for p in path[len("/api/run/") :].split("/")]
        if len(parts) < 2:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "missing run endpoint"})
            return
        endpoint = parts[-1]
        run_key = "/".join(parts[:-1])
        run_dir = _resolve_run_path(self.cfg, run_key)
        if endpoint == "meta":
            self._json(HTTPStatus.OK, {"schema_version": "run_meta_v1", "run": _run_row(self.cfg.runs_root, run_dir), "run_root": str(run_dir), "meta": _run_json_docs(run_dir)})
            return
        if endpoint == "checkpoints":
            rows = []
            for pattern in ("**/*checkpoint*", "**/*.ckpt", "**/*.pt", "**/*.pth"):
                for path_obj in run_dir.glob(pattern):
                    if path_obj.is_file():
                        rows.append({"path": path_obj.relative_to(run_dir).as_posix(), "bytes": path_obj.stat().st_size})
            self._json(HTTPStatus.OK, {"schema_version": "run_checkpoints_v1", "checkpoints": rows[:500]})
            return
        if endpoint == "epiav":
            after = int(str((qs.get("after_gen") or ["-1"])[0] or "-1"))
            limit = int(str((qs.get("limit") or ["1000"])[0] or "1000"))
            header, points = _run_points(run_dir, after_gen=after, limit=limit)
            self._json(HTTPStatus.OK, {"schema_version": "run_epiav_v1", "source": str(_metrics_path(run_dir) or ""), "source_kind": "metrics_csv", "header": header, "points": points, "has_epi": any("epi" in p for p in points), "has_av": any("av" in p for p in points), "provisional": False, "parse_errors": 0})
            return
        if endpoint == "metrics_schema":
            header, _row = _csv_tail_row(_metrics_path(run_dir) or run_dir / "__missing__")
            metrics = [name for name in header if name not in {"generation", "gen"}]
            self._json(HTTPStatus.OK, {"schema_version": "run_metrics_schema_v1", "source": str(_metrics_path(run_dir) or ""), "metrics": metrics})
            return
        if endpoint == "metrics_series":
            metric = str((qs.get("metric") or [""])[0])
            after = int(str((qs.get("after_gen") or ["-1"])[0] or "-1"))
            limit = int(str((qs.get("limit") or ["1000"])[0] or "1000"))
            _header, points = _run_points(run_dir, metric=metric, after_gen=after, limit=limit)
            latest = points[-1] if points else None
            self._json(HTTPStatus.OK, {"schema_version": "run_metric_series_v1", "metric": metric, "source": str(_metrics_path(run_dir) or ""), "points": points, "latest": latest, "provisional": False, "parse_errors": 0})
            return
        if endpoint in {"applied_commands", "autopilot_actions", "inbox", "notes", "timeline_summary", "timeline_range", "traits", "knobs"}:
            empty: dict[str, Any] = {
                "applied_commands": {"commands": [], "items": [], "window": {}},
                "autopilot_actions": {"actions": [], "items": [], "window": {}},
                "inbox": {"items": [], "commands": []},
                "notes": {"text": "", "updated_at": ""},
                "timeline_summary": {"bins": [], "items": []},
                "timeline_range": {"items": []},
                "traits": {"traits": []},
                "knobs": {"rows": [], "source": ""},
            }
            self._json(HTTPStatus.OK, {"schema_version": f"run_{endpoint}_v1", **empty[endpoint]})
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "unknown run endpoint"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path or "/"
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(min(length, 2_000_000)) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            payload = {}
        if path == "/api/emergence/live_simulations/chronicle/live":
            try:
                self._json(HTTPStatus.OK, server_live_simulations.ingest_live_chronicle_event(payload))
            except Exception as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return
        if path == "/api/emergence/pressure_search/control":
            from tools import pressure_search

            self._json(HTTPStatus.OK, pressure_search.request_action(str(payload.get("campaign_id") or ""), str(payload.get("action") or payload.get("requested_action") or ""), target_run_key=str(payload.get("target_run_key") or ""), note=str(payload.get("note") or ""), requested_by="webui"))
            return
        if path == "/api/emergence/pressure_search/step":
            from tools import pressure_search

            self._json(HTTPStatus.OK, pressure_search.supervisor_step(str(payload.get("campaign_id") or ""), runs_root=self.cfg.runs_root))
            return
        if path.startswith("/api/builder/"):
            self._json(HTTPStatus.ACCEPTED, {"ok": False, "error": "standalone builder mutation/launch endpoint is not enabled in this migration branch"})
            return
        if path.startswith("/api/run/") and path.endswith("/notes"):
            self._json(HTTPStatus.ACCEPTED, {"ok": False, "error": "standalone run notes write endpoint is not enabled in this migration branch"})
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Emergence WebUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--runs_root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--scan_depth", type=int, default=12)
    parser.add_argument("--chronicle_root", action="append", default=None)
    parser.add_argument("--chronicle_scan_depth", type=int, default=8)
    parser.add_argument("--chronicle_live_root", action="append", default=None)
    parser.add_argument("--chronicle_archive_root", action="append", default=None)
    return parser.parse_args(argv)


def _cfg_from_args(args: argparse.Namespace) -> SimpleNamespace:
    runs_root = Path(args.runs_root).expanduser().resolve()
    default_chronicle_roots = (
        runs_root,
        DEFAULT_RUNTIME_ROOT / "evidence_raw",
        REPO_ROOT / "experiments" / "runs",
        LEGACY_RUNTIME_ROOT / "live",
        LEGACY_RUNTIME_ROOT / "evidence_raw",
    )
    default_archive_roots = (
        DEFAULT_ARCHIVE_ROOT,
        REPO_ROOT / "chronicle_history",
        LEGACY_RUNTIME_ROOT / "chronicle_history",
        COLD_CHRONICLE_ARCHIVE_ROOT,
    )
    chronicle_roots = [
        Path(p).expanduser().resolve()
        for p in (args.chronicle_root or [str(path) for path in default_chronicle_roots])
    ]
    archive_roots = [
        Path(p).expanduser().resolve()
        for p in (args.chronicle_archive_root or [str(path) for path in default_archive_roots])
    ]
    canopy_library_roots = (
        ("runtime", DEFAULT_RUNTIME_ROOT.expanduser().resolve()),
        ("cold storage", COLD_STORAGE_ROOT.expanduser().resolve()),
    )
    return SimpleNamespace(
        host=str(args.host),
        port=int(args.port),
        runs_root=runs_root,
        runtime_root=DEFAULT_RUNTIME_ROOT.expanduser().resolve(),
        scan_depth=max(1, min(32, int(args.scan_depth))),
        chronicle_roots=tuple(dict.fromkeys(chronicle_roots)),
        chronicle_scan_depth=max(1, min(32, int(args.chronicle_scan_depth))),
        chronicle_live_roots=tuple(Path(p).expanduser().resolve() for p in (args.chronicle_live_root or [])),
        chronicle_archive_root=archive_roots[0],
        chronicle_archive_roots=tuple(dict.fromkeys(archive_roots)),
        canopy_library_roots=canopy_library_roots,
        slot_telemetry_root=DEFAULT_SLOT_ROOT.expanduser().resolve(),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = _cfg_from_args(args)
    cfg.runs_root.mkdir(parents=True, exist_ok=True)
    cfg.slot_telemetry_root.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((cfg.host, cfg.port), EmergenceWebUIHandler)
    server.cfg = cfg  # type: ignore[attr-defined]
    print(f"[emergence-webui] repo_root={REPO_ROOT}", flush=True)
    print(f"[emergence-webui] runs_root={cfg.runs_root}", flush=True)
    print(f"[emergence-webui] http://{cfg.host}:{cfg.port}/", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
