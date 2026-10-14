#!/usr/bin/env python3
from __future__ import annotations

"""Validation-only Chronicle server for LIVE1-R1 browser checks."""

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT
for path in (str(REPO_ROOT), str(PROJECT_ROOT), str(PROJECT_ROOT / "engine")):
    if path not in sys.path:
        sys.path.insert(0, path)

from infra.webui.server_chronicle import handle_get_request  # noqa: E402


STATIC_ROOT = REPO_ROOT / "infra" / "webui" / "static"
MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}


class Handler(BaseHTTPRequestHandler):
    cfg: SimpleNamespace

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(int(status))
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/chronicle"):
            status, payload = handle_get_request(cfg=self.cfg, path=parsed.path, qs=parse_qs(parsed.query))
            self._send(int(status), json.dumps(payload, sort_keys=True).encode("utf-8"), "application/json; charset=utf-8")
            return
        path = parsed.path
        if path in {"/", "/chronicle"}:
            path = "/chronicle.html"
        candidate = (STATIC_ROOT / path.lstrip("/")).resolve()
        try:
            candidate.relative_to(STATIC_ROOT.resolve())
        except ValueError:
            self._send(HTTPStatus.FORBIDDEN, b"forbidden", "text/plain; charset=utf-8")
            return
        if not candidate.is_file():
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain; charset=utf-8")
            return
        self._send(HTTPStatus.OK, candidate.read_bytes(), MIME_TYPES.get(candidate.suffix, "application/octet-stream"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve Chronicle LIVE1-R1 validation UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8781)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--chronicle-root", type=Path, action="append", required=True)
    parser.add_argument("--chronicle-live-root", type=Path, action="append", required=True)
    parser.add_argument("--chronicle-scan-depth", type=int, default=8)
    args = parser.parse_args(argv)
    Handler.cfg = SimpleNamespace(
        runs_root=args.runs_root.expanduser().resolve(),
        chronicle_roots=tuple(path.expanduser().resolve() for path in args.chronicle_root),
        chronicle_live_roots=tuple(path.expanduser().resolve() for path in args.chronicle_live_root),
        chronicle_scan_depth=max(1, min(32, int(args.chronicle_scan_depth))),
    )
    server = ThreadingHTTPServer((str(args.host), int(args.port)), Handler)
    print(f"http://{args.host}:{args.port}/chronicle", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
