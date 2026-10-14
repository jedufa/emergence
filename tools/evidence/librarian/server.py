"""Small token-protected LAN UI for Knowledge Librarian document ingress."""

from __future__ import annotations

import html
import json
import os
import secrets
import tempfile
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from .core import LibrarianError
from .ingress import (
    DEFAULT_HUMAN_INGRESS_RUNTIME_ROOT,
    IngressRequest,
    _allowed_document_classes,
    execute_external_document_ingress,
    preview_external_document,
)
from .publication import ALLOWED_DOC_SHELVES


MAX_UPLOAD_BYTES = 64 * 1024 * 1024


class IngressServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        repo_root: Path,
        source_root: Path,
        runtime_root: Path,
        token: str | None = None,
    ) -> None:
        super().__init__(server_address, IngressHandler)
        self.repo_root = repo_root.resolve()
        self.source_root = source_root.resolve()
        self.runtime_root = runtime_root.resolve()
        self.token = token or secrets.token_urlsafe(24)
        self.pending: dict[str, dict[str, Any]] = {}


def _page(title: str, body: str) -> bytes:
    return (
        "<!doctype html><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title>"
        "<style>"
        "body{font:14px system-ui,sans-serif;margin:24px;max-width:1100px}"
        "label{display:block;margin:10px 0 4px;font-weight:600}"
        "input,select,textarea{width:100%;max-width:780px;padding:7px}"
        "button{padding:8px 12px;margin-top:12px}"
        "pre{background:#f5f5f5;padding:12px;overflow:auto}"
        ".bad{color:#9b1c1c}.ok{color:#1b6b2a}"
        "</style>"
        f"<h1>{html.escape(title)}</h1>{body}"
        "<script>"
        "(function(){"
        "var origin=location.protocol+'//'+location.hostname+':8787';"
        "document.querySelectorAll('[data-webui-path]').forEach(function(a){"
        "a.href=origin+a.getAttribute('data-webui-path');"
        "a.target='_top';"
        "});"
        "}());"
        "</script>"
    ).encode("utf-8")


def _select(name: str, selected: str, options: list[str], *, element_id: str | None = None) -> str:
    rows = []
    for option in options:
        attr = " selected" if option == selected else ""
        rows.append(f"<option value='{html.escape(option, quote=True)}'{attr}>{html.escape(option)}</option>")
    id_attr = f" id='{html.escape(element_id, quote=True)}'" if element_id else ""
    return f"<select{id_attr} name='{html.escape(name, quote=True)}' required>{''.join(rows)}</select>"


def _shelf_groups(shelves: list[str]) -> dict[str, list[str]]:
    groups: dict[str, set[str]] = {}
    for shelf in shelves:
        parts = shelf.split("/", 1)
        root = parts[0]
        child = parts[1] if len(parts) == 2 else ""
        groups.setdefault(root, set()).add(child)
    return {root: sorted(children) for root, children in sorted(groups.items())}


def _shelf_controls(selected: str, shelves: list[str]) -> str:
    groups = _shelf_groups(shelves)
    selected_parts = selected.split("/", 1)
    selected_root = selected_parts[0] if selected_parts[0] in groups else sorted(groups)[0]
    selected_child = selected_parts[1] if len(selected_parts) == 2 else ""
    if selected_child not in groups[selected_root]:
        selected_child = groups[selected_root][0]
    roots = list(groups)
    children = groups[selected_root]
    group_json = json.dumps(groups, sort_keys=True).replace("</", "<\\/")
    return (
        f"<input id='shelfCombined' type='hidden' name='shelf' value='{html.escape(selected, quote=True)}'>"
        f"<label>Shelf</label>{_select('shelf_root', selected_root, roots, element_id='shelfRoot')}"
        f"<label>Destination Folder</label>{_select('shelf_child', selected_child, children, element_id='shelfChild')}"
        f"<script type='application/json' id='shelfGroups'>{group_json}</script>"
        "<script>"
        "(function(){"
        "var root=document.getElementById('shelfRoot');"
        "var child=document.getElementById('shelfChild');"
        "var combined=document.getElementById('shelfCombined');"
        "var raw=document.getElementById('shelfGroups');"
        "if(!root||!child||!combined||!raw)return;"
        "var groups=JSON.parse(raw.textContent||'{}');"
        "function syncChild(){"
        "var current=child.value;"
        "child.textContent='';"
        "(groups[root.value]||['']).forEach(function(value){"
        "var opt=document.createElement('option');"
        "opt.value=value;"
        "opt.textContent=value||'(top level)';"
        "if(value===current)opt.selected=true;"
        "child.appendChild(opt);"
        "});"
        "if(!child.value&&child.options.length)child.selectedIndex=0;"
        "syncCombined();"
        "}"
        "function syncCombined(){combined.value=child.value?root.value+'/'+child.value:root.value;}"
        "root.addEventListener('change',syncChild);"
        "child.addEventListener('change',syncCombined);"
        "syncCombined();"
        "}());"
        "</script>"
    )


def _form(
    token: str,
    pending_id: str | None = None,
    values: dict[str, Any] | None = None,
    *,
    document_classes: list[str] | None = None,
    shelves: list[str] | None = None,
) -> str:
    pending = f"<input type='hidden' name='pending_id' value='{html.escape(pending_id)}'>" if pending_id else ""
    action = "/ingress/confirm" if pending_id else "/ingress/preview"
    button = "Confirm Publish" if pending_id else "Preview"
    values = values or {}
    title = html.escape(str(values.get("title") or ""), quote=True)
    document_class = str(values.get("document_class") or "research_reference")
    shelf = str(values.get("shelf") or "600_references-publications/600_research-references")
    document_classes = document_classes or ["research_reference"]
    shelves = shelves or ["600_references-publications/600_research-references"]
    dossiers = html.escape(str(values.get("dossiers") or ""), quote=True)
    publication_id = html.escape(str(values.get("publication_id") or "human-ingress-v1"), quote=True)
    return f"""
<form method="post" action="{action}" enctype="multipart/form-data">
  <input type="hidden" name="token" value="{html.escape(token)}">
  {pending}
  <label>External File</label><input type="file" name="upload">
  <label>Or Source Path</label><input name="source_path" placeholder="/absolute/path/to/report.md">
  <label>Title</label><input name="title" value="{title}" required>
  <label>Document Class</label>{_select("document_class", document_class, document_classes)}
  {_shelf_controls(shelf, shelves)}
  <label>Dossier IDs, comma separated</label><input name="dossiers" value="{dossiers}">
  <label>Publication ID</label><input name="publication_id" value="{publication_id}">
  <label>Mode</label><select name="mode"><option value="stage">Stage</option><option value="commit">Commit</option></select>
  <label><input style="width:auto" type="checkbox" name="trash_source" value="1"> Trash source after successful publish</label>
  <button>{button}</button>
</form>
"""


def _multipart(content_type: str, body: bytes) -> dict[str, Any]:
    message = BytesParser(policy=policy.default).parsebytes(
        f"Content-Type: {content_type}\nMIME-Version: 1.0\n\n".encode("utf-8") + body
    )
    fields: dict[str, Any] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename:
            fields[name] = {"filename": filename, "bytes": payload}
        else:
            fields[name] = payload.decode(part.get_content_charset() or "utf-8")
    return fields


class IngressHandler(BaseHTTPRequestHandler):
    server: IngressServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send(self, status: HTTPStatus, body: bytes, content_type: str = "text/html; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _form(self, pending_id: str | None = None, values: dict[str, Any] | None = None) -> str:
        document_classes = sorted(_allowed_document_classes(self.server.source_root)) or ["research_reference"]
        preferred_classes = ["research_reference", "decision_report", "implementation_report", "campaign_report", "handoff"]
        document_classes = [value for value in preferred_classes if value in document_classes] + [
            value for value in document_classes if value not in preferred_classes
        ]
        preferred_shelves = [
            "600_references-publications/600_research-references",
            "600_references-publications/610_books-papers-external",
            "500_campaigns-decisions-closeouts/500_decision-reports",
            "500_campaigns-decisions-closeouts/520_campaign-reports",
            "400_implementation-engineering/410_implementation-reports",
            "400_implementation-engineering/400_implementation-plans",
            "900_lifecycle/940_incoming",
        ]
        existing_shelves = set(ALLOWED_DOC_SHELVES)
        for record_path in sorted((self.server.source_root / "documents").glob("*.json")):
            try:
                row = json.loads(record_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            path = Path(str(row.get("path") or ""))
            if len(path.parts) >= 3 and path.parts[0] == "docs":
                existing_shelves.add(Path(*path.parts[1:-1]).as_posix())
        shelves = sorted(existing_shelves | set(preferred_shelves))
        shelves = [value for value in preferred_shelves if value in shelves] + [
            value for value in shelves if value not in preferred_shelves
        ]
        return _form(
            self.server.token,
            pending_id,
            values,
            document_classes=document_classes,
            shelves=shelves,
        )

    def do_GET(self) -> None:
        if self.path not in {"/", "/add-document"}:
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain; charset=utf-8")
            return
        body = (
            "<p><a href='/add-document'>Add document</a> | "
            "<a href='#' data-webui-path='/docs/'>Docs</a> | "
            "<a href='#' data-webui-path='/docs/000_knowledge-catalog.html'>Knowledge Catalog</a></p>"
            + self._form()
        )
        self._send(HTTPStatus.OK, _page("EIEL Knowledge Librarian", body))

    def _read_fields(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > MAX_UPLOAD_BYTES:
            raise LibrarianError("upload exceeds configured limit")
        body = self.rfile.read(length)
        ctype = self.headers.get("Content-Type", "")
        if ctype.startswith("multipart/form-data"):
            return _multipart(ctype, body)
        return {key: values[-1] for key, values in parse_qs(body.decode("utf-8")).items()}

    def _source_from_fields(self, fields: dict[str, Any], pending_id: str | None = None) -> tuple[Path, str | None]:
        if pending_id:
            pending = self.server.pending.get(pending_id)
            if not pending:
                raise LibrarianError("pending preview is missing or expired")
            return Path(str(pending["source"])), str(pending["source_label"])
        upload = fields.get("upload")
        if isinstance(upload, dict) and upload.get("bytes"):
            filename = Path(str(upload.get("filename") or "upload")).name
            if not filename:
                raise LibrarianError("uploaded filename is empty")
            target_dir = self.server.runtime_root / "web_uploads"
            target_dir.mkdir(parents=True, exist_ok=True)
            fd, target_name = tempfile.mkstemp(prefix="upload-", suffix=f"-{filename}", dir=target_dir)
            with os.fdopen(fd, "wb") as handle:
                handle.write(upload["bytes"])
            return Path(target_name), f"uploaded:{filename}"
        source_path = str(fields.get("source_path") or "").strip()
        if not source_path:
            raise LibrarianError("choose an upload or provide a source path")
        return Path(source_path), source_path

    def _request(self, fields: dict[str, Any], source: Path, source_label: str | None) -> IngressRequest:
        dossiers = [
            value.strip()
            for value in str(fields.get("dossiers") or "").split(",")
            if value.strip()
        ]
        return IngressRequest(
            source=source,
            source_label=source_label,
            title=str(fields.get("title") or "").strip(),
            document_class=str(fields.get("document_class") or "").strip(),
            destination_shelf=str(fields.get("shelf") or "").strip(),
            linked_dossier_ids=tuple(dossiers),
            publication_id=str(fields.get("publication_id") or "human-ingress-v1").strip(),
        )

    def _pending_values(self, fields: dict[str, Any]) -> dict[str, Any]:
        return {
            "title": str(fields.get("title") or ""),
            "document_class": str(fields.get("document_class") or ""),
            "shelf": str(fields.get("shelf") or ""),
            "dossiers": str(fields.get("dossiers") or ""),
            "publication_id": str(fields.get("publication_id") or ""),
        }

    def do_POST(self) -> None:
        try:
            fields = self._read_fields()
            if fields.get("token") != self.server.token:
                self._send(HTTPStatus.FORBIDDEN, b"forbidden", "text/plain; charset=utf-8")
                return
            if self.path == "/ingress/preview":
                source, source_label = self._source_from_fields(fields)
                request = self._request(fields, source, source_label)
                preview = preview_external_document(
                    repo_root=self.server.repo_root,
                    source_root=self.server.source_root,
                    request=request,
                )
                pending_id = secrets.token_urlsafe(16)
                values = self._pending_values(fields)
                self.server.pending[pending_id] = {
                    "source": source.as_posix(),
                    "source_label": source_label,
                    "values": values,
                }
                status_class = "ok" if preview["ok"] else "bad"
                body = (
                    f"<p class='{status_class}'>Preview {'ready' if preview['ok'] else 'blocked'}.</p>"
                    f"<pre>{html.escape(json.dumps(preview, indent=2, sort_keys=True))}</pre>"
                    + (self._form(pending_id, values) if preview["ok"] else self._form(values=values))
                )
                self._send(HTTPStatus.OK, _page("Decision Preview", body))
                return
            if self.path == "/ingress/confirm":
                pending_id = str(fields.get("pending_id") or "")
                pending = self.server.pending.get(pending_id)
                if pending:
                    merged = dict(pending.get("values") or {})
                    merged.update({key: value for key, value in fields.items() if value not in {"", None}})
                    fields = merged
                source, source_label = self._source_from_fields(fields, pending_id=pending_id)
                request = self._request(fields, source, source_label)
                receipt = execute_external_document_ingress(
                    repo_root=self.server.repo_root,
                    source_root=self.server.source_root,
                    request=request,
                    mode=str(fields.get("mode") or "stage"),
                    runtime_root=self.server.runtime_root,
                    trash_source=str(fields.get("trash_source") or "") == "1",
                )
                self.server.pending.pop(pending_id, None)
                body = f"<p class='ok'>Published {html.escape(receipt['document_id'])}.</p><pre>{html.escape(json.dumps(receipt, indent=2, sort_keys=True))}</pre>"
                self._send(HTTPStatus.OK, _page("Ingress Receipt", body))
                return
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain; charset=utf-8")
        except LibrarianError as exc:
            self._send(
                HTTPStatus.BAD_REQUEST,
                _page("Ingress Blocked", f"<p class='bad'>{html.escape(str(exc))}</p>{self._form()}"),
            )


def serve(
    *,
    host: str,
    port: int,
    repo_root: Path,
    source_root: Path,
    runtime_root: Path = DEFAULT_HUMAN_INGRESS_RUNTIME_ROOT,
    token: str | None = None,
) -> None:
    server = IngressServer(
        (host, port),
        repo_root=repo_root,
        source_root=source_root,
        runtime_root=runtime_root,
        token=token,
    )
    print(f"EIEL Knowledge Librarian ingress UI: http://{host}:{port}/")
    print(f"Token: {server.token}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
