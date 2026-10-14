#!/usr/bin/env python3
"""Regenerate the reading copy of the EIEL qualification authority register."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "docs"
    / "100_governance-authority"
    / "110_authority-registers"
    / "EIEL-DOC-000005--eiel-qualification-authority-register.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "docs"
    / "100_governance-authority"
    / "110_authority-registers"
    / "EIEL-DOC-000004--eiel-qualification-authority-register.html"
)
STATUS_SORT_ORDER = {
    "qualified": 0,
    "blocked": 1,
    "superseded": 2,
    "partial": 3,
    "in_progress": 4,
    "rejected": 5,
    "expired": 6,
    "unknown": 7,
}

STYLE = """
:root { --bg:#f4f6fb; --panel:#fff; --text:#172033; --muted:#657087; --line:#dce2ee; --brand:#4353cc; --ok:#176a3a; --warn:#9a5b00; --bad:#a12b2b; }
@media (prefers-color-scheme: dark) {
:root { --bg:#10131a; --panel:#171b24; --text:#edf1f8; --muted:#a8b1c2; --line:#303746; --brand:#9ca8ff; --ok:#65d38b; --warn:#f2b84b; --bad:#ff8a8a; }
}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,sans-serif}
.shell{width:min(1500px,100%);margin:auto;padding:28px}
header,.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:22px;margin-bottom:18px}
h1{font-size:2rem;line-height:1.15;margin:.1rem 0 .4rem} h2{margin:0 0 12px}
.muted{color:var(--muted)} code{background:color-mix(in srgb,var(--panel) 75%,var(--line));border-radius:5px;padding:.12em .35em;word-break:break-word}
table{width:100%;border-collapse:collapse;font-size:.88rem} th,td{border:1px solid var(--line);padding:8px 9px;vertical-align:top} th{text-align:left;background:color-mix(in srgb,var(--panel) 80%,var(--line))}
.status{font-weight:700}.qualified{color:var(--ok)}.in_progress,.partial,.unknown,.superseded,.expired{color:var(--warn)}.blocked,.rejected{color:var(--bad)}
.table-scroll{max-width:100%;overflow:auto;border:1px solid var(--line);border-radius:8px}
.table-scroll table{min-width:1180px;border:0}
.table-scroll th:first-child,.table-scroll td:first-child{border-left:0}
.table-scroll th:last-child,.table-scroll td:last-child{border-right:0}
.table-scroll thead tr:first-child th{border-top:0}
.table-scroll tbody tr:last-child td{border-bottom:0}
#tracks-table th:first-child,#tracks-table td:first-child,#tracks-table td:first-child code{white-space:nowrap;word-break:normal}
#tracks-table .device{white-space:nowrap;font-weight:700;text-transform:uppercase}
#tracks-table .world{min-width:12rem}
#tracks-table .coverage{text-align:center;font-weight:700;white-space:nowrap}
#tracks-table .covered{color:var(--ok)}
#tracks-table .blocked-cell{color:var(--bad)}
#tracks-table .not-covered{color:var(--muted)}
#tracks-table .comment{min-width:22rem}
#tracks-table th.sortable{cursor:pointer;user-select:none}
#tracks-table th.sortable::after{content:attr(data-sort-indicator);color:var(--muted);font-weight:400;margin-left:.35rem}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}@media(max-width:900px){.grid{grid-template-columns:1fr}.shell{padding:12px}}
.copy-actions{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 12px}
button{border:1px solid var(--line);border-radius:8px;background:color-mix(in srgb,var(--panel) 84%,var(--line));color:var(--text);font:inherit;font-weight:700;padding:7px 10px;cursor:pointer}
button:hover{border-color:var(--brand)}
textarea{width:100%;min-height:150px;border:1px solid var(--line);border-radius:8px;background:color-mix(in srgb,var(--panel) 90%,var(--line));color:var(--text);font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;padding:10px;resize:vertical}
""".strip()


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def code(value: Any) -> str:
    return f"<code>{esc(value)}</code>"


def sentence_list(prefix: str, values: list[Any] | None) -> str:
    if not values:
        return ""
    return f"{esc(prefix)}: " + "; ".join(esc(value) for value in values)


def notes_for_track(track: dict[str, Any]) -> str:
    parts: list[str] = []
    if report := track.get("durable_report_path"):
        parts.append(f"Durable report: {code(report)}.")
    if blocker := track.get("blocker"):
        parts.append("Blocker: " + esc(blocker.get("reason", blocker)))
    if notes := track.get("resource_notes"):
        if isinstance(notes, dict):
            rendered = ", ".join(f"{esc(k)}={esc(v)}" for k, v in notes.items())
            parts.append(f"Resource notes: {rendered}")
        else:
            parts.append(f"Resource notes: {esc(notes)}")
    if evidence := track.get("volatile_raw_evidence"):
        parts.append(
            "Volatile raw evidence:<br>"
            + "<br>".join(code(path) for path in evidence)
        )
    for label, key in (("Next", "next_required_evidence"), ("Limitations", "limitations")):
        rendered = sentence_list(label, track.get(key))
        if rendered:
            parts.append(rendered)
    return "<br>".join(parts)


def device_for_track(track: dict[str, Any]) -> str:
    envelope = track.get("execution_envelope")
    if isinstance(envelope, dict) and envelope.get("device"):
        return str(envelope["device"]).lower()
    label = str(track.get("label", "")).lower()
    if label.endswith("gpu"):
        return "gpu"
    if label.endswith("cpu"):
        return "cpu"
    return ""


def comment_for_track(track: dict[str, Any]) -> str:
    status = track.get("status")
    if status not in {"blocked", "superseded"}:
        return ""
    blocker = track.get("blocker")
    if isinstance(blocker, dict) and blocker.get("reason"):
        return str(blocker["reason"])
    if blocker:
        return str(blocker)
    limitations = track.get("limitations")
    if limitations:
        return str(limitations[0])
    return ""


def coverage_for_track(track: dict[str, Any]) -> dict[str, bool]:
    world = str(track.get("world", "")).lower()
    return {
        "patch": world == "patch" or world.startswith("patch+"),
        "wdl": "wdl" in world,
        "jungle": "jungle" in world,
        "eed": "eed" in world,
        "births": "birth" in world,
        "genome_v2_birth": "genome_v2_ordinary_birth" in world,
        "genome_v2_mutation": "genome_v2_mutation" in world,
        "recombination": "recombination" in world,
    }


def coverage_cell(covered: bool, status: Any) -> str:
    if covered and status == "qualified":
        return '<td class="coverage covered" aria-label="qualified">qualified</td>'
    if covered:
        return f'<td class="coverage blocked-cell" aria-label="{esc(status)}">{esc(status)}</td>'
    return '<td class="coverage not-covered" aria-label="not covered">--</td>'


def sorted_tracks_for_human_table(tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        track
        for _, track in sorted(
            enumerate(tracks),
            key=lambda item: (
                STATUS_SORT_ORDER.get(str(item[1].get("status", "unknown")), 99),
                item[0],
            ),
        )
    ]


def render(data: dict[str, Any]) -> str:
    repo = data.get("repository", {})
    policy = data.get("policy", {})
    preservation = policy.get("qualification_preservation_rule", {})
    if not isinstance(preservation, dict):
        preservation = {}
    tracks = data.get("tracks", [])
    counts = Counter(track.get("status", "unknown") for track in tracks)
    count_text = ", ".join(f"{status}: {count}" for status, count in sorted(counts.items()))

    report_rows = []
    for key, report in data.get("durable_reports", {}).items():
        report_rows.append(
            "<tr>"
            f"<td>{code(key)}</td>"
            f"<td>{code(report.get('path', ''))}</td>"
            f"<td>{esc(report.get('scope', ''))}</td>"
            "</tr>"
        )

    track_rows = []
    for track in sorted_tracks_for_human_table(tracks):
        status = track.get("status", "unknown")
        coverage = coverage_for_track(track)
        track_rows.append(
            "<tr>"
            f"<td>{code(track.get('label', ''))}</td>"
            f"<td class=\"status {esc(status)}\">{esc(status)}</td>"
            f"<td class=\"device\">{esc(device_for_track(track))}</td>"
            f"<td class=\"world\">{esc(track.get('world', ''))}</td>"
            f"{coverage_cell(coverage['patch'], status)}"
            f"{coverage_cell(coverage['wdl'], status)}"
            f"{coverage_cell(coverage['jungle'], status)}"
            f"{coverage_cell(coverage['births'], status)}"
            f"{coverage_cell(coverage['eed'], status)}"
            f"{coverage_cell(coverage['genome_v2_birth'], status)}"
            f"{coverage_cell(coverage['genome_v2_mutation'], status)}"
            f"{coverage_cell(coverage['recombination'], status)}"
            f"<td>{esc(track.get('role', ''))}</td>"
            f"<td class=\"comment\">{esc(comment_for_track(track))}</td>"
            "</tr>"
        )

    artifact_rows = []
    for key, artifact in data.get("artifacts", {}).items():
        digest = artifact.get("sha256") or artifact.get("manifest_sha256") or ""
        artifact_rows.append(
            "<tr>"
            f"<td>{code(key)}</td>"
            f"<td>{esc(artifact.get('status', ''))}</td>"
            f"<td>{esc(artifact.get('storage_class', ''))}</td>"
            f"<td>{code(digest) if digest else ''}<br><span class=\"muted\">raw path:</span> {code(artifact.get('path', ''))}</td>"
            "</tr>"
        )

    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>EIEL Qualification Authority Register</title>\n"
        f"<style>\n{STYLE}\n</style>\n</head>\n<body>\n<div class=\"shell\">\n"
        "<header>\n<h1>EIEL Qualification Authority Register</h1>\n"
        f"<p class=\"muted\">Reading copy generated from {code(data.get('canonical_file', DEFAULT_SOURCE))}. "
        f"Last updated {esc(data.get('last_updated', ''))} on {code(repo.get('branch', ''))} at {code(repo.get('commit', ''))}.</p>\n"
        f"<p>{esc(policy.get('no_claims', ''))}</p>\n</header>\n"
        "<section class=\"panel\">\n<h2>Qualification Preservation and Delta Qualification</h2>\n"
        f"<p><strong>Default:</strong> {esc(preservation.get('default_presumption', ''))}</p>\n"
        f"<p><strong>Impact map:</strong> {esc(preservation.get('impact_map_required', ''))}</p>\n"
        f"<p><strong>Inheritance:</strong> {esc(preservation.get('inheritance_rule', ''))}</p>\n"
        f"<p><strong>Envelopes and variants:</strong> {esc(preservation.get('envelope_and_variant_rule', ''))}</p>\n"
        f"<p><strong>New capabilities and routes:</strong> {esc(preservation.get('new_capability_rule', ''))}</p>\n"
        f"<p><strong>Broad requalification:</strong> {esc(preservation.get('broad_requalification_rule', ''))}</p>\n"
        f"<p><strong>Governing procedure:</strong> {code(preservation.get('procedure', ''))}</p>\n"
        "</section>\n"
        "<section class=\"panel\">\n<h2>Current Summary</h2>\n<div class=\"grid\">\n"
        f"<div><strong>Status Counts</strong><br><span class=\"muted\">{esc(count_text)}</span></div>\n"
        "<div><strong>Durable Reports</strong><br><span class=\"muted\">Required for accepted, blocked, partial, superseded, rejected, or retired entries.</span></div>\n"
        "<div><strong>Volatile Raw Evidence</strong><br><span class=\"muted\">/data, runtime, campaign, and artifact paths are locators only, not durable authority.</span></div>\n"
        "</div>\n</section>\n"
        "<section class=\"panel\">\n<h2>Copy Routes</h2>\n"
        '<div class="copy-actions" aria-label="Route copy actions">\n'
        '<button type="button" data-copy-mode="qualified">Copy Qualified</button>\n'
        '<button type="button" data-copy-mode="not-qualified">Copy Not Qualified</button>\n'
        '<button type="button" data-copy-mode="all">Copy All</button>\n'
        "</div>\n"
        '<textarea id="route-copy-output" readonly aria-label="Routes with status"></textarea>\n'
        '<p id="copy-feedback" class="muted">Buttons copy tab-separated rows from the visible table.</p>\n'
        "</section>\n"
        "<section class=\"panel\">\n<h2>Durable Reports</h2>\n"
        "<table><thead><tr><th>Key</th><th>Path</th><th>Scope</th></tr></thead><tbody>\n"
        + "\n".join(report_rows)
        + "\n</tbody></table>\n</section>\n"
        "<section class=\"panel\">\n<h2>Tracks</h2>\n"
        '<div class="table-scroll" role="region" aria-label="Scrollable qualification tracks table" tabindex="0">\n'
        '<table id="tracks-table">\n<thead><tr><th data-sort-key="track">Track</th><th data-sort-key="status">Status</th><th data-sort-key="device">CPU/GPU</th><th data-sort-key="world">World</th><th data-sort-key="patch">Patch</th><th data-sort-key="wdl">WDL</th><th data-sort-key="jungle">Jungle</th><th data-sort-key="births">Births</th><th data-sort-key="eed">EED</th><th data-sort-key="genome-v2-birth">GV2 Birth</th><th data-sort-key="genome-v2-mutation">GV2 Mutation</th><th data-sort-key="recombination">Recombination</th><th data-sort-key="role">Role</th><th data-sort-key="comment">Comment</th></tr></thead>\n<tbody>\n'
        + "\n".join(track_rows)
        + "\n</tbody>\n</table>\n</div>\n</section>\n"
        "<section class=\"panel\">\n<h2>Artifacts</h2>\n"
        "<table><thead><tr><th>Key</th><th>Status</th><th>Storage</th><th>Digest / Raw Path</th></tr></thead><tbody>\n"
        + "\n".join(artifact_rows)
        + "\n</tbody></table>\n</section>\n"
        "<section class=\"panel\">\n<h2>Maintenance Rule</h2>\n"
        f"<p>{esc(policy.get('html_sync_rule', ''))}</p>"
        f"<p>{esc(policy.get('durable_report_rule', ''))}</p>"
        f"<p>{esc(policy.get('volatile_storage_rule', ''))}</p>"
        f"<p>{esc(policy.get('update_rule', ''))}</p>\n"
        "</section>\n"
        "</div>\n"
        "<script>\n"
        "(() => {\n"
        "  const output = document.querySelector('#route-copy-output');\n"
        "  const feedback = document.querySelector('#copy-feedback');\n"
        "  const table = document.querySelector('#tracks-table');\n"
        "  const statusOrder = { qualified: 0, blocked: 1, superseded: 2, partial: 3, in_progress: 4, rejected: 5, expired: 6, unknown: 7 };\n"
        "  const sortState = { index: 1, direction: 'asc' };\n"
        "  function cellText(cell) {\n"
        "    return (cell?.textContent || '').replace(/\\s+/g, ' ').trim();\n"
        "  }\n"
        "  function sortValue(row, index) {\n"
        "    const text = cellText(row.cells[index]);\n"
        "    if (index === 1) return statusOrder[text] ?? 99;\n"
        "    return text.toLowerCase();\n"
        "  }\n"
        "  function compareRows(index, direction) {\n"
        "    return (left, right) => {\n"
        "      const leftValue = sortValue(left, index);\n"
        "      const rightValue = sortValue(right, index);\n"
        "      let result = 0;\n"
        "      if (typeof leftValue === 'number' && typeof rightValue === 'number') {\n"
        "        result = leftValue - rightValue;\n"
        "      } else {\n"
        "        result = String(leftValue).localeCompare(String(rightValue), undefined, { numeric: true, sensitivity: 'base' });\n"
        "      }\n"
        "      if (result === 0) result = Number(left.dataset.originalIndex) - Number(right.dataset.originalIndex);\n"
        "      return direction === 'asc' ? result : -result;\n"
        "    };\n"
        "  }\n"
        "  function updateSortHeaders(index, direction) {\n"
        "    table.querySelectorAll('thead th').forEach((header, headerIndex) => {\n"
        "      header.setAttribute('aria-sort', headerIndex === index ? (direction === 'asc' ? 'ascending' : 'descending') : 'none');\n"
        "      header.dataset.sortIndicator = headerIndex === index ? (direction === 'asc' ? 'asc' : 'desc') : '';\n"
        "    });\n"
        "  }\n"
        "  function sortTable(index, direction) {\n"
        "    const body = table.tBodies[0];\n"
        "    Array.from(body.rows).sort(compareRows(index, direction)).forEach((row) => body.appendChild(row));\n"
        "    sortState.index = index;\n"
        "    sortState.direction = direction;\n"
        "    updateSortHeaders(index, direction);\n"
        "    refreshOutput();\n"
        "  }\n"
        "  function tableRows(mode) {\n"
        "    const headers = Array.from(table.querySelectorAll('thead th')).map(cellText);\n"
        "    const rows = Array.from(table.querySelectorAll('tbody tr'));\n"
        "    const selected = rows.filter((row) => {\n"
        "      const status = cellText(row.cells[1]);\n"
        "      if (mode === 'qualified') return status === 'qualified';\n"
        "      if (mode === 'not-qualified') return status !== 'qualified';\n"
        "      return true;\n"
        "    });\n"
        "    return [headers.join('\\t'), ...selected.map((row) => Array.from(row.cells).map(cellText).join('\\t'))];\n"
        "  }\n"
        "  function refreshOutput(mode = 'all') {\n"
        "    output.value = tableRows(mode).join('\\n');\n"
        "    return output.value;\n"
        "  }\n"
        "  Array.from(table.tBodies[0].rows).forEach((row, index) => { row.dataset.originalIndex = String(index); });\n"
        "  table.querySelectorAll('thead th').forEach((header, index) => {\n"
        "    header.classList.add('sortable');\n"
        "    header.tabIndex = 0;\n"
        "    header.addEventListener('click', () => sortTable(index, sortState.index === index && sortState.direction === 'asc' ? 'desc' : 'asc'));\n"
        "    header.addEventListener('keydown', (event) => {\n"
        "      if (event.key === 'Enter' || event.key === ' ') {\n"
        "        event.preventDefault();\n"
        "        header.click();\n"
        "      }\n"
        "    });\n"
        "  });\n"
        "  updateSortHeaders(sortState.index, sortState.direction);\n"
        "  refreshOutput();\n"
        "  document.querySelectorAll('button[data-copy-mode]').forEach((button) => {\n"
        "    button.addEventListener('click', async () => {\n"
        "      const text = refreshOutput(button.dataset.copyMode);\n"
        "      const rowCount = Math.max(0, text.split('\\n').length - 1);\n"
        "      try { await navigator.clipboard.writeText(text); feedback.textContent = `Copied ${rowCount} rows.`; }\n"
        "      catch { feedback.textContent = 'Copy failed; select the text area manually.'; }\n"
        "    });\n"
        "  });\n"
        "})();\n"
        "</script>\n"
        "</body>\n</html>\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.source.read_text(encoding="utf-8"))
    rendered = render(data)
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit("EIEL qualification authority register HTML is stale")
        return 0
    args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
