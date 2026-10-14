#!/usr/bin/env python3
"""Regenerate the reading copy of the canonical EIEL work register."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import markdown


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "docs"
    / "100_governance-authority"
    / "110_authority-registers"
    / "EIEL-DOC-000007--eiel-unfinished-work-register.md"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "docs"
    / "100_governance-authority"
    / "110_authority-registers"
    / "EIEL-DOC-000006--eiel-unfinished-work-register.html"
)

STYLE = """
:root { --bg:#f4f6fb; --panel:#fff; --text:#172033; --muted:#657087; --line:#dce2ee; --brand:#4353cc; }
@media (prefers-color-scheme: dark) {
:root { --bg:#10131a; --panel:#171b24; --text:#edf1f8; --muted:#a8b1c2; --line:#303746; --brand:#9ca8ff; }
}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font:16px/1.65 system-ui,sans-serif}
.shell{width:min(1480px,100%);margin:auto;padding:28px;display:grid;grid-template-columns:280px minmax(0,920px);gap:30px;justify-content:center}
aside{position:sticky;top:20px;max-height:calc(100vh - 40px);overflow:auto;background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:18px}
main{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:44px 54px}
h1{font-size:2.6rem;line-height:1.1} h2{margin-top:3rem;border-top:1px solid var(--line);padding-top:1rem} h3{margin-top:2rem}
a{color:var(--brand)} code,pre{background:color-mix(in srgb,var(--panel) 75%,var(--line));border-radius:6px} code{padding:.12em .35em} pre{padding:16px;overflow:auto}
table{width:100%;border-collapse:collapse;font-size:.9rem;margin:1.4rem 0} th,td{border:1px solid var(--line);padding:9px;vertical-align:top} th{text-align:left}
blockquote{border-left:4px solid var(--brand);padding-left:16px;color:var(--muted)} aside ul{list-style:none;padding-left:12px} aside li{margin:5px 0}
@media(max-width:980px){.shell{display:block;padding:12px}aside{position:relative;max-height:none;margin-bottom:12px}main{padding:28px 20px}}
""".strip()


def render(source: Path, *, title: str, sidebar_title: str) -> str:
    converter = markdown.Markdown(
        extensions=("extra", "toc", "sane_lists"),
        extension_configs={"toc": {"permalink": True}},
        output_format="html5",
    )
    body = converter.convert(source.read_text(encoding="utf-8"))
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n<style>\n{STYLE}\n</style>\n"
        "</head>\n<body>\n<div class=\"shell\">\n"
        f"<aside><strong>{html.escape(sidebar_title)}</strong>"
        '<p style="color:var(--muted)">Generated from canonical Markdown</p>'
        f"{converter.toc}</aside>\n<main>{body}</main>\n"
        "</div>\n</body>\n</html>\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--title", default="EIEL Unfinished Work Register")
    parser.add_argument("--sidebar-title", default="EIEL Work Register")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = render(args.source, title=args.title, sidebar_title=args.sidebar_title)
    if args.check:
        if not args.output.is_file() or args.output.read_text(
            encoding="utf-8"
        ) != rendered:
            raise SystemExit("EIEL unfinished-work register HTML is stale")
        return 0
    args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
