"""CLI for evidence campaign schema checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.evidence.contracts import check_campaign_schema, dumps_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check an evidence campaign manifest and run outputs.")
    parser.add_argument("--manifest", required=True, help="Path to campaign_evidence_manifest JSON")
    parser.add_argument("--out", help="Optional JSON output path")
    args = parser.parse_args(argv)

    result = check_campaign_schema(args.manifest)
    text = dumps_result(result)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
