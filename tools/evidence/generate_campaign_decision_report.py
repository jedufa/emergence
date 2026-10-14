"""CLI for deterministic campaign decision reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.evidence.reports import generate_campaign_decision_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an evidence campaign decision report.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--generated-at")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = generate_campaign_decision_report(
            args.manifest,
            args.profile,
            args.out_dir,
            generated_at=args.generated_at,
            deterministic=args.deterministic,
        )
    except NotImplementedError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({"out_dir": result["out_dir"], "decision": result["decision"]["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
