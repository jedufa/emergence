"""CLI for deterministic qualification evidence review."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools.actors.canonical import canonical_bytes

from .engine import review_spec
from .validators.temporal import parse_utc


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Validate qualification evidence from a declarative JSON spec."
    )
    value.add_argument("spec", type=Path, help="review specification JSON")
    value.add_argument("--output", type=Path, help="write the canonical receipt")
    value.add_argument("--reuse-receipt", type=Path, help="reuse an exact-input green receipt")
    value.add_argument("--now", help="UTC/offset ISO-8601 evaluation time (testing/replay)")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    now = parse_utc(args.now) if args.now else None
    receipt = review_spec(args.spec, now=now, reuse_receipt=args.reuse_receipt)
    payload = canonical_bytes(receipt) + b"\n"
    if args.output:
        target = args.output.resolve()
        if target == args.spec.resolve():
            parser().error("--output may not overwrite the review spec")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    sys.stdout.buffer.write(payload)
    return 0 if receipt["MECHANICAL_REVIEW"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
