from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backfill import backfill_sources, bulk_backfill


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only EIEL Canopy checkpoint backfill")
    parser.add_argument("--source", action="append", type=Path)
    parser.add_argument("--source-root", action="append", type=Path)
    parser.add_argument("--derived-root", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--generated-at")
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--max-count", type=int)
    parser.add_argument("--include")
    parser.add_argument("--exclude")
    args = parser.parse_args(argv)
    if not args.source and not args.source_root:
        parser.error("one of --source or --source-root is required")
    result = backfill_sources(args.source, derived_root=args.derived_root, dry_run=args.dry_run, generated_at=args.generated_at) if args.source else bulk_backfill(args.source_root, derived_root=args.derived_root, dry_run=args.dry_run, max_depth=args.max_depth, include=args.include, exclude=args.exclude, max_count=args.max_count, generated_at=args.generated_at)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
