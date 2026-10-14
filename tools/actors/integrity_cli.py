"""Read-only composite integrity front door for EIEL durable state.

The command owns no records and performs no repair.  It delegates to the
Workspace repository-wide audit, which traverses Actor stores, Scheduler
stores, Context Receipts, Work Package receipts, WRA artifacts, Missions,
review/qualification bindings, and dependency/lifecycle edges.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from tools.actors import workspace


def _help() -> str:
    return (
        "Usage: python -m tools.eiel integrity audit [workspace audit options]\n"
        "\n"
        "Run the non-mutating composite graph audit.  The result separates\n"
        "CORRUPT, RECOVERABLE_STALE, SUPPORTED_HISTORICAL, OPTIONAL_ABSENT,\n"
        "and EXTERNAL_UNVERIFIABLE outcomes.\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(_help(), end="")
        return 0
    if args[0] != "audit":
        print(f"Unknown integrity command: {args[0]!r}\n", file=sys.stderr)
        print(_help(), file=sys.stderr, end="")
        return 2
    return workspace.main(["audit", *args[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
