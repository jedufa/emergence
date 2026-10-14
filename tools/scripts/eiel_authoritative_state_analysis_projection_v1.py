#!/usr/bin/env python3
"""Publish one immutable authoritative-state analysis projection receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine.campaigns.authoritative_state_analysis_projection_v1 import (
    AuthoritativeAnalysisProjectionError,
    build_analysis_projection_receipt_v1,
    write_immutable_analysis_projection_receipt_v1,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--analysis-source", required=True, type=Path)
    result.add_argument("--chronicle-events", required=True, type=Path)
    result.add_argument("--chronicle-event-id", required=True)
    result.add_argument("--output", required=True, type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        receipt = build_analysis_projection_receipt_v1(
            analysis_source_path=args.analysis_source,
            chronicle_events_path=args.chronicle_events,
            chronicle_event_id=args.chronicle_event_id,
        )
        digest = write_immutable_analysis_projection_receipt_v1(
            args.output, receipt
        )
    except (AuthoritativeAnalysisProjectionError, OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output.resolve()),
                "receipt_sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
