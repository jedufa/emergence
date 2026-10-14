from __future__ import annotations

import argparse
import json
from pathlib import Path

from .relay import LineagePlaybackRelay, build_staged_playback_source, validate_relay_manifest


def _parse_boundaries(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("boundary list cannot be empty")
    try:
        return [int(part) for part in parts]
    except ValueError as error:
        raise argparse.ArgumentTypeError("boundaries must be comma-separated integers") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLC near-live playback relay V0")
    sub = parser.add_subparsers(dest="command", required=True)

    build_fixture = sub.add_parser("build-staged-source", help="build a staged normalized-playback source fixture")
    build_fixture.add_argument("--playback", required=True, type=Path)
    build_fixture.add_argument("--provenance", required=True, type=Path)
    build_fixture.add_argument("--out", required=True, type=Path)
    build_fixture.add_argument("--boundaries", required=True, type=_parse_boundaries)
    build_fixture.add_argument(
        "--source-state",
        choices=["source_interrupted", "source_complete", "source_stale"],
        default="source_interrupted",
    )

    update = sub.add_parser("update-once", help="scan source once and publish at most one relay update")
    update.add_argument("--source-root", required=True, type=Path)
    update.add_argument("--output-root", required=True, type=Path)
    update.add_argument(
        "--source-kind",
        choices=["auto", "staged_normalized_playback", "gv2_checkpoint_bundle_v2"],
        default="auto",
    )
    update.add_argument("--cell-id")
    update.add_argument("--poll-seconds", type=float, default=2.0)
    update.add_argument("--relay-id")

    watch = sub.add_parser("watch", help="poll source and publish relay updates")
    watch.add_argument("--source-root", required=True, type=Path)
    watch.add_argument("--output-root", required=True, type=Path)
    watch.add_argument(
        "--source-kind",
        choices=["auto", "staged_normalized_playback", "gv2_checkpoint_bundle_v2"],
        default="auto",
    )
    watch.add_argument("--cell-id")
    watch.add_argument("--poll-seconds", type=float, default=2.0)
    watch.add_argument("--relay-id")
    watch.add_argument("--max-updates", type=int)

    validate = sub.add_parser("validate-manifest", help="validate a relay manifest and referenced files")
    validate.add_argument("--manifest", required=True, type=Path)

    args = parser.parse_args(argv)
    if args.command == "build-staged-source":
        manifest = build_staged_playback_source(
            playback_path=args.playback,
            provenance_path=args.provenance,
            out=args.out,
            boundaries=args.boundaries,
            source_state=args.source_state,
        )
        print(json.dumps({"ok": True, "stage_count": manifest["stage_count"], "final_boundary": manifest["final_boundary"]}, indent=2, sort_keys=True))
        return 0
    if args.command == "update-once":
        relay = LineagePlaybackRelay(
            source_root=args.source_root,
            output_root=args.output_root,
            source_kind=args.source_kind,
            poll_seconds=args.poll_seconds,
            cell_id=args.cell_id,
            relay_id=args.relay_id,
        )
        print(json.dumps(relay.update_once(), indent=2, sort_keys=True))
        return 0
    if args.command == "watch":
        relay = LineagePlaybackRelay(
            source_root=args.source_root,
            output_root=args.output_root,
            source_kind=args.source_kind,
            poll_seconds=args.poll_seconds,
            cell_id=args.cell_id,
            relay_id=args.relay_id,
        )
        relay.watch(max_updates=args.max_updates)
        return 0
    if args.command == "validate-manifest":
        manifest = validate_relay_manifest(args.manifest)
        print(json.dumps({"ok": True, "sequence_number": manifest["sequence_number"], "boundary": manifest["latest_exported_playback_boundary"]}, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
