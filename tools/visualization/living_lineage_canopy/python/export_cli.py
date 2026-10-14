from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .artifact_readers.checkpoint_bundle_v2 import projection_paths_for_cell
from .production_exporter import cross_check_gv2, cross_check_p12b, export_gv2_checkpoint_run, export_p12b_partial
from .provenance import validate_provenance_binding, verify_source_file_hashes, write_canonical_json
from .schema import peak_living_count, validate_playback
from .source_audit import audit_many, audit_source


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EIEL lineage playback exporter V0")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="audit candidate source roots")
    audit.add_argument("--source", action="append", required=True, type=Path)
    audit.add_argument("--out", required=True, type=Path)
    audit.add_argument("--overwrite", action="store_true")

    export = sub.add_parser("export", help="export strongest supported normalized playback")
    export.add_argument("--source", required=True, type=Path)
    export.add_argument("--out", required=True, type=Path)
    export.add_argument("--provenance-out", required=True, type=Path)
    export.add_argument("--summary-out", type=Path)
    export.add_argument("--cross-check-out", type=Path)
    export.add_argument("--generated-at", required=True)
    export.add_argument("--limit-rows", type=int)
    export.add_argument("--cell-id")
    export.add_argument("--overwrite", action="store_true")

    validate = sub.add_parser("validate", help="validate playback and provenance")
    validate.add_argument("--playback", required=True, type=Path)
    validate.add_argument("--provenance", type=Path)
    validate.add_argument("--verify-source", action="store_true", help="also verify source file hashes recorded in provenance")

    summarize = sub.add_parser("summarize", help="summarize a normalized playback")
    summarize.add_argument("--playback", required=True, type=Path)
    summarize.add_argument("--out", type=Path)
    summarize.add_argument("--overwrite", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "audit":
        result = audit_many(args.source)
        write_canonical_json(args.out, result, overwrite=args.overwrite)
        print(json.dumps(result["selected_source"], indent=2, sort_keys=True))
        return 0
    if args.command == "export":
        command = " ".join(sys.argv if argv is None else ["export_cli", *argv])
        if projection_paths_for_cell(args.source, args.cell_id):
            playback, provenance, summary = export_gv2_checkpoint_run(
                args.source,
                cell_id=args.cell_id,
                generated_at=args.generated_at,
                command=command,
            )
            cross_check_factory = lambda: cross_check_gv2(args.source, playback, cell_id=args.cell_id)
        else:
            playback, provenance, summary = export_p12b_partial(
                args.source,
                limit_rows=args.limit_rows,
                generated_at=args.generated_at,
                command=command,
            )
            cross_check_factory = lambda: cross_check_p12b(args.source, playback)
        write_canonical_json(args.out, playback, overwrite=args.overwrite)
        write_canonical_json(args.provenance_out, provenance, overwrite=args.overwrite)
        if args.summary_out:
            write_canonical_json(args.summary_out, summary, overwrite=args.overwrite)
        if args.cross_check_out:
            cross_check = cross_check_factory()
            write_canonical_json(args.cross_check_out, cross_check, overwrite=args.overwrite)
            if not cross_check["ok"]:
                raise SystemExit("cross-check mismatch: " + "; ".join(cross_check["mismatches"]))
        print(f"wrote {args.out}")
        print(f"classification={summary['classification']}")
        print(f"agents={summary['agent_count']} lineages={summary['lineage_count']} branches={summary['branch_count']}")
        print(f"output_sha256={summary['output_sha256']}")
        return 0
    if args.command == "validate":
        playback = _load_json(args.playback)
        result = validate_playback(playback)
        if args.provenance:
            provenance = _load_json(args.provenance)
            binding = validate_provenance_binding(playback, provenance)
            print(f"provenance binding ok output_sha256={binding['output_sha256']}")
            if args.verify_source:
                source_result = verify_source_file_hashes(provenance)
                print(f"source hashes ok checked={len(source_result['checked'])}")
        elif args.verify_source:
            raise SystemExit("--verify-source requires --provenance")
        print(f"valid schema={playback['metadata']['schema_version']} peak_living_count={peak_living_count(playback)}")
        for warning in result.warnings:
            print(f"warning: {warning}")
        return 0
    if args.command == "summarize":
        playback = _load_json(args.playback)
        validate_playback(playback)
        summary = {
            "schema_version": "eiel_lineage_playback_summary_v1",
            "dataset_id": playback["metadata"].get("dataset_id"),
            "source_classification": playback["metadata"].get("source_classification"),
            "semantics_mode": playback["metadata"].get("semantics_mode"),
            "time_unit": playback["metadata"].get("time_unit"),
            "branch_semantics": playback["metadata"].get("branch_semantics"),
            "first_tick": playback["metadata"].get("first_tick"),
            "last_tick": playback["metadata"].get("last_tick"),
            "agents": len(playback["agents"]),
            "lineages": len(playback["lineages"]),
            "branches": len(playback["branches"]),
            "peak_living_count": peak_living_count(playback),
            "warnings": playback["metadata"].get("warnings", []),
        }
        if args.out:
            write_canonical_json(args.out, summary, overwrite=args.overwrite)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
