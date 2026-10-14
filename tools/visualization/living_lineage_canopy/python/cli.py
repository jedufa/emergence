from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters.p12b import adapt_p12b, classify_p12b
from .generator import SCENARIOS
from .schema import peak_living_count, validate_playback


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Living lineage canopy V0 tooling")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="generate a deterministic synthetic fixture")
    generate.add_argument("scenario", choices=sorted(SCENARIOS))
    generate.add_argument("--out", required=True, type=Path)
    generate.add_argument("--seed", type=int)

    validate = sub.add_parser("validate", help="validate a normalized playback JSON")
    validate.add_argument("path", type=Path)

    adapt = sub.add_parser("adapt-p12b", help="adapt partial P12B birth-event evidence")
    adapt.add_argument("root", type=Path)
    adapt.add_argument("--out", required=True, type=Path)
    adapt.add_argument("--limit-rows", type=int, default=800)

    classify = sub.add_parser("classify-p12b", help="classify P12B evidence only")
    classify.add_argument("root", type=Path)
    classify.add_argument("--out", type=Path)

    args = parser.parse_args(argv)
    if args.command == "generate":
        factory = SCENARIOS[args.scenario]
        data = factory(args.seed) if args.seed is not None else factory()
        _write_json(args.out, data)
        print(f"wrote {args.out}")
        print(f"peak_living_count={peak_living_count(data)}")
        return 0
    if args.command == "validate":
        data = json.loads(args.path.read_text())
        result = validate_playback(data)
        print(f"valid schema={data['metadata']['schema_version']} peak_living_count={peak_living_count(data)}")
        for warning in result.warnings:
            print(f"warning: {warning}")
        return 0
    if args.command == "adapt-p12b":
        data = adapt_p12b(args.root, limit_rows=args.limit_rows)
        _write_json(args.out, data)
        print(f"wrote {args.out}")
        print(f"classification={data['metadata']['adapter_classification']['classification']}")
        print(f"peak_living_count={peak_living_count(data)}")
        return 0
    if args.command == "classify-p12b":
        result = classify_p12b(args.root)
        if args.out:
            _write_json(args.out, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
