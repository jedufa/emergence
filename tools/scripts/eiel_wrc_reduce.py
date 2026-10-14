"""CLI for deterministic live/replay WRC trajectory reduction."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys
from typing import Any, Mapping
import zipfile

from engine.wrc.reducer.canonical import canonical_sha256, pretty_json
from engine.wrc.reducer import (
    Observation,
    RETAIN_ALL,
    RETAIN_COMPACT,
    RETAIN_SELECTED,
    ReducerConfig,
    SourceLocator,
    WorldIdentity,
    WorldReducer,
    WrcLiveReducer,
    load_checkpoint,
    retain,
    reducer_schema,
    save_checkpoint,
)
from engine.wrc.reducer.campaign import CampaignReductionError, reduce_input
from engine.wrc.reducer.output import write_json, write_jsonl
from engine.wrc.reducer.providers import (
    ChronicleObservationProvider,
    MetricsCsvObservationProvider,
    PathTextResource,
)


def _config(path: str | None) -> ReducerConfig:
    if not path:
        return ReducerConfig()
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ERROR: cannot read reducer config {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise SystemExit("ERROR: reducer config root must be an object")
    return ReducerConfig.from_mapping(value)


def _json_scalar(raw: str) -> Any:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    if isinstance(value, (dict, list)):
        raise SystemExit("ERROR: parameter value must be a JSON scalar")
    return value


def _write_world_output(output_dir: Path, result: Mapping[str, Any], config: ReducerConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "reducer_schema.json", reducer_schema(config))
    write_json(output_dir / "world_reduction.json", dict(result))


def _world_command(args: argparse.Namespace) -> int:
    config = _config(args.config)
    identity = WorldIdentity(
        world_id=args.world_id or Path(args.metrics).parent.name,
        parameter_id=args.parameter_id,
        parameter_value=_json_scalar(args.parameter_value),
        seed=args.seed,
        ticks_per_observation=args.ticks_per_observation,
        metadata={"source": str(Path(args.metrics).resolve())},
    )
    provider = MetricsCsvObservationProvider(
        PathTextResource(args.metrics),
        ticks_per_observation=args.ticks_per_observation,
        config=config,
        required_metrics=args.required_metric,
        allow_partial_last_row=args.allow_partial,
    )
    if args.checkpoint and Path(args.checkpoint).exists():
        checkpoint = load_checkpoint(args.checkpoint)
        if checkpoint.identity.sha256 != identity.sha256:
            raise SystemExit("ERROR: checkpoint identity differs from input world")
        reducer = checkpoint.restore(config=config)
    else:
        reducer = WorldReducer(identity, config=config)
    if not reducer.finalized:
        reducer.ingest_many(
            observation
            for observation in provider
            if reducer.last_sequence is None
            or observation.sequence > reducer.last_sequence
        )
    result = reducer.finalize()
    if args.checkpoint:
        save_checkpoint(args.checkpoint, reducer, finalized=True, finalized_result=result)
    _write_world_output(Path(args.output_dir), result, config)
    print(pretty_json({"ok": True, "output_dir": str(Path(args.output_dir).resolve()), "result_sha256": result["result_sha256"]}), end="")
    return 0


def _chronicle_command(args: argparse.Namespace) -> int:
    config = _config(args.config)
    identity = WorldIdentity(
        world_id=args.world_id or Path(args.events).parent.parent.name,
        parameter_id=args.parameter_id,
        parameter_value=_json_scalar(args.parameter_value),
        seed=args.seed,
        ticks_per_observation=1,
        metadata={"source": str(Path(args.events).resolve()), "coverage": "compact_chronicle_generation_summary"},
    )
    provider = ChronicleObservationProvider(
        PathTextResource(args.events),
        config=config,
        required_metrics=args.required_metric,
    )
    reducer = WorldReducer(identity, config=config)
    reducer.ingest_many(provider)
    result = reducer.finalize()
    _write_world_output(Path(args.output_dir), result, config)
    print(pretty_json({"ok": True, "output_dir": str(Path(args.output_dir).resolve()), "result_sha256": result["result_sha256"]}), end="")
    return 0


def _reduce_command(args: argparse.Namespace) -> int:
    config = _config(args.config)
    try:
        result = reduce_input(
            args.input,
            output_dir=args.output_dir,
            row_ids=args.row,
            config=config,
            allow_partial_worlds=args.allow_partial_worlds,
            continue_on_error=not args.fail_fast,
        )
    except CampaignReductionError as exc:
        print(pretty_json({"ok": False, "error": str(exc)}), end="")
        return 2
    print(pretty_json({"ok": result["failed_row_count"] == 0, **result}), end="")
    return 0 if result["failed_row_count"] == 0 else 1


def _retain_command(args: argparse.Namespace) -> int:
    try:
        result = retain(
            args.source,
            args.output_dir,
            mode=args.mode,
            selected_paths=args.selected,
            expected_count=args.expected_count,
            expected_world_count=args.expected_world_count,
            prune_raw=args.prune_raw,
            raw_paths=args.raw,
        )
    except (OSError, ValueError) as exc:
        print(pretty_json({"ok": False, "error": str(exc)}), end="")
        return 2
    print(pretty_json({"ok": True, **result}), end="")
    return 0


def _read_json_object(path: str, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} root must be an object: {path}")
    return value


def _status_command(args: argparse.Namespace) -> int:
    try:
        reducer = WrcLiveReducer.from_snapshot_file(args.snapshot)
        live = reducer.status()
        status: dict[str, Any] = {
            "schema_version": live["schema_version"],
            "reducer_version": live["reducer_version"],
            "enabled": live["enabled"],
            "retention_mode": live["retention_mode"],
            "finalized": live["finalized"],
            "retention_outcome": "not_materialized",
            "source_bytes": None,
            "retained_bytes": None,
            "source_count": None,
            "retained_count": None,
        }
        if args.retention_manifest:
            manifest = _read_json_object(args.retention_manifest, label="retention manifest")
            if manifest.get("complete") is not True:
                raise ValueError("retention manifest is not complete")
            if manifest.get("manifest_sha256") is None:
                raise ValueError("retention manifest lacks manifest_sha256")
            manifest_without_hash = {
                key: value for key, value in manifest.items() if key != "manifest_sha256"
            }
            if canonical_sha256(manifest_without_hash) != str(manifest["manifest_sha256"]):
                raise ValueError("retention manifest hash verification failed")
            counters: dict[str, int] = {}
            for name in ("source_bytes", "retained_bytes", "source_count", "retained_count"):
                value = manifest.get(name)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValueError(f"retention manifest has invalid {name}")
                counters[name] = value
            if counters["retained_bytes"] > counters["source_bytes"]:
                raise ValueError("retention manifest retained_bytes exceeds source_bytes")
            if counters["retained_count"] > counters["source_count"]:
                raise ValueError("retention manifest retained_count exceeds source_count")
            status.update(counters)
            status["retention_outcome"] = {
                "status": "complete",
                "mode": manifest.get("mode"),
                "manifest_sha256": str(manifest["manifest_sha256"]),
                "raw_prune_complete": bool(manifest.get("raw_prune_complete", False)),
            }
    except (OSError, ValueError) as exc:
        print(pretty_json({"ok": False, "error": str(exc)}), end="")
        return 2
    print(pretty_json({"ok": True, **status}), end="")
    return 0


def _open_source_uri(uri: str):
    if uri.startswith("zip://"):
        body = uri[len("zip://") :]
        if "!/" not in body:
            raise ValueError(f"malformed ZIP source URI: {uri}")
        archive, member = body.split("!/", 1)
        bundle = zipfile.ZipFile(archive)
        raw = bundle.open(member, "r")
        wrapper = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
        return bundle, wrapper
    path = Path(uri)
    return None, path.open(newline="", encoding="utf-8", errors="replace")


def _extract_command(args: argparse.Namespace) -> int:
    index_path = Path(args.index)
    try:
        with index_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise SystemExit(f"ERROR: cannot read raw-row index: {exc}") from exc

    requested_metrics = {str(name) for name in args.metric if str(name)}
    wanted: dict[str, dict[int, set[str]]] = {}
    for row in rows:
        uri = str(row.get("source_uri") or "")
        number = row.get("source_row_number")
        if not uri or not number:
            continue
        try:
            raw_metrics = json.loads(str(row.get("metrics") or "{}"))
        except json.JSONDecodeError:
            raw_metrics = {}
        metrics = (
            {str(name) for name in raw_metrics}
            if isinstance(raw_metrics, Mapping)
            else set()
        )
        if requested_metrics:
            metrics &= requested_metrics
            if not metrics:
                continue
        wanted.setdefault(uri, {}).setdefault(int(number), set()).update(metrics)

    extracted: list[dict[str, Any]] = []
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 10
    time_fields = ("generation", "tick", "step", "sequence")
    for uri, row_requests in sorted(wanted.items()):
        bundle = None
        handle = None
        try:
            bundle, handle = _open_source_uri(uri)
            reader = csv.DictReader(handle)
            for data_index, row in enumerate(reader):
                row_number = data_index + 2
                if row_number not in row_requests:
                    continue
                selected = row_requests[row_number]
                if args.full_row:
                    payload = dict(row)
                else:
                    names = [
                        *[name for name in time_fields if name in row],
                        *sorted(name for name in selected if name in row),
                    ]
                    payload = {name: row.get(name) for name in names}
                extracted.append(
                    {
                        "source_uri": uri,
                        "source_row_number": row_number,
                        "selected_metrics": sorted(selected),
                        "row": payload,
                    }
                )
        finally:
            if handle is not None:
                handle.close()
            if bundle is not None:
                bundle.close()
    output = Path(args.output)
    write_jsonl(output, extracted)
    print(
        pretty_json(
            {
                "ok": True,
                "row_count": len(extracted),
                "full_row": bool(args.full_row),
                "metric_filter": sorted(requested_metrics),
                "output": str(output.resolve()),
            }
        ),
        end="",
    )
    return 0


def _schema_command(args: argparse.Namespace) -> int:
    print(pretty_json(reducer_schema(_config(args.config))), end="")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Reduce EIEL/WRC live or completed simulation trajectories"
    )
    subparsers = result.add_subparsers(dest="command", required=True)

    schema = subparsers.add_parser("schema", help="print the reducer contract")
    schema.add_argument("--config")
    schema.set_defaults(func=_schema_command)

    world = subparsers.add_parser("world", help="reduce one metrics.csv")
    world.add_argument("metrics")
    world.add_argument("--parameter-id", required=True)
    world.add_argument("--parameter-value", required=True, help="JSON scalar")
    world.add_argument("--seed", required=True, type=int)
    world.add_argument("--world-id")
    world.add_argument("--ticks-per-observation", type=int, default=1)
    world.add_argument("--required-metric", action="append", default=[])
    world.add_argument("--allow-partial", action="store_true")
    world.add_argument("--config")
    world.add_argument("--checkpoint", help="reducer-only checkpoint path for resume")
    world.add_argument("--output-dir", required=True)
    world.set_defaults(func=_world_command)

    chronicle = subparsers.add_parser(
        "chronicle", help="reduce Chronicle formal generation summaries"
    )
    chronicle.add_argument("events")
    chronicle.add_argument("--parameter-id", required=True)
    chronicle.add_argument("--parameter-value", required=True, help="JSON scalar")
    chronicle.add_argument("--seed", required=True, type=int)
    chronicle.add_argument("--world-id")
    chronicle.add_argument("--required-metric", action="append", default=[])
    chronicle.add_argument("--config")
    chronicle.add_argument("--output-dir", required=True)
    chronicle.set_defaults(func=_chronicle_command)

    reduce_parser = subparsers.add_parser(
        "reduce", help="reduce a WRC row/Atlas directory or ZIP"
    )
    reduce_parser.add_argument("input")
    reduce_parser.add_argument("--output-dir", required=True)
    reduce_parser.add_argument("--row", action="append", default=[])
    reduce_parser.add_argument("--allow-partial-worlds", action="store_true")
    reduce_parser.add_argument("--fail-fast", action="store_true")
    reduce_parser.add_argument("--config")
    reduce_parser.set_defaults(func=_reduce_command)

    retain_parser = subparsers.add_parser(
        "retain", help="materialize a hash-verified reducer retention transaction"
    )
    retain_parser.add_argument("source")
    retain_parser.add_argument("--output-dir", required=True)
    retain_parser.add_argument(
        "--mode", choices=(RETAIN_ALL, RETAIN_COMPACT, RETAIN_SELECTED), default=RETAIN_ALL
    )
    retain_parser.add_argument("--selected", action="append", default=[])
    retain_parser.add_argument("--raw", action="append", default=[])
    retain_parser.add_argument("--expected-count", type=int)
    retain_parser.add_argument("--expected-world-count", type=int)
    retain_parser.add_argument("--prune-raw", action="store_true")
    retain_parser.set_defaults(func=_retain_command)

    status = subparsers.add_parser(
        "status", help="report bounded live reducer and retention status"
    )
    status.add_argument("--snapshot", required=True)
    status.add_argument("--retention-manifest")
    status.set_defaults(func=_status_command)

    extract = subparsers.add_parser(
        "extract", help="extract raw CSV rows referenced by raw_row_index.csv"
    )
    extract.add_argument("--index", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument(
        "--metric",
        action="append",
        default=[],
        help=(
            "limit extraction to this metric; repeatable. By default, extract "
            "only the metrics referenced for each row by the reducer index"
        ),
    )
    extract.add_argument(
        "--full-row",
        action="store_true",
        help="extract every raw CSV column instead of the compact metric subset",
    )
    extract.set_defaults(func=_extract_command)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
