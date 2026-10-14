#!/usr/bin/env python3
from __future__ import annotations

"""CLI for Scientific Chronicle analysis and read-only companion generation."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = PROJECT_ROOT / "engine"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from simulation.chronicle.compare import compare_chronicles  # noqa: E402
from simulation.chronicle.analytics import search_run, write_group_exports, write_run_exports  # noqa: E402
from simulation.chronicle.genome_attribution import (  # noqa: E402
    load_outcome_jsonl,
    load_source_jsonl,
    write_attribution_outputs,
)
from simulation.chronicle.genome_readiness import write_readiness_outputs  # noqa: E402
from simulation.chronicle.observer import ObserverConfig, write_observer_outputs  # noqa: E402
from simulation.chronicle.cast import observer_cast_regenerate  # noqa: E402
from simulation.chronicle.documentary import write_documentary_outputs  # noqa: E402
from simulation.chronicle.live_observer import (  # noqa: E402
    LiveObserverConfig,
    catch_up_approved_root,
    catch_up_live_observer,
    follow_live_observer,
)
from simulation.chronicle.config import ChronicleConfig, storage_preflight  # noqa: E402
from simulation.chronicle.retrospective import regenerate_chronicle  # noqa: E402
from simulation.chronicle.query import filter_events  # noqa: E402
from simulation.chronicle.schema import canonical_json  # noqa: E402
from simulation.chronicle.storage import read_jsonl  # noqa: E402
from simulation.chronicle.trace import (  # noqa: E402
    estimate_trace_storage,
    make_trace_request,
    plan_trace_windows,
    produce_fixture_trace,
    query_trace,
    storage_projection_table,
    trace_from_chronicle_fixture,
    write_trace_outputs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate, preflight, or compare EIEL Scientific Chronicles.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight", help="Estimate Chronicle event and storage bounds.")
    preflight.add_argument("--profile", choices=("off", "summary", "notable", "scientific_full", "debug_trace"), default="notable")
    preflight.add_argument("--generations", type=int, required=True)
    preflight.add_argument("--cells", type=int, default=1)
    preflight.add_argument("--max-events-per-generation", type=int, default=32)
    preflight.add_argument("--allow-large-output", action="store_true")
    regenerate = subparsers.add_parser("regenerate", help="Regenerate a Chronicle without mutating its source run.")
    regenerate.add_argument("run_dir", type=Path)
    regenerate.add_argument("output_dir", type=Path)
    compare = subparsers.add_parser("compare", help="Compare two structured run Chronicles.")
    compare.add_argument("run_a", type=Path)
    compare.add_argument("run_b", type=Path)
    compare.add_argument("output_dir", type=Path)
    query = subparsers.add_parser("query", help="Filter a formal event feed deterministically.")
    query.add_argument("events_jsonl", type=Path)
    query.add_argument("--event-type", action="append", default=[])
    query.add_argument("--category", action="append", default=[])
    query.add_argument("--priority", action="append", default=[])
    query.add_argument("--lineage-id", action="append", default=[])
    query.add_argument("--subject-id", action="append", default=[])
    query.add_argument("--region-id", action="append", default=[])
    query.add_argument("--generation-start", type=int)
    query.add_argument("--generation-end", type=int)
    query.add_argument("--minimum-scientific-importance", type=int, default=0)
    analytics_run = subparsers.add_parser("analytics-run", help="Export deterministic read-only analytics for one Chronicle.")
    analytics_run.add_argument("run_dir", type=Path)
    analytics_run.add_argument("output_dir", type=Path)
    analytics_group = subparsers.add_parser("analytics-group", help="Export an explicitly selected multi-run summary.")
    analytics_group.add_argument("output_dir", type=Path)
    analytics_group.add_argument("run_dirs", type=Path, nargs="+")
    analytics_group.add_argument("--group-id", required=True)
    analytics_group.add_argument("--selection-mode", choices=("explicit_operator_selection", "declared_group"), required=True)
    analytics_group.add_argument("--selection-reason", required=True)
    analytics_group.add_argument("--pooling-key", action="append", default=[])
    analytics_search = subparsers.add_parser("analytics-search", help="Search exact Chronicle fields.")
    analytics_search.add_argument("run_dir", type=Path)
    analytics_search.add_argument("--id")
    analytics_search.add_argument("--agent-id", type=int)
    analytics_search.add_argument("--lineage-id")
    analytics_search.add_argument("--region-id")
    analytics_search.add_argument("--tag")
    analytics_search.add_argument("--event-type")
    analytics_search.add_argument("--episode-family")
    analytics_search.add_argument("--generation-start", type=int)
    analytics_search.add_argument("--generation-end", type=int)
    analytics_search.add_argument("--importance-min", type=int)
    analytics_search.add_argument("--importance-max", type=int)
    analytics_search.add_argument("--limit", type=int, default=1000)
    genome = subparsers.add_parser("genome-attribution", help="Project passive authoritative genome observations into GEN1-F0 outputs.")
    genome.add_argument("source_jsonl", type=Path)
    genome.add_argument("output_dir", type=Path)
    genome.add_argument("--outcomes-jsonl", type=Path)
    genome.add_argument("--include-full-genomes", action="store_true")
    genome.add_argument("--acknowledge-large-output", action="store_true")
    readiness = subparsers.add_parser(
        "genome-attribution-readiness",
        help="Audit explicit existing artifacts for deterministic GEN1 retrospective readiness.",
    )
    readiness.add_argument("output_dir", type=Path)
    readiness.add_argument("--source-root", type=Path, action="append", default=[])
    readiness.add_argument("--fixture-root", type=Path, action="append", default=[])
    readiness.add_argument("--selection-reason", required=True)
    readiness.add_argument("--maximum-records-per-source", type=int, default=100_000)
    observer = subparsers.add_parser(
        "observer-generate",
        help="Generate a deterministic public Observer companion from one Scientific Chronicle.",
    )
    observer.add_argument("run_dir", type=Path)
    observer.add_argument("output_dir", type=Path)
    observer.add_argument(
        "--audience-profile",
        choices=("public_plain", "field_observer", "dry_humour", "documentary", "cosmic_creator"),
        default="public_plain",
    )
    observer.add_argument(
        "--storage-profile",
        choices=("observer_summary", "observer_notable", "observer_full"),
        default="observer_notable",
    )
    observer.add_argument("--minimum-public-interest", type=int, default=0)
    observer.add_argument("--minimum-visual-interest", type=int, default=0)
    observer.add_argument(
        "--minimum-live-priority",
        choices=("critical", "high", "medium", "low", "background"),
        default="background",
    )
    observer.add_argument("--category", action="append", default=[])
    observer.add_argument("--event-type", action="append", default=[])
    observer.add_argument("--episode-family", action="append", default=[])
    observer.add_argument("--generation-start", type=int)
    observer.add_argument("--generation-end", type=int)
    observer.add_argument("--agent-id", type=int, action="append", default=[])
    observer.add_argument("--lineage-id", action="append", default=[])
    observer.add_argument("--region-id", action="append", default=[])
    observer.add_argument("--max-messages-per-generation", type=int, default=8)
    observer.add_argument("--repeated-theme-cooldown", type=int, default=1)
    observer.add_argument("--feed-budget", type=int, default=10_000)
    observer.add_argument("--maximum-audio-sentence-length", type=int, default=180)
    observer.add_argument("--gen1-root", type=Path)
    observer.add_argument("--acknowledge-large-output", action="store_true")
    cast = subparsers.add_parser(
        "observer-cast-regenerate",
        help="Regenerate an Observer CAST1 companion with stable names, retrospective cast, and callbacks.",
    )
    cast.add_argument("run_dir", type=Path)
    cast.add_argument("output_dir", type=Path)
    cast.add_argument("--observer-dir", type=Path)
    cast.add_argument(
        "--audience-profile",
        choices=("dry_humour", "cosmic_creator"),
        default="cosmic_creator",
    )
    cast.add_argument("--max-featured-agents", type=int, default=5)
    documentary = subparsers.add_parser(
        "documentary-generate",
        help="Generate a deterministic POD1 two-host documentary package from one completed Chronicle run.",
    )
    documentary.add_argument("run_dir", type=Path)
    documentary.add_argument("output_dir", type=Path)
    documentary.add_argument("--observer-dir", type=Path)
    documentary.add_argument("--cast-dir", type=Path)
    documentary.add_argument(
        "--generation-mode",
        choices=("deterministic_template", "external_ai_import", "llm_adapter_disabled"),
        default="deterministic_template",
    )
    live = subparsers.add_parser(
        "observer-watch",
        help="Read a Chronicle externally and publish a crash-safe live Observer companion.",
    )
    live.add_argument("source_root", type=Path)
    live.add_argument("output_root", type=Path)
    live.add_argument("--approved-root", type=Path, action="append", default=[])
    live.add_argument(
        "--audience-profile",
        choices=("public_plain", "field_observer", "dry_humour", "documentary", "cosmic_creator"),
        action="append",
        default=[],
    )
    live.add_argument("--poll-interval", type=float, default=1.0)
    live.add_argument("--maximum-source-records", type=int, default=100_000)
    live.add_argument("--maximum-simultaneous-bubbles", type=int, default=8)
    live.add_argument("--maximum-transport-page", type=int, default=200)
    live.add_argument("--follow", action="store_true")
    live.add_argument("--discover-runs", action="store_true")
    live.add_argument("--maximum-polls", type=int, help=argparse.SUPPRESS)
    trace_preflight = subparsers.add_parser(
        "trace-preflight",
        help="Estimate bounded selective Simulation Trace storage.",
    )
    trace_preflight.add_argument("--field-group", action="append", default=[])
    trace_preflight.add_argument("--step-records", type=int, default=150)
    trace_preflight.add_argument("--windows", type=int, default=1)
    trace_preflight.add_argument("--agents-per-step", type=int, default=1)
    trace_preflight.add_argument("--projection-table", action="store_true")
    trace_fixture = subparsers.add_parser(
        "trace-fixture",
        help="Generate deterministic TRACE1-F0 fixture artifacts from a Chronicle run.",
    )
    trace_fixture.add_argument("run_dir", type=Path)
    trace_fixture.add_argument("output_dir", type=Path)
    trace_fixture.add_argument(
        "--profile",
        choices=("trace_triggered", "trace_subjects", "trace_sampled", "trace_diagnostic_full"),
        default="trace_triggered",
    )
    trace_fixture.add_argument("--steps-before", type=int, default=4)
    trace_fixture.add_argument("--steps-after", type=int, default=4)
    trace_fixture.add_argument("--max-windows", type=int, default=8)
    trace_query = subparsers.add_parser("trace-query", help="Filter TRACE1-F0 step records.")
    trace_query.add_argument("trace_dir", type=Path)
    trace_query.add_argument("--window-id")
    trace_query.add_argument("--generation", type=int)
    trace_query.add_argument("--step", type=int)
    trace_query.add_argument("--action")
    trace_query.add_argument("--field-group")
    trace_query.add_argument("--source-completeness")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        config = ChronicleConfig(
            enabled=args.profile != "off",
            profile=args.profile,
            max_stored_events_per_generation=max(1, int(args.max_events_per_generation)),
            allow_large_output=bool(args.allow_large_output),
        )
        config.validate()
        result = storage_preflight(config, generations=args.generations, planned_cells=args.cells)
    elif args.command == "regenerate":
        result = regenerate_chronicle(args.run_dir, args.output_dir)
    elif args.command == "compare":
        result = compare_chronicles(args.run_a, args.run_b, args.output_dir)
    elif args.command == "query":
        result = filter_events(
            read_jsonl(args.events_jsonl),
            event_types=args.event_type,
            categories=args.category,
            priorities=args.priority,
            lineage_ids=args.lineage_id,
            subject_ids=args.subject_id,
            region_ids=args.region_id,
            generation_start=args.generation_start,
            generation_end=args.generation_end,
            minimum_scientific_importance=args.minimum_scientific_importance,
        )
    elif args.command == "analytics-run":
        result = write_run_exports(args.run_dir, args.output_dir)
    elif args.command == "analytics-group":
        result = write_group_exports(
            args.run_dirs,
            {
                "group_id": args.group_id,
                "selection_mode": args.selection_mode,
                "selection_reason": args.selection_reason,
                "pooling_keys": args.pooling_key,
            },
            args.output_dir,
        )
    elif args.command == "analytics-search":
        query_values = {
            key: value
            for key, value in {
                "id": args.id,
                "agent_id": args.agent_id,
                "lineage_id": args.lineage_id,
                "region_id": args.region_id,
                "tag": args.tag,
                "event_type": args.event_type,
                "episode_family": args.episode_family,
                "generation_start": args.generation_start,
                "generation_end": args.generation_end,
                "importance_min": args.importance_min,
                "importance_max": args.importance_max,
            }.items()
            if value is not None and value != ""
        }
        result = search_run(args.run_dir, query_values, limit=args.limit)
    elif args.command == "genome-attribution":
        result = write_attribution_outputs(
            load_source_jsonl(args.source_jsonl),
            args.output_dir,
            outcomes=load_outcome_jsonl(args.outcomes_jsonl),
            include_full_genomes=bool(args.include_full_genomes),
            acknowledge_large_output=bool(args.acknowledge_large_output),
        )
    elif args.command == "genome-attribution-readiness":
        result = write_readiness_outputs(
            args.source_root,
            args.output_dir,
            selection_reason=args.selection_reason,
            fixture_roots=args.fixture_root,
            maximum_records_per_source=args.maximum_records_per_source,
        )
    elif args.command == "trace-preflight":
        result = (
            storage_projection_table()
            if args.projection_table
            else estimate_trace_storage(
                field_groups=args.field_group,
                step_records=args.step_records,
                windows=args.windows,
                agents_per_step=args.agents_per_step,
            )
        )
    elif args.command == "trace-fixture":
        result = trace_from_chronicle_fixture(
            args.run_dir,
            args.output_dir,
            profile=args.profile,
            steps_before=args.steps_before,
            steps_after=args.steps_after,
            max_windows=args.max_windows,
        )
    elif args.command == "trace-query":
        result = {
            "schema_version": "eiel_selective_trace_query_result_v1",
            "items": query_trace(
                read_jsonl(args.trace_dir / "trace_step_records.jsonl"),
                window_id=args.window_id,
                generation=args.generation,
                step=args.step,
                action=args.action,
                field_group=args.field_group,
                source_completeness=args.source_completeness,
            ),
        }
    elif args.command == "observer-generate":
        result = write_observer_outputs(
            args.run_dir,
            args.output_dir,
            config=ObserverConfig(
                audience_profile=args.audience_profile,
                storage_profile=args.storage_profile,
                minimum_public_interest=args.minimum_public_interest,
                minimum_visual_interest=args.minimum_visual_interest,
                minimum_live_priority=args.minimum_live_priority,
                categories=tuple(args.category),
                event_types=tuple(args.event_type),
                episode_families=tuple(args.episode_family),
                generation_start=args.generation_start,
                generation_end=args.generation_end,
                agent_ids=tuple(args.agent_id),
                lineage_ids=tuple(args.lineage_id),
                region_ids=tuple(args.region_id),
                max_messages_per_generation=args.max_messages_per_generation,
                repeated_theme_cooldown=args.repeated_theme_cooldown,
                feed_budget=args.feed_budget,
                maximum_audio_sentence_length=args.maximum_audio_sentence_length,
                acknowledge_large_output=bool(args.acknowledge_large_output),
            ),
            gen1_root=args.gen1_root,
        )
    elif args.command == "observer-cast-regenerate":
        result = observer_cast_regenerate(
            args.run_dir,
            args.output_dir,
            observer_dir=args.observer_dir,
            audience_profile=args.audience_profile,
            max_featured_agents=args.max_featured_agents,
        )
    elif args.command == "documentary-generate":
        result = write_documentary_outputs(
            args.run_dir,
            args.output_dir,
            observer_dir=args.observer_dir,
            cast_dir=args.cast_dir,
            generation_mode=args.generation_mode,
        )
    else:
        profiles = tuple(args.audience_profile) or (
            "public_plain",
            "field_observer",
            "dry_humour",
            "documentary",
            "cosmic_creator",
        )
        config = LiveObserverConfig(
            audience_profiles=profiles,
            poll_interval_seconds=args.poll_interval,
            maximum_source_records=args.maximum_source_records,
            maximum_simultaneous_bubbles=args.maximum_simultaneous_bubbles,
            maximum_transport_page=args.maximum_transport_page,
        )
        approved = tuple(args.approved_root) or (args.source_root,)
        if args.discover_runs:
            if args.follow:
                _parser().error("--follow and --discover-runs are mutually exclusive")
            result = catch_up_approved_root(args.source_root, args.output_root, config=config)
        elif args.follow:
            result = follow_live_observer(
                args.source_root,
                args.output_root,
                approved_roots=approved,
                config=config,
                maximum_polls=args.maximum_polls,
            )
        else:
            result = catch_up_live_observer(
                args.source_root,
                args.output_root,
                approved_roots=approved,
                config=config,
            )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
