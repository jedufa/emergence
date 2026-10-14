#!/usr/bin/env python3
from __future__ import annotations

"""Bounded, process-local benchmark for Scientific Chronicle profiles."""

import argparse
import json
from pathlib import Path
import shutil
import statistics
import sys
import time


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark EIEL Scientific Chronicle profile overhead.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profile", action="append", choices=("off", "summary", "notable", "scientific_full"))
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--population", type=int, default=24)
    return parser


def _directory_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = args.project_root.resolve()
    sys.path[:0] = [str(project_root), str(project_root / "engine")]

    from env.env_config import EnvConfig
    from evolution.population_pkg.population_config import PopConfig
    from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
    from simulation.runner import SimulationRunner
    from simulation.runner_pkg.runner_config import SimConfig

    profiles = args.profile or ["off", "summary", "notable", "scientific_full"]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    def execute(profile: str, repetition: int, *, warmup: bool = False) -> dict[str, object]:
        label = f"{'warmup' if warmup else 'run'}_{profile}_{repetition:02d}"
        run_dir = output_root / "runs" / label
        shutil.rmtree(run_dir, ignore_errors=True)
        config = SimConfig(
            seed=7300 + int(repetition),
            generations=int(args.generations),
            steps_per_generation=int(args.steps),
            device="cpu",
            bank_device="cpu",
            torch_threads=1,
            behavior_log_enable=False,
            policy_deterministic_actions=True,
            policy_tick_stride=1,
        )
        if hasattr(config, "chronicle_enable"):
            config.chronicle_enable = profile != "off"
            config.chronicle_profile = profile if profile != "off" else "notable"
            config.chronicle_terminal_enable = False
            config.chronicle_run_id = f"benchmark-{profile}"
            config.allow_large_output_profile = profile == "scientific_full"
        runner = SimulationRunner(
            env_cfg=EnvConfig(
                width=12,
                height=12,
                view_radius=1,
                patch_world_enable=True,
                patch_world_observation_enable=True,
            ),
            pop_cfg=PopConfig(population_size=int(args.population)),
            sim_cfg=config,
            logger=NoOpMetricsLogger(),
            run_dir=str(run_dir),
            perf=False,
        )
        started = time.perf_counter()
        runner.run()
        elapsed = time.perf_counter() - started
        chronicle_root = run_dir / "chronicle"
        summary_path = chronicle_root / "scientific_chronicle_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        return {
            "profile": profile,
            "repetition": int(repetition),
            "elapsed_seconds": elapsed,
            "chronicle_bytes": _directory_bytes(chronicle_root),
            "event_count": int(summary.get("event_count", 0)),
            "suppressed_count": sum(int(value) for value in dict(summary.get("suppressed_counts", {})).values()),
            "episode_count": int(summary.get("episode_count", 0)),
        }

    execute("off", -1, warmup=True)
    rows: list[dict[str, object]] = []
    for repetition in range(max(1, int(args.repetitions))):
        offset = repetition % len(profiles)
        for profile in profiles[offset:] + profiles[:offset]:
            rows.append(execute(profile, repetition))

    profile_summaries: dict[str, dict[str, object]] = {}
    for profile in profiles:
        selected = [row for row in rows if row["profile"] == profile]
        elapsed = [float(row["elapsed_seconds"]) for row in selected]
        profile_summaries[profile] = {
            "repetitions": len(selected),
            "median_elapsed_seconds": statistics.median(elapsed),
            "minimum_elapsed_seconds": min(elapsed),
            "maximum_elapsed_seconds": max(elapsed),
            "median_chronicle_bytes": statistics.median(int(row["chronicle_bytes"]) for row in selected),
            "median_event_count": statistics.median(int(row["event_count"]) for row in selected),
            "median_suppressed_count": statistics.median(int(row["suppressed_count"]) for row in selected),
            "median_episode_count": statistics.median(int(row["episode_count"]) for row in selected),
        }
    off_median = float(profile_summaries.get("off", {}).get("median_elapsed_seconds", 0.0) or 0.0)
    for summary in profile_summaries.values():
        elapsed = float(summary["median_elapsed_seconds"])
        summary["overhead_vs_same_build_off_percent"] = (
            ((elapsed / off_median) - 1.0) * 100.0 if off_median > 0.0 else None
        )
    result = {
        "schema_version": "eiel_scientific_chronicle_benchmark_v1",
        "project_root": str(project_root),
        "shape": {
            "generations": int(args.generations),
            "steps_per_generation": int(args.steps),
            "population": int(args.population),
            "patch_world": True,
            "device": "cpu",
        },
        "profiles": profile_summaries,
        "rows": rows,
    }
    output_path = output_root / "benchmark.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
