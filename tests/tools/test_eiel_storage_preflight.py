from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from experiments.harness.output_storage import write_storage_summary
from tools.scripts.eiel_runtime_storage_prune import build_prune_plan


def _write_registry(path: Path) -> None:
    path.write_text(
        """
harness_api: 1
version: 1
name: storage_preflight_fixture
description: "storage preflight fixture"
intent:
  question: "fixture"
matrix_order:
  - scenario
  - seed
axes:
  scenario:
    values:
      - base
    flags:
      base:
        - --knob
        - sim_cfg.checkpoint_bundle_v2_period_gen=1
        - --knob
        - sim_cfg.lineage_reproduction_funnel_write_events=true
seeds:
  - 101
cli_defaults:
  - - --generations
    - "1"
  - - --steps
    - "1"
  - - --pop
    - "4"
  - - --grid_size
    - "4"
  - - --device
    - cpu
  - - --bank_device
    - cpu
naming:
  run_id_format: "{name}__scenario_{scenario}__seed{seed:06d}"
  run_tag_prefix: "storage_preflight_fixture"
  run_tag_format: "{run_tag_prefix}_{scenario}_seed{seed:06d}"
expected_artifacts:
  - manifest.json
  - run_config.csv
""".lstrip(),
        encoding="utf-8",
    )


def test_harness_compact_default_writes_storage_plan_and_overrides_heavy_registry_knobs(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    _write_registry(registry)
    runs_root = tmp_path / "runs"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.harness.run_experiments",
            "--registry",
            str(registry),
            "--runs_root",
            str(runs_root),
            "--batch_tag",
            "storage_fixture",
            "--dry_run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    storage_plans = list(runs_root.rglob("storage_plan.json"))
    assert len(storage_plans) == 1
    plan = json.loads(storage_plans[0].read_text(encoding="utf-8"))
    assert plan["ok"] is True
    assert plan["risk"] == "compact"
    row = plan["rows"][0]
    assert row["settings"]["checkpoint_period_gen"] == 0
    assert row["settings"]["lineage_funnel_write_events"] is False


def test_harness_compact_funnel_keeps_generation_funnel_without_events(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    _write_registry(registry)
    runs_root = tmp_path / "runs"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.harness.run_experiments",
            "--registry",
            str(registry),
            "--runs_root",
            str(runs_root),
            "--batch_tag",
            "storage_fixture",
            "--campaign_output_profile",
            "compact_funnel",
            "--dry_run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    plan = json.loads(next(runs_root.rglob("storage_plan.json")).read_text(encoding="utf-8"))
    assert plan["ok"] is True
    row = plan["rows"][0]
    assert row["settings"]["checkpoint_period_gen"] == 0
    assert row["settings"]["lineage_funnel_write_events"] is False
    assert row["settings"]["lineage_funnel_write_per_generation"] is True


def test_harness_full_replay_without_ack_fails_storage_preflight(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    _write_registry(registry)
    runs_root = tmp_path / "runs"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.harness.run_experiments",
            "--registry",
            str(registry),
            "--runs_root",
            str(runs_root),
            "--batch_tag",
            "storage_fixture",
            "--campaign_output_profile",
            "full_replay",
            "--dry_run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    plan = json.loads(next(runs_root.rglob("storage_plan.json")).read_text(encoding="utf-8"))
    assert plan["ok"] is False
    assert any("checkpoint_period_1_requires_allow_generation_checkpointing" in item for item in plan["errors"])


def test_storage_summary_and_prune_plan_classify_heavy_outputs(tmp_path: Path) -> None:
    root = tmp_path / "run"
    (root / "metrics").mkdir(parents=True)
    (root / "metrics" / "lineage_reproduction_funnel_events.jsonl").write_text("x\n", encoding="utf-8")
    first = root / "checkpoint_bundles_v2" / "bundle_gen000001_strict"
    final = root / "checkpoint_bundles_v2" / "bundle_gen000002_strict"
    first.mkdir(parents=True)
    final.mkdir(parents=True)
    (first / "manifest.json").write_text("{}", encoding="utf-8")
    (final / "manifest.json").write_text("{}", encoding="utf-8")

    summary = write_storage_summary(root / "storage_summary.json", root, output_profile="full_replay")
    plan = build_prune_plan(root, {"event_jsonl", "intermediate_checkpoints"})

    assert summary["event_jsonl_file_count"] == 1
    assert summary["checkpoint_bundle_count"] == 2
    selected = {row["path"] for row in plan["selected"]}
    assert "metrics/lineage_reproduction_funnel_events.jsonl" in selected
    assert "checkpoint_bundles_v2/bundle_gen000001_strict" in selected
    assert "checkpoint_bundles_v2/bundle_gen000002_strict" not in selected
