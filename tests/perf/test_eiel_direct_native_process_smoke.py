from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_direct_native_subprocess_exits_cleanly_with_prebuilt_extension(tmp_path: Path) -> None:
    extension = os.environ.get("EIEL_NATIVE_RUNTIME_EXTENSION", "").strip()
    if not extension:
        pytest.skip("EIEL_NATIVE_RUNTIME_EXTENSION is required for direct native subprocess smoke")
    if not Path(extension).is_file():
        pytest.skip(f"EIEL_NATIVE_RUNTIME_EXTENSION is not a file: {extension}")

    runs_dir = tmp_path / "runs"
    run_id = "direct_native_process_smoke"
    env = os.environ.copy()
    env["EIEL_NATIVE_RUNTIME_EXTENSION"] = extension
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(PROJECT_ROOT),
            str(PROJECT_ROOT / "engine"),
            env.get("PYTHONPATH", ""),
        ]
    )
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "engine" / "main.py"),
        "--seed",
        "1001",
        "--gen",
        "1",
        "--steps",
        "8",
        "--pop",
        "16",
        "--grid_size",
        "8",
        "--device",
        "cpu",
        "--bank_device",
        "cpu",
        "--torch_threads",
        "1",
        "--policy_tick_stride",
        "2",
        "--checkpoint_bundle_v2_period_gen",
        "0",
        "--checkpoint_bundle_v2_strict_replay_mode",
        "1",
        "--run_id",
        run_id,
        "--runs_dir",
        str(runs_dir),
        "--allow_existing_run_dir",
        "--native_runtime_enable",
        "--native_runtime_cxx26_scope_lock_enable",
        "--native_runtime_fixed_policy_family",
        "mlp",
        "--native_runtime_max_births_per_generation",
        "128",
    ]
    for knob in (
        "sim_cfg.maintenance_enable=false",
        "sim_cfg.chemistry_functional_coupling_enable=false",
            "sim_cfg.genome_c1_enable=false",
            "sim_cfg.genome_c2_enable=false",
        "sim_cfg.force_prediction_disabled=true",
        "sim_cfg.typed_lifecycle_public_output_route_neutral_enable=true",
        "sim_cfg.lineage_observability_enable=true",
        "sim_cfg.behavior_log_enable=true",
        "sim_cfg.runnercore1_write_summary=false",
        "sim_cfg.native_runtime_write_summary=false",
        "sim_cfg.developmental_trace_write_enable=false",
        "sim_cfg.phase_e_e1_enable=false",
        "sim_cfg.phase_e_e2_enable=false",
        "sim_cfg.a3_recovery_taxonomy_enable=false",
        "sim_cfg.lineage_qd_archive_enable=false",
        "sim_cfg.campaign_output_profile=compact_default",
        "env_cfg.reproduce_requires_mate=false",
        "env_cfg.reproduce_try_prob=1.0",
        "env_cfg.food_spawn_per_step=4",
        "env_cfg.max_food=24",
        "env_cfg.view_radius=1",
        "genome.use_gru=false",
        "genome.memory_dim=0",
    ):
        cmd.extend(["--knob", knob])

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    run_dir = runs_dir / run_id
    metrics = run_dir / "metrics.csv"
    assert metrics.is_file()
    assert len(metrics.read_text(encoding="utf-8").splitlines()) >= 2
    for relative_path in (
        "behaviors.csv",
        "metrics/lineage_descriptors.jsonl",
        "metrics/lineage_summary.csv",
        "metrics/typed_lifecycle_events.csv",
        "metrics/typed_lifecycle_events.jsonl",
        "viz/metrics_summary.csv",
        "viz/run_status.json",
    ):
        assert (run_dir / relative_path).is_file(), relative_path
    assert not (run_dir / "native_runtime_failure.json").exists()
    assert "hidden_fallback" not in result.stdout + result.stderr
