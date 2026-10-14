from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from engine.authoritative_state.schema import load_schema_v1
from simulation.checkpoint_bundle_v2 import (
    AUTHORITATIVE_STATE_CHECKPOINT_ADMISSION_FIELD,
    AUTHORITATIVE_STATE_CHECKPOINT_METADATA_FIELD,
    CheckpointStrictError,
    checkpoint_bundle_v2_load,
    checkpoint_bundle_v2_save,
)
from tests.evolvability.phase_e.test_e1_metrics_engine import _make_sim


def test_checkpoint_bundle_strict_requires_metrics_state_and_e1_state(tmp_path: Path) -> None:
    run_dir = tmp_path / "strict_bundle"
    sim = _make_sim(run_dir, seed=222, generations=2)
    sim.sim_cfg.phase_e_e1_enable = True
    sim.run(start_gen=0, skip_init=False)

    bundle = run_dir / "checkpoint_bundles_v2" / "bundle_gen000002_strict"
    checkpoint_bundle_v2_save(sim, str(bundle), generation_index=2, strict_replay_mode=True)

    payload = checkpoint_bundle_v2_load(str(bundle), strict=True)
    metrics_state = payload.get("metrics_state", {}) if isinstance(payload, dict) else {}
    assert isinstance(metrics_state, dict)
    assert metrics_state.get("schema_version") == "metrics_state_v1"
    assert isinstance(metrics_state.get("phase_e_e1_state"), dict)
    schema = load_schema_v1()
    identity = payload["metadata"][AUTHORITATIVE_STATE_CHECKPOINT_METADATA_FIELD]
    assert identity["state_schema_version"] == schema.root["schema_version"]
    assert identity["state_schema_sha256"] == schema.schema_hash
    assert identity["component_hashes"] == schema.root["component_hashes"]
    assert payload[AUTHORITATIVE_STATE_CHECKPOINT_ADMISSION_FIELD] == {
        "admission_schema_version": (
            "eiel_checkpoint_bundle_v2_authoritative_state_admission_v1"
        ),
        "classification": "authoritative_state_v1_exact",
        "migration": "identity",
        "admitted_as_authoritative_state_v1": True,
        "state_schema_sha256": schema.schema_hash,
        "identity_sha256": identity["identity_sha256"],
    }


def test_checkpoint_bundle_strict_fails_when_metrics_state_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "strict_missing"
    sim = _make_sim(run_dir, seed=333, generations=2)
    sim.sim_cfg.phase_e_e1_enable = True
    sim.run(start_gen=0, skip_init=False)

    src = run_dir / "checkpoint_bundles_v2" / "bundle_gen000002_strict_src"
    checkpoint_bundle_v2_save(sim, str(src), generation_index=2, strict_replay_mode=True)

    broken = run_dir / "checkpoint_bundles_v2" / "bundle_gen000002_strict_broken"
    shutil.copytree(src, broken)
    (broken / "metrics_state.json").unlink()

    with pytest.raises((CheckpointStrictError, RuntimeError), match="metrics_state.json"):
        checkpoint_bundle_v2_load(str(broken), strict=True)
