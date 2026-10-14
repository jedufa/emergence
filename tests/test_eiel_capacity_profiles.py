from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from engine.capacity_profiles import (
    CapacityProfileError,
    load_profile,
    validate_profile,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "registries/capacity_profiles"


def _profile() -> dict[str, object]:
    return {
        "schema_version": "eiel_capacity_profile_v1",
        "profile_id": "unit-profile-v1",
        "paths": {"runtime_root": "/tmp/unit-runtime"},
    }


def test_json_and_yaml_load_to_same_canonical_profile(tmp_path: Path) -> None:
    raw = _profile()
    json_path = tmp_path / "profile.json"
    yaml_path = tmp_path / "profile.yaml"
    json_path.write_text(json.dumps(raw), encoding="utf-8")
    yaml_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    assert load_profile(json_path) == load_profile(yaml_path) == validate_profile(raw)


def test_defaults_are_conservative_and_artifact_root_is_derived() -> None:
    canonical = validate_profile(_profile())

    assert canonical["build"] == {"max_jobs": 1}
    assert canonical["paths"] == {
        "runtime_root": "/tmp/unit-runtime",
        "artifact_root": "/tmp/unit-runtime/artifacts",
    }
    assert canonical["validation"]["max_parallel_steps"] == 1
    assert canonical["validation"]["failure_policy"] == "fail_fast"
    assert canonical["campaign"] == {"workers": 1}
    assert canonical["benchmark"]["device_selector"] == "cpu"
    assert canonical["benchmark"]["objective"] == "correctness"


def test_loading_has_no_environment_or_filesystem_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _profile()
    runtime = tmp_path / "must-not-exist"
    profile["paths"] = {"runtime_root": str(runtime)}
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    monkeypatch.setenv("EIEL_RUNTIME_ROOT", str(tmp_path / "environment-root"))
    before = dict(os.environ)

    canonical = load_profile(path)

    assert canonical["paths"]["runtime_root"] == str(runtime)
    assert not runtime.exists()
    assert dict(os.environ) == before


def test_example_profiles_load_and_are_not_user_specific() -> None:
    conservative = load_profile(REGISTRY / "conservative_cpu_v1.yaml")
    developer = load_profile(REGISTRY / "developer_local_v1.json")

    assert conservative["benchmark"]["device_selector"] == "cpu"
    assert developer["build"]["max_jobs"] == 4
    rendered = json.dumps([conservative, developer])
    assert "/home/" not in rendered
    assert "jedufa" not in rendered


@pytest.mark.parametrize(
    ("suffix", "payload"),
    [
        (
            ".json",
            '{"schema_version":"eiel_capacity_profile_v1",'
            '"schema_version":"duplicate","profile_id":"x"}',
        ),
        (
            ".yaml",
            "schema_version: eiel_capacity_profile_v1\n"
            "schema_version: duplicate\nprofile_id: x\n",
        ),
    ],
)
def test_loader_rejects_duplicate_fields(
    tmp_path: Path, suffix: str, payload: str
) -> None:
    path = tmp_path / f"duplicate{suffix}"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(CapacityProfileError, match="duplicate"):
        load_profile(path)


@pytest.mark.parametrize(
    ("suffix", "payload"),
    [
        (
            ".json",
            '{"schema_version":"eiel_capacity_profile_v1","profile_id":"x",'
            '"benchmark":{"memory_headroom_bytes":NaN}}',
        ),
        (
            ".yaml",
            "schema_version: eiel_capacity_profile_v1\nprofile_id: x\n"
            "benchmark: {memory_headroom_bytes: .inf}\n",
        ),
    ],
)
def test_loader_rejects_non_finite_numbers(
    tmp_path: Path, suffix: str, payload: str
) -> None:
    path = tmp_path / f"nonfinite{suffix}"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(CapacityProfileError, match="non-finite|integer"):
        load_profile(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda profile: profile.update(extra=True), "unsupported fields"),
        (lambda profile: profile.update(schema_version="v2"), "schema_version"),
        (lambda profile: profile.update(build={"max_jobs": 0}), "build.max_jobs"),
        (lambda profile: profile.update(paths={"runtime_root": "relative"}), "absolute"),
        (lambda profile: profile.update(validation={"failure_policy": "ignore"}), "failure_policy"),
        (lambda profile: profile.update(campaign={"workers": True}), "campaign.workers"),
        (lambda profile: profile.update(benchmark={"device_selector": "gpu"}), "device_selector"),
        (lambda profile: profile.update(benchmark={"objective": "speed"}), "objective"),
    ],
)
def test_profile_fails_closed_on_invalid_policy(mutate, message: str) -> None:
    profile = _profile()
    mutate(profile)
    with pytest.raises(CapacityProfileError, match=message):
        validate_profile(profile)


def test_explicit_fields_override_defaults() -> None:
    profile = _profile()
    profile.update(
        build={"max_jobs": 8},
        validation={
            "max_parallel_steps": 3,
            "capture_bytes": 0,
            "failure_policy": "continue",
        },
        campaign={"workers": 6},
        benchmark={
            "batch_cells": 12,
            "warmups": 2,
            "repetitions": 7,
            "device_selector": "cuda:1",
            "objective": "latency",
            "memory_headroom_bytes": 0,
        },
    )

    canonical = validate_profile(profile)

    assert canonical["build"]["max_jobs"] == 8
    assert canonical["validation"]["capture_bytes"] == 0
    assert canonical["campaign"]["workers"] == 6
    assert canonical["benchmark"]["device_selector"] == "cuda:1"


def test_uuid_device_selector_is_supported() -> None:
    profile = _profile()
    profile["benchmark"] = {"device_selector": "uuid:GPU-stable-123"}

    canonical = validate_profile(profile)

    assert canonical["benchmark"]["device_selector"] == "uuid:GPU-stable-123"
