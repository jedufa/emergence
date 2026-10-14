from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from engine.hapb_rngv2.cuda import native_sources_exist
from engine.hapb_rngv2.route import (
    DEFAULT_HAPB_RNG_SEMANTICS,
    LEGACY_HAPB_FRAG_ROUTE_ID,
    V2_HAPB_FRAG_ROUTE_ID,
    apply_hapb_rng_route,
    hapb_rngv2_env_enabled,
    resolve_hapb_rng_route,
)
from engine.perf.cuda_architecture_b_hapb_v1 import (
    HAPB_DIRECT_ENTRYPOINT_V1,
    resolve_hapb_frag_rng_route_v1,
)


ROOT = Path(__file__).resolve().parents[1]


def test_default_hapb_frag_route_remains_legacy() -> None:
    resolved = resolve_hapb_rng_route()
    assert DEFAULT_HAPB_RNG_SEMANTICS == 1
    assert resolved["default"] is True
    assert resolved["route_id"] == LEGACY_HAPB_FRAG_ROUTE_ID
    assert resolved["host_tapes_allowed"] is True
    assert resolved["entrypoint"] == HAPB_DIRECT_ENTRYPOINT_V1
    shipped = resolve_hapb_frag_rng_route_v1()
    assert shipped["route_id"] == LEGACY_HAPB_FRAG_ROUTE_ID
    assert shipped["default"] is True


def test_rngv2_route_is_explicit_and_default_off() -> None:
    resolved = resolve_hapb_rng_route(hapb_rng_semantics_version=2)
    assert resolved["default"] is False
    assert resolved["route_id"] == V2_HAPB_FRAG_ROUTE_ID
    assert resolved["host_tapes_allowed"] is False
    cfg = SimpleNamespace()
    apply_hapb_rng_route(cfg, hapb_rng_semantics_version=2)
    assert cfg.rng_semantics_version == 2
    assert hapb_rngv2_env_enabled() is True
    apply_hapb_rng_route(cfg, hapb_rng_semantics_version=1)
    assert hapb_rngv2_env_enabled() is False


def test_source_builder_v2_activates_hapb_policy_refill_hook() -> None:
    from engine.perf.cuda_architecture_b_direct_resident_source_builder_v1 import (
        birth_state_capacity_overrides_v1,
    )

    apply_hapb_rng_route(SimpleNamespace(), hapb_rng_semantics_version=1)
    assert hapb_rngv2_env_enabled() is False
    cfg = SimpleNamespace(rng_semantics_version=2)
    overrides = birth_state_capacity_overrides_v1(
        cfg,
        ordinary_births_enabled=True,
        resident_policy_episode_capacity=8,
        turnover_refill_capacity=0,
    )
    assert overrides["scalar_gauss_raw_capacity"] == 0
    assert hapb_rngv2_env_enabled() is True
    apply_hapb_rng_route(cfg, hapb_rng_semantics_version=1)
    assert hapb_rngv2_env_enabled() is False


def test_tensorworld_sources_are_unmodified() -> None:
    tracked = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "status",
            "--porcelain",
            "--",
            "engine/tensor_world",
            "engine/tw_rngv2",
        ],
        text=True,
    )
    assert tracked.strip() == ""
    native = ROOT / "engine" / "hapb_rngv2" / "native"
    tw_copies = list(native.glob("philox.cuh")) + list(native.glob("twdmath.cuh"))
    assert tw_copies == []
    assert native_sources_exist() is True
    header = (native / "hapb_rngv2.cuh").read_text(encoding="utf-8")
    assert '#include "../../tw_rngv2/native/tw_rngv2.cuh"' in header
