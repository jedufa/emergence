from __future__ import annotations

import hashlib
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest
import torch

from agents.genome import Genome
from evolution.reproduction import crossover_genome
import perf.recomb72_gpu_native as gpu_native
from perf.recomb72_gpu_native import (
    AUDITED_LOCI,
    advance_recomb72_gpu_state_v1,
    initialize_recomb72_gpu_state_v1,
    load_recomb72_gpu_extension,
    restore_recomb72_gpu_state_v1,
    summarize_recomb72_gpu_state_v1,
)


MODES = (
    "current_authoritative",
    "parent_copy_per_locus",
    "bounded_interpolation",
    "whole_package_single_parent",
)


def _production_genome(loci: list[float]) -> Genome:
    genome = Genome(
        hidden_size=0,
        exploration_bias=loci[0],
        aggression_bias=loci[1],
        reproduction_threshold=40.0,
        mutation_rate=0.0,
        action_temperature=loci[2],
        use_gru=False,
        memory_dim=0,
        drive_inertia=loci[4],
        drive_noise=0.02,
        drive_logit_gain=1.0,
        pred_lr=loci[5],
        developmental_program={},
    )
    genome.resource_handling_axis = loci[3]
    return genome


def test_gpu_primitive_rejects_cpu_execution() -> None:
    with pytest.raises(RuntimeError, match="REQUIRES_CUDA"):
        initialize_recomb72_gpu_state_v1(
            seeds=[1],
            modes=[MODES[0]],
            patterns=["single_extreme_scalar"],
            device="cpu",
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_prebuilt_loader_fails_before_compilation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gpu_native,
        "load_inline",
        lambda *args, **kwargs: pytest.fail("runtime compilation was attempted"),
    )
    with pytest.raises(RuntimeError, match="PREBUILT_REQUIRED"):
        load_recomb72_gpu_extension(require_prebuilt=True)
    with pytest.raises(RuntimeError, match="PREBUILT_MISSING"):
        load_recomb72_gpu_extension(
            prebuilt_extension_path=tmp_path / "absent.so",
            require_prebuilt=True,
        )
    artifact = tmp_path / "invalid.so"
    artifact.write_bytes(b"not an extension")
    with pytest.raises(RuntimeError, match="SHA256_MISMATCH"):
        load_recomb72_gpu_extension(
            prebuilt_extension_path=artifact,
            prebuilt_sha256="0" * 64,
            require_prebuilt=True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_b1_founders_turnover_metrics_and_checkpoint(
    tmp_path: Path,
) -> None:
    build = tmp_path / "build"
    state = initialize_recomb72_gpu_state_v1(
        seeds=[991],
        modes=["whole_package_single_parent"],
        patterns=["correlated_multi_trait_packages"],
        build_directory=build,
    )
    assert state.loci.is_cuda
    assert state.alive.sum().item() == 64
    assert state.generation.cpu().tolist() == [0]
    founders = state.loci[0, :64].cpu()
    lower = torch.tensor([-1.0, -2.0, 0.25, -1.0, 0.0, 0.0])
    upper = torch.tensor([1.0, 2.0, 2.5, 1.0, 0.999, 1.0])
    torch.testing.assert_close(founders[:32], lower.expand(32, 6))
    torch.testing.assert_close(founders[32:], upper.expand(32, 6))
    torch.testing.assert_close(
        state.ancestry[0, :64].cpu(), torch.eye(64), rtol=0, atol=0
    )
    g0 = summarize_recomb72_gpu_state_v1(state, build_directory=build)
    assert g0["intact_package_fraction"].cpu().tolist() == [1.0]
    assert g0["effective_founder_ancestry_count"].cpu().tolist() == [64.0]
    assert g0["dominant_founder_ancestry_share"].cpu().tolist() == [1.0 / 64.0]

    result = advance_recomb72_gpu_state_v1(
        state, generations=2, build_directory=build
    )
    assert state.generation.cpu().tolist() == [2]
    assert state.alive.sum().item() == 64
    assert state.next_id.cpu().tolist() == [128]
    assert result["mutation_events"].count_nonzero().item() == 0
    assert set(result["inheritance_draw_count"].cpu().flatten().tolist()) == {1}
    assert torch.all(
        result["parent_agent_ids"] != result["mate_agent_ids"]
    ).item()
    donors = result["donor_by_locus"].cpu()
    assert set(donors.flatten().tolist()) <= {0, 1}
    assert torch.all(donors == donors[..., :1]).item()
    assert torch.all(result["variance_population"] >= 0).item()
    assert torch.all(
        (result["fraction_near_either_extreme"] >= 0)
        & (result["fraction_near_either_extreme"] <= 1)
    ).item()
    assert torch.all(
        (result["intact_package_fraction"] >= 0)
        & (result["intact_package_fraction"] <= 1)
    ).item()
    assert torch.all(
        result["effective_founder_ancestry_count"] >= 1
    ).item()
    ancestry_sums = state.ancestry[0, state.alive[0]].sum(dim=1)
    torch.testing.assert_close(
        ancestry_sums, torch.ones_like(ancestry_sums), rtol=0, atol=1.0e-6
    )

    restored = restore_recomb72_gpu_state_v1(state.checkpoint_payload())
    expected = advance_recomb72_gpu_state_v1(
        state, generations=1, build_directory=build
    )
    resumed = advance_recomb72_gpu_state_v1(
        restored, generations=1, build_directory=build
    )
    for name in (
        "loci", "alive", "agent_ids", "ancestry", "next_id", "generation"
    ):
        torch.testing.assert_close(
            getattr(restored, name), getattr(state, name), rtol=0, atol=0
        )
    for name in expected:
        torch.testing.assert_close(resumed[name], expected[name], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_prebuilt_extension_reloads_exact_artifact_without_jit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = load_recomb72_gpu_extension(build_directory=tmp_path / "build")
    artifact = Path(built.__file__).resolve()
    artifact_sha256 = _sha256(artifact)
    module_name = artifact.name.split(".", 1)[0]

    gpu_native._MODULE = None
    gpu_native._MODULE_ORIGIN = None
    sys.modules.pop(module_name, None)
    monkeypatch.setattr(
        gpu_native,
        "load_inline",
        lambda *args, **kwargs: pytest.fail("runtime compilation was attempted"),
    )

    prebuilt = load_recomb72_gpu_extension(
        prebuilt_extension_path=artifact,
        prebuilt_sha256=artifact_sha256,
        require_prebuilt=True,
    )
    assert Path(prebuilt.__file__).resolve() == artifact
    state = initialize_recomb72_gpu_state_v1(
        seeds=[88],
        modes=["current_authoritative"],
        patterns=["single_extreme_scalar"],
        prebuilt_extension_path=artifact,
        prebuilt_sha256=artifact_sha256,
        require_prebuilt=True,
    )
    result = advance_recomb72_gpu_state_v1(
        state,
        generations=1,
        prebuilt_extension_path=artifact,
        prebuilt_sha256=artifact_sha256,
        require_prebuilt=True,
    )
    assert result["mutation_events"].count_nonzero().item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("mode", MODES)
def test_gpu_four_modes_use_six_loci_fixed_budget_and_mode_independent_pairs(
    mode: str,
    tmp_path: Path,
) -> None:
    build = tmp_path / "build"
    state = initialize_recomb72_gpu_state_v1(
        seeds=[1234],
        modes=[mode],
        patterns=["correlated_multi_trait_packages"],
        build_directory=build,
    )
    result = advance_recomb72_gpu_state_v1(
        state, generations=1, build_directory=build
    )
    assert tuple(state.loci.shape) == (1, 256, len(AUDITED_LOCI))
    assert tuple(result["parent_loci"].shape) == (1, 1, 32, 6)
    assert tuple(result["mate_loci"].shape) == (1, 1, 32, 6)
    assert tuple(result["child_loci"].shape) == (1, 1, 32, 6)
    assert tuple(result["child_agent_ids"].shape) == (1, 1, 32)
    expected_budget = {
        "current_authoritative": 0,
        "parent_copy_per_locus": 6,
        "bounded_interpolation": 6,
        "whole_package_single_parent": 1,
    }[mode]
    assert result["inheritance_draw_count"].cpu().unique().tolist() == [
        expected_budget
    ]
    assert result["mutation_events"].count_nonzero().item() == 0
    assert state.alive.sum().item() == 64
    if mode == "current_authoritative":
        torch.testing.assert_close(
            result["child_loci"],
            0.5 * (result["parent_loci"] + result["mate_loci"]),
            rtol=0,
            atol=0,
        )
        assert set(result["donor_by_locus"].cpu().flatten().tolist()) == {-1}
    elif mode == "whole_package_single_parent":
        donors = result["donor_by_locus"]
        expected = torch.where(
            donors == 0,
            result["parent_loci"],
            result["mate_loci"],
        )
        torch.testing.assert_close(
            result["child_loci"], expected, rtol=0, atol=0
        )
        assert torch.all(donors == donors[..., :1]).item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_parent_schedule_is_mode_independent(tmp_path: Path) -> None:
    build = tmp_path / "build"
    state = initialize_recomb72_gpu_state_v1(
        seeds=[77] * 4,
        modes=list(MODES),
        patterns=["single_extreme_scalar"] * 4,
        build_directory=build,
    )
    result = advance_recomb72_gpu_state_v1(
        state, generations=1, build_directory=build
    )
    for index in range(1, 4):
        torch.testing.assert_close(
            result["parent_agent_ids"][0],
            result["parent_agent_ids"][index],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            result["mate_agent_ids"][0],
            result["mate_agent_ids"][index],
            rtol=0,
            atol=0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_current_mode_matches_production_crossover_for_every_g1_event(
    tmp_path: Path,
) -> None:
    state = initialize_recomb72_gpu_state_v1(
        seeds=[101],
        modes=["current_authoritative"],
        patterns=["correlated_multi_trait_packages"],
        build_directory=tmp_path / "build",
    )
    result = advance_recomb72_gpu_state_v1(
        state,
        generations=1,
        build_directory=tmp_path / "build",
    )
    config = SimpleNamespace(
        recomb72_audit_enable=True,
        recomb72_inheritance_mode="current_authoritative",
        recomb72_mutation_off=True,
        resource_handling_representation_enable=True,
        inheritance_profile="legacy_midpoint_v1",
        genome_c1_enable=True,
        genome_c2_enable=False,
        mutation_weather_enabled=False,
        policy_family_head_residual_genome_enable=False,
    )
    expected: list[list[float]] = []
    for parent, mate in zip(
        result["parent_loci"][0, 0].cpu().tolist(),
        result["mate_loci"][0, 0].cpu().tolist(),
        strict=True,
    ):
        child = crossover_genome(
            _production_genome(parent),
            _production_genome(mate),
            random.Random(0),
            sim_cfg=config,
        )
        event = child._recomb72_inheritance_event
        assert event["production_current_path"] is True
        expected.append([event["child"][locus] for locus in AUDITED_LOCI])
    torch.testing.assert_close(
        result["child_loci"][0, 0].cpu(),
        torch.tensor(expected, dtype=torch.float32),
        rtol=0,
        atol=0,
    )
