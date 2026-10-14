from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from env.env_config import EnvConfig
from env.gridworld_pkg.jungle import validate_jungle_config
from env.gridworld_pkg.patch_world import (
    MOSAIC0A_PATCH_BURST,
    MOSAIC0A_PATCH_RENEWAL,
    PATCH_WORLD_ARRAY_NAMES,
    initialize_patch_world_arrays,
    mosaic0a_map_digest,
    mosaic0a_resource_budget_contract,
    patch_array_summary,
    patch_food_metrics,
    validate_patch_world_arrays,
)


def _cfg(*, enabled: bool = True, control: str = "real_mosaic") -> EnvConfig:
    cfg = EnvConfig(width=8, height=8)
    cfg.patch_world_enable = True
    cfg.patch_food_regrow_per_step = 0.02
    cfg.patch_food_regrow_use_fertility = True
    cfg.mosaic0a_depletion_recovery_enable = bool(enabled)
    cfg.mosaic0a_class_mode = "budget_matched_clustered_v1"
    cfg.mosaic0a_cluster_size = 4
    cfg.mosaic0a_control_mode = control
    return cfg


def test_mosaic0a_defaults_are_disabled() -> None:
    cfg = EnvConfig(width=4, height=4)
    cfg.patch_world_enable = True
    arrays = initialize_patch_world_arrays(cfg, np.ones((4, 4), dtype=np.int8))

    assert cfg.mosaic0a_depletion_recovery_enable is False
    assert "mosaic0a_patch_class" in PATCH_WORLD_ARRAY_NAMES
    assert np.all(np.asarray(arrays["mosaic0a_patch_class"], dtype=np.float32) == 0.0)


def test_mosaic0a_real_overlay_creates_budget_matched_clustered_classes() -> None:
    cfg = _cfg(enabled=True)
    arrays = initialize_patch_world_arrays(cfg, np.ones((8, 8), dtype=np.int8))
    classes = np.asarray(arrays["mosaic0a_patch_class"], dtype=np.int32)
    burst = classes == MOSAIC0A_PATCH_BURST
    renewal = classes == MOSAIC0A_PATCH_RENEWAL
    contract = mosaic0a_resource_budget_contract(cfg, classes=classes)
    digest = mosaic0a_map_digest(classes)

    assert int(np.sum(burst)) == 32
    assert int(np.sum(renewal)) == 32
    assert contract["budget_matched"] is True
    assert contract["burst_budget_per_cell"] == pytest.approx(contract["renewal_budget_per_cell"])
    assert digest["burst_cell_count"] == 32
    assert digest["renewal_cell_count"] == 32
    assert np.all(arrays["patch_food_amount"][burst] == np.float32(1.0))
    assert np.all(arrays["patch_food_amount"][renewal] == np.float32(0.4))
    assert np.all(arrays["patch_fertility"][burst] == np.float32(0.4))
    assert np.all(arrays["patch_fertility"][renewal] == np.float32(1.0))


def test_mosaic0a_controls_separate_labels_depletion_and_recovery() -> None:
    grid = np.ones((8, 8), dtype=np.int8)
    base = initialize_patch_world_arrays(_cfg(enabled=False), grid)
    label = initialize_patch_world_arrays(_cfg(enabled=True, control="label_only_sham"), grid)
    depletion = initialize_patch_world_arrays(_cfg(enabled=True, control="depletion_only"), grid)
    recovery = initialize_patch_world_arrays(_cfg(enabled=True, control="recovery_only"), grid)
    uniform = initialize_patch_world_arrays(_cfg(enabled=True, control="budget_matched_uniform"), grid)

    assert not np.array_equal(label["mosaic0a_patch_class"], base["mosaic0a_patch_class"])
    assert np.array_equal(label["patch_food_amount"], base["patch_food_amount"])
    assert np.array_equal(label["patch_fertility"], base["patch_fertility"])
    assert not np.array_equal(depletion["patch_food_amount"], base["patch_food_amount"])
    assert np.array_equal(depletion["patch_fertility"], base["patch_fertility"])
    assert np.array_equal(recovery["patch_food_amount"], base["patch_food_amount"])
    assert not np.array_equal(recovery["patch_fertility"], base["patch_fertility"])
    assert len(set(np.asarray(uniform["patch_food_amount"]).ravel().tolist())) == 1
    assert len(set(np.asarray(uniform["patch_fertility"]).ravel().tolist())) == 1


def test_mosaic0a_metrics_report_class_level_food_and_fertility() -> None:
    cfg = _cfg(enabled=True)
    arrays = initialize_patch_world_arrays(cfg, np.ones((8, 8), dtype=np.int8))
    env = SimpleNamespace(cfg=cfg, **arrays)
    metrics = patch_food_metrics(env)
    summary = patch_array_summary(arrays)

    assert metrics["mosaic0a_burst_cell_frac"] == pytest.approx(0.5)
    assert metrics["mosaic0a_renewal_cell_frac"] == pytest.approx(0.5)
    assert metrics["mosaic0a_burst_food_amount_mean"] == pytest.approx(1.0)
    assert metrics["mosaic0a_renewal_food_amount_mean"] == pytest.approx(0.4)
    assert metrics["mosaic0a_burst_fertility_mean"] == pytest.approx(0.4)
    assert metrics["mosaic0a_renewal_fertility_mean"] == pytest.approx(1.0)
    assert summary["mosaic0a_patch_class"]["max"] == pytest.approx(float(MOSAIC0A_PATCH_RENEWAL))


def test_mosaic0a_validation_rejects_disabled_patch_world_and_mixed_overlays() -> None:
    no_patch = _cfg(enabled=True)
    no_patch.patch_world_enable = False
    with pytest.raises(ValueError, match="requires patch_world_enable"):
        validate_jungle_config(no_patch)

    mixed_respec = _cfg(enabled=True)
    mixed_respec.respec0b_resource_specialization_enable = True
    with pytest.raises(ValueError, match="cannot combine with respec0b"):
        validate_jungle_config(mixed_respec)

    mixed_roz = _cfg(enabled=True)
    mixed_roz.roz2a_patch_niche_enable = True
    with pytest.raises(ValueError, match="cannot combine with roz2a"):
        validate_jungle_config(mixed_roz)

    invalid = _cfg(enabled=True)
    invalid.mosaic0a_control_mode = "unknown"
    with pytest.raises(ValueError, match="unsupported mosaic0a_control_mode"):
        validate_jungle_config(invalid)


def test_mosaic0a_patch_class_survives_patch_world_array_validation() -> None:
    cfg = _cfg(enabled=True)
    arrays = initialize_patch_world_arrays(cfg, np.ones((8, 8), dtype=np.int8))
    restored = validate_patch_world_arrays(arrays, shape=(8, 8), cfg=cfg)
    legacy_payload = {name: value for name, value in arrays.items() if name != "mosaic0a_patch_class"}
    legacy_restored = validate_patch_world_arrays(legacy_payload, shape=(8, 8), cfg=cfg)

    assert np.array_equal(restored["mosaic0a_patch_class"], arrays["mosaic0a_patch_class"])
    assert np.array_equal(legacy_restored["mosaic0a_patch_class"], arrays["mosaic0a_patch_class"])
