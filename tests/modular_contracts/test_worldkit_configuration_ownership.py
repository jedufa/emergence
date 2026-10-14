from __future__ import annotations

import base64
import hashlib
import importlib
import json
import pickle
import sys
from dataclasses import MISSING, fields, replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASELINE_COMMIT = "fb00e803ee0603b1645ba72de197008511193e3d"
BASELINE_FIXTURE = ROOT / "tests/fixtures/release_baseline/worldkit_configuration_fb00e803.json"
BASELINE_FIXTURE_SHA256 = "1d6f1e86d3c38fac604b30341066fa83b1d0f8e8bd33b03bbebb464df7cc2b58"
CONFIGS = (
    ("env_config", "EnvConfig"),
    ("env_config_biomes", "EnvBiomeConfig"),
    ("env_config_energy_economy", "EnvEnergyEconomyConfig"),
    ("env_config_gru_cost", "EnvGruCostConfig"),
    ("env_config_health_combat", "EnvHealthCombatConfig"),
    ("env_config_injury", "EnvInjuryConfig"),
    ("env_config_reproduction", "EnvReproductionConfig"),
    ("env_config_scar", "EnvScarConfig"),
    ("env_config_tier1_spatial_ecology", "EnvTier1SpatialEcologyConfig"),
    ("env_config_tier2_spatial_droughts", "EnvTier2SpatialDroughtsConfig"),
    ("env_config_tier3_migration_pressure", "EnvTier3MigrationPressureConfig"),
    ("env_config_world", "EnvWorldConfig"),
)


def _baseline_fixture():
    raw = BASELINE_FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASELINE_FIXTURE_SHA256
    fixture = json.loads(raw)
    assert fixture["schema_version"] == "worldkit_configuration_baseline_v1"
    assert fixture["source_commit"] == BASELINE_COMMIT
    assert fixture["source_paths"] == [f"engine/env/{module_stem}.py" for module_stem, _ in CONFIGS]
    return fixture


def _field_default(field):
    if field.default is not MISSING:
        return field.default
    assert field.default_factory is not MISSING
    return field.default_factory()


def _field_shape(config_class):
    return [
        {
            "name": field.name,
            "type": str(field.type),
            "default_origin": "factory" if field.default is MISSING else "value",
            "default": _field_default(field),
        }
        for field in fields(config_class)
    ]


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _canonical_class(module_stem: str, class_name: str):
    module_name = "emergence_worldkit.config" if module_stem == "env_config" else "emergence_worldkit.config_sections"
    return getattr(importlib.import_module(module_name), class_name)


def test_fields_types_and_defaults_match_pre_extraction_baseline():
    fixture = _baseline_fixture()
    for module_stem, class_name in CONFIGS:
        current = _canonical_class(module_stem, class_name)
        baseline = fixture["configurations"][module_stem]
        assert baseline["class"] == class_name
        assert _canonical_json(_field_shape(current)) == _canonical_json(baseline["fields"]), module_stem


def test_legacy_import_orders_share_one_class_and_pickle_address(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "engine"))
    for order in (("env", "engine.env"), ("engine.env", "env")):
        for prefix in ("env", "engine.env"):
            for module_stem, _ in CONFIGS:
                sys.modules.pop(f"{prefix}.{module_stem}", None)

        first_route = {}
        for prefix in order:
            for module_stem, class_name in CONFIGS:
                legacy_module = importlib.import_module(f"{prefix}.{module_stem}")
                legacy_class = getattr(legacy_module, class_name)
                canonical = _canonical_class(module_stem, class_name)
                assert legacy_class is canonical
                assert legacy_class.__module__ == f"env.{module_stem}"
                if module_stem in first_route:
                    assert legacy_class is first_route[module_stem]
                first_route[module_stem] = legacy_class

    # Load each captured pre-migration env.* pickle through the current
    # compatibility module and canonical class object.
    fixture = _baseline_fixture()
    for module_stem, class_name in CONFIGS:
        payload = base64.b64decode(fixture["legacy_default_instance_pickles_base64"][module_stem], validate=True)
        restored = pickle.loads(payload)
        canonical = _canonical_class(module_stem, class_name)
        assert type(restored) is canonical
        assert restored.__dict__ == canonical().__dict__


def test_replace_and_existing_gridworld_consumer(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "engine"))
    EnvConfig = importlib.import_module("env.env_config").EnvConfig
    config = replace(
        EnvConfig(),
        width=4,
        height=4,
        patch_world_enable=True,
    )
    assert config.width == 4
    assert config.patch_world_enable is True

    sections = importlib.import_module("emergence_worldkit.config_sections")
    injury = replace(sections.EnvInjuryConfig(), injury_max=7.5)
    assert injury.injury_max == 7.5

    GridWorld = importlib.import_module("env.gridworld_pkg.world").GridWorld
    world = GridWorld(config, seed=17, rng=np.random.default_rng(17))
    assert world.cfg is config
    assert world.grid.shape == (4, 4)
