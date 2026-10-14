from __future__ import annotations

import json
from typing import Any

import pytest

from native_runtime.engine import NativeEielEngine, _compact_raw_result


PATCH_REQUIRED_REPORTING_FIELDS = (
    "patch_movement_energy_event_count",
    "patch_food_harvest_total",
    "patch_scent_emit_total",
    "patch_waste_emit_total",
    "patch_movement_energy_total",
)


def _patch_raw() -> dict[str, Any]:
    return {
        "schema_version": "eiel_native_patch_generation_lifecycle_microcycle_v1",
        "decision": "EIEL-NATIVE-PATCH-CPU-LIFECYCLE-MICROCYCLE",
        "world": "patch",
        "profile": {"world": "patch", "supported": True, "profile": "P1"},
        "native_stage_count": 4,
        "native_tick_loop_count": 2,
        "typed_event_rows": 3,
        "typed_death_event_rows": 1,
        "env_rng_padding_draws": 7,
        "patch_spawn_sync_rows": 2,
        "patch_movement_energy_event_count": 3,
        "patch_food_harvest_total": 2.5,
        "patch_scent_move_queued_total": 0.125,
        "patch_scent_gather_queued_total": 0.25,
        "patch_waste_move_queued_total": 0.375,
        "patch_waste_gather_queued_total": 0.5,
        "patch_scent_emit_total": 0.375,
        "patch_waste_emit_total": 0.875,
        "patch_movement_energy_extra": -1.5,
        "patch_movement_energy_total": -1.5,
        "energy_collected_delta": 4.75,
        "patch_food_depleted_frac": 0.25,
        "patch_food_full_frac": 0.5,
        "patch_density_hotspot_frac": 1.0,
        "stage_order": (
            "native_lifecycle_microcycle_loop",
            "patch_tick_core",
            "population_death_cull",
            "population_resident_birth_commit",
        ),
        "patch_generation_core": {"schema_version": "patch_core_v1", "tick": 1},
        "population_death_cull": {"schema_version": "death_cull_v1", "death_rows": 1},
        "population_asexual_resident_birth_commit": {
            "schema_version": "birth_commit_v1",
            "committed_birth_rows": 2,
        },
        "microcycles": [
            {"tick_offset": 0, "death_rows": 1},
            {"tick_offset": 1, "committed_birth_rows": 2},
        ],
        "blocker": "bounded_patch_profile",
    }


def test_patch_compaction_preserves_typed_reporting_and_diagnostics_exactly() -> None:
    raw = _patch_raw()

    compact = _compact_raw_result(raw)

    preserved = (
        "typed_death_event_rows",
        "env_rng_padding_draws",
        "patch_spawn_sync_rows",
        *PATCH_REQUIRED_REPORTING_FIELDS,
        "patch_scent_move_queued_total",
        "patch_scent_gather_queued_total",
        "patch_waste_move_queued_total",
        "patch_waste_gather_queued_total",
        "patch_movement_energy_extra",
        "energy_collected_delta",
        "patch_food_depleted_frac",
        "patch_food_full_frac",
        "patch_density_hotspot_frac",
        "stage_order",
        "patch_generation_core",
        "population_death_cull",
        "population_asexual_resident_birth_commit",
        "microcycles",
    )
    assert compact["world"] == "patch"
    assert compact["profile"] == "P1"
    for key in preserved:
        assert compact[key] == raw[key], key
    assert compact["patch_generation_core"] is not raw["patch_generation_core"]
    assert compact["microcycles"] is not raw["microcycles"]


@pytest.mark.parametrize(
    ("key", "value", "message"),
    (
        ("patch_movement_energy_event_count", True, "integer count"),
        ("patch_spawn_sync_rows", 1.5, "integer count"),
        ("typed_event_rows", -1, "nonnegative"),
        ("patch_food_harvest_total", "2.5", "finite float total"),
        ("patch_movement_energy_total", 1, "finite float total"),
        ("patch_scent_emit_total", float("inf"), "must be finite"),
        ("patch_waste_emit_total", float("nan"), "must be finite"),
        ("patch_food_depleted_frac", -0.01, r"must be in \[0, 1\]"),
        ("patch_food_full_frac", 1.01, r"must be in \[0, 1\]"),
        ("patch_density_hotspot_frac", float("nan"), "must be finite"),
    ),
)
def test_patch_compaction_rejects_malformed_typed_values(
    key: str,
    value: Any,
    message: str,
) -> None:
    raw = _patch_raw()
    raw[key] = value

    with pytest.raises(ValueError, match=message):
        _compact_raw_result(raw)


@pytest.mark.parametrize("missing", PATCH_REQUIRED_REPORTING_FIELDS)
def test_patch_compaction_fails_closed_for_missing_required_reporting_field(
    missing: str,
) -> None:
    raw = _patch_raw()
    raw.pop("world")
    raw.pop("profile")
    raw.pop(missing)

    with pytest.raises(ValueError, match=rf"missing required reporting fields:.*{missing}"):
        _compact_raw_result(raw, profile="P0")


def test_patch_compaction_requires_finite_profile_metadata() -> None:
    raw = _patch_raw()
    raw.pop("world")
    raw.pop("profile")

    with pytest.raises(ValueError, match="missing finite P0/P1/P2 profile metadata"):
        _compact_raw_result(raw, world="patch")

    from_profile = _compact_raw_result(raw, profile="P2")

    assert from_profile["world"] == "patch"
    assert from_profile["profile"] == "P2"


def test_patch_compaction_rejects_aliased_profile_names() -> None:
    raw = _patch_raw()
    raw["profile"] = "resident_patch_v1"

    with pytest.raises(ValueError, match="invalid raw profile declaration"):
        _compact_raw_result(raw)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    (
        ("stage_order", "patch_tick_core", "sequence of stage names"),
        ("patch_generation_core", [], "must be a mapping"),
        ("population_death_cull", 1, "must be a mapping"),
        ("population_asexual_resident_birth_commit", None, "must be a mapping"),
        ("microcycles", [{"tick_offset": 0}, "bad"], "sequence of mappings"),
    ),
)
def test_patch_compaction_rejects_malformed_structured_diagnostics(
    key: str,
    value: Any,
    message: str,
) -> None:
    raw = _patch_raw()
    raw[key] = value

    with pytest.raises(ValueError, match=message):
        _compact_raw_result(raw)


def test_patch_compaction_requires_diagnostic_declared_by_stage_order() -> None:
    raw = _patch_raw()
    raw.pop("patch_generation_core")

    with pytest.raises(ValueError, match="missing declared diagnostic payloads: patch_generation_core"):
        _compact_raw_result(raw)


def test_flat_compaction_output_is_byte_for_byte_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EIEL_NATIVE_RUNTIME_DEBUG_ROW_SAMPLE", raising=False)
    raw = {
        "schema_version": "flat_result_v1",
        "decision": "FLAT",
        "world": "flat",
        "profile": {"world": "flat"},
        "native_stage_count": "2",
        "native_tick_loop_count": "3",
        "peak_injury": "0.25",
        "patch_food_harvest_total": float("nan"),
        "blocker": "none",
    }

    encoded = json.dumps(
        _compact_raw_result(raw, world="flat"),
        separators=(",", ":"),
    ).encode("ascii")

    assert encoded == (
        b'{"schema_version":"eiel_native_generation_compact_result_v2",'
        b'"raw_result_schema_version":"flat_result_v1","decision":"FLAT",'
        b'"native_stage_count":2,"blocker":"none","native_tick_loop_count":3,'
        b'"peak_injury":0.25}'
    )


def test_production_engine_checkpoint_restore_does_not_promote_patch() -> None:
    with pytest.raises(RuntimeError, match="patch_checkpoint_route_not_qualified"):
        NativeEielEngine.from_checkpoint_projection(
            {
                "metadata": {
                    "world": "patch",
                    "tick": 0,
                    "agent_grid_runner_phase": "generation_boundary",
                },
                "tensors": {},
            }
        )


def test_production_engine_checkpoint_restore_never_defaults_world_to_flat() -> None:
    with pytest.raises(RuntimeError, match="requires exact world metadata"):
        NativeEielEngine.from_checkpoint_projection(
            {
                "metadata": {
                    "tick": 0,
                    "agent_grid_runner_phase": "generation_boundary",
                },
                "tensors": {},
            }
        )
