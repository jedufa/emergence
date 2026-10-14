from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import hashlib

import pytest

from emergence_contracts import (
    SCIENCE_GENOTYPE_ATTRIBUTION_CAPABILITIES_V1,
    SCIENCE_STRATEGY_ATTRIBUTION_CAPABILITIES_V1,
    SCIENCE_STRATEGY_ATTRIBUTION_DENOMINATOR_FIELDS_V1,
    SCIENCE_STRATEGY_ATTRIBUTION_EVENT_KINDS_V1,
    SCIENCE_STRATEGY_ATTRIBUTION_SCHEMA_V1,
    canonical_json_bytes,
    canonical_sha256,
)
from emergence_contracts.adaptive_science_expedition_v1 import (
    StateClassV1,
    StateRefV1,
)
from emergence_contracts.artifacts import RunArtifactRefV1
from emergence_contracts.ids import ArtifactId, ContentDigest, SchemaVersion
from emergence_contracts.science_request_v1 import (
    validate_eiel_science_request_v1,
)
from eiel_composition.science_request_v1_compiler import (
    ScienceRequestV1CompilerError,
    _exact_natural_source_policy_bindings,
    compile_science_request_v1,
)


SEED_OWNERS = {
    "ecology",
    "founder",
    "policy",
    "mutation",
    "task_environment",
    "initialization",
    "intervention",
}

G5_ENGINEERING_ROOT = Path(
    "_spec/campaigns/v1/eiel_science_request_v1/examples"
)
G5_ENGINEERING_FIXTURES = (
    (
        "EIEL_SCIENCE_REQUEST_V1_expedition_g5_engineering_1024_w32_b64_v1.json",
        1024,
        [64] * 16,
    ),
    (
        "EIEL_SCIENCE_REQUEST_V1_expedition_g5_engineering_10000_w32_b64_v1.json",
        10000,
        [64] * 156 + [16],
    ),
)


def _artifact_ref(
    *, artifact_id: str, kind: str, schema: str, digest: str, locator: str
) -> RunArtifactRefV1:
    return RunArtifactRefV1(
        artifact_id=ArtifactId(artifact_id),
        artifact_kind=kind,
        artifact_schema_version=SchemaVersion(schema),
        locator=locator,
        content_digest=ContentDigest("sha256", digest),
        size_bytes=123,
    )


def _selected_world_request() -> dict:
    source_ref = _artifact_ref(
        artifact_id="selected-parent-v1",
        kind="checkpoint.workhorse_selected_world_v1",
        schema="eiel_workhorse_selected_world_checkpoint_v1",
        digest="1" * 64,
        locator="/retained/selected-world-checkpoint.json",
    )
    state_ref = StateRefV1(
        expedition_id="expedition-selected-v1",
        world_trajectory_id="trajectory-selected-parent-v1",
        logical_world_id=9102,
        next_absolute_tick=25,
        state_class=StateClassV1.BRANCHABLE_EXACT,
        semantic_world_sha256=ContentDigest("sha256", "2" * 64),
        control_history_sha256=ContentDigest("sha256", "3" * 64),
        source_object_sha256=source_ref.content_digest,
        source_artifact_ref=source_ref,
        exact_state_sha256=ContentDigest("sha256", "4" * 64),
        checkpoint_occurrence_sha256=ContentDigest("sha256", "5" * 64),
        rng_state_sha256=ContentDigest("sha256", "6" * 64),
        event_prefix_sha256=ContentDigest("sha256", "7" * 64),
        parent_occurrence_sha256=ContentDigest("sha256", "8" * 64),
        replay_recipe_sha256=None,
    )
    authority_ref = _artifact_ref(
        artifact_id="selected-parent-authority-v1",
        kind="checkpoint.workhorse_occurrence_projection_v2",
        schema="eiel_workhorse_row_checkpoint_occurrence_projection_v2",
        digest="9" * 64,
        locator="/retained/selected-world-occurrence-projection.json",
    )
    dimensions = {
        "environment": ["selected-parent"],
        "response": ["branch-local"],
        "replicate": [0],
    }
    return {
        "schema_version": "eiel_science_request_v1",
        "campaign_id": "EIEL-SELECTED-WORLD-CONTINUATION-V1",
        "test": "SELECTED_WORLD_CONTINUATION_V1",
        "question": {
            "description": "Exact selected-world continuation compiler fixture.",
            "no_claims": True,
        },
        "world": {
            "width": 128,
            "height": 80,
            "population": 2560,
            "capacity": 5120,
        },
        "run": {
            "ticks": 30,
            "batch": {"policy": "exact", "worlds": 1},
            "checkpoint_ticks": [28, 30],
        },
        "dimensions": dimensions,
        "world_template": {
            "controls": {
                "environment.resource.multiplier": 1.5,
                "mutation.rate": 0.25,
            }
        },
        "replicate_dimension": "replicate",
        "observe": {
            "profile": "SCIENCE_STANDARD_V1",
            "events": ["birth"],
            "state": ["population_count"],
            "runtime_counters": ["birth_count"],
            "snapshot_ticks": [25, 30],
        },
        "reduce": _reducer(
            list(dimensions),
            "environment",
            "response",
            "SELECTED_WORLD_CONTINUATION",
        ),
        "expect_worlds": 1,
        "continuation": {
            "schema_version": "eiel_science_selected_world_continuation_v1",
            "state_ref": state_ref.to_dict(),
            "authority_receipt_ref": authority_ref.to_dict(),
        },
    }


def _controls(
    *, selection: object, mutation_rate: object, mutation_step: object
) -> dict:
    return {
        "environment.patch.enabled": True,
        "environment.jungle.enabled": True,
        "environment.jungle.damage_threshold": 24,
        "environment.wdl.enabled": True,
        "environment.wdl.damage": 1,
        "environment.eed.enabled": True,
        "environment.eed.loss_shift": 5,
        "environment.resource.carrying_base": 64,
        "environment.resource.carrying_variation": 96,
        "environment.resource.growth_base": 1,
        "environment.resource.growth_variation": 3,
        "environment.resource.multiplier": 0.8,
        "environment.resource.uptake_base": 8,
        "environment.movement.cost": 1,
        "reproduction.enabled": True,
        "reproduction.energy_threshold": 100,
        "reproduction.energy_cost": 30,
        "recombination.enabled": True,
        "selection.pressure": selection,
        "mutation.enabled": "${mutation_enabled}",
        "mutation.rate": mutation_rate,
        "mutation.step": mutation_step,
    }


def _observe(*, total_ticks: int, snapshots: list[int]) -> dict:
    assert snapshots[0] == 0 and snapshots[-1] == total_ticks
    return {
        "profile": "SCIENCE_STANDARD_V1",
        "events": [
            "birth",
            "birth_parentage",
            "causal_lifecycle",
            "death",
            "lineage_ancestry",
            "mutation",
            "mutation_provenance",
        ],
        "state": [
            "compact.state_snapshots",
            "genome_identity",
            "population_count",
        ],
        "runtime_counters": [
            "birth_count",
            "compact.agent_lifetime_metrics",
            "control_readback",
            "mutation_count",
        ],
        "snapshot_ticks": snapshots,
    }


def _reducer(dimensions: list[str], axis_a: str, axis_b: str, family: str) -> dict:
    return {
        "reducer": "science.compact.world_atlas.v1",
        "configuration": {
            "analysis_dimensions": dimensions,
            "axis_a": axis_a,
            "axis_b": axis_b,
            "decision_family": family,
            "reporting_group": family,
            "tracked_policy_loci": [0, 1, 2, 3, 4, 5, 12, 19, 26, 40],
            "tracked_trait_loci": [1, 2, 3, 4, 6],
        },
    }


def _selection_window_request() -> dict:
    dimensions = {
        "pressure_onset": [0, 2500, 10000, 25000],
        "pressure_intensity": [0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
        "mutation_enabled": [False, True],
        "mutation_step": [1, 4],
        "seed": [1103, 2207, 3301, 4409, 5519, 6619, 7723, 8837],
    }
    return {
        "schema_version": "eiel_science_request_v1",
        "campaign_id": "EIEL-SELECTION-WINDOW-W32-B1024-EXAMPLE-V1",
        "test": "SELECTION_WINDOW_V1",
        "question": {
            "description": "Selection-window compiler acceptance fixture.",
            "no_claims": True,
        },
        "world": {"width": 32, "height": 32, "population": 256, "capacity": 512},
        "run": {
            "ticks": 100000,
            "batch": {"policy": "exact", "worlds": 1024},
            "checkpoint_ticks": [25000, 50000, 75000, 100000],
        },
        "dimensions": dimensions,
        "world_template": {
            "initialization": {
                "initial_population": 256,
                "initial_live_count": 256,
                "founder_profiles": ["native-broad-diverse-a-v1"],
                "exact_founder_composition": [
                    {"founder_profile_id": "native-broad-diverse-a-v1", "count": 256}
                ],
                "spatial_placement_profile": "deterministic_seeded_shuffle_v1",
                "standing_variation_profile": "broad-standing-variation-v1",
            },
            "controls": _controls(
                selection={
                    "value": "${pressure_intensity}",
                    "schedule": {
                        "kind": "step_absolute_tick",
                        "onset_tick": "${pressure_onset}",
                        "baseline": 0.0,
                    },
                },
                mutation_rate=0.03125,
                mutation_step="${mutation_step}",
            ),
            "seeds": {owner: "${seed}" for owner in sorted(SEED_OWNERS)},
        },
        "replicate_dimension": "seed",
        "observe": _observe(
            total_ticks=100000,
            snapshots=[
                0,
                100,
                250,
                500,
                1000,
                2500,
                5000,
                7500,
                10000,
                15000,
                25000,
                50000,
                75000,
                100000,
            ],
        ),
        "reduce": _reducer(
            list(dimensions),
            "pressure_onset",
            "pressure_intensity",
            "SELECTION_WINDOW",
        ),
        "expect_worlds": 1024,
    }


def _mutation_pressure_request() -> dict:
    dimensions = {
        "selection_pressure": [0.0, 0.5, 1.25, 2.0],
        "mutation_rate": [0.0, 0.0078125, 0.03125, 0.125],
        "founder_profile": ["native-neutral-v1", "native-broad-diverse-a-v1"],
        "seed": [1103, 2207, 3301, 4409, 5519, 6619, 7723, 8837],
    }
    controls = _controls(
        selection="${selection_pressure}",
        mutation_rate="${mutation_rate}",
        mutation_step=2,
    )
    controls["mutation.enabled"] = True
    controls.update(
        {
            "environment.jungle.damage_threshold": 18,
            "environment.wdl.damage": 3,
            "environment.eed.loss_shift": 4,
            "environment.resource.carrying_base": 96,
            "environment.resource.carrying_variation": 48,
            "environment.resource.growth_base": 2,
            "environment.resource.growth_variation": 2,
            "environment.resource.multiplier": 1.0,
            "environment.resource.uptake_base": 5,
            "environment.movement.cost": 4,
            "reproduction.energy_threshold": 110,
            "reproduction.energy_cost": 36,
        }
    )
    return {
        "schema_version": "eiel_science_request_v1",
        "campaign_id": "EIEL-MUTATION-PRESSURE-W128X80-MULTIBATCH-EXAMPLE-V1",
        "test": "MUTATION_PRESSURE_INTERACTION_V1",
        "question": {
            "description": "W128x80 structural multi-batch compiler acceptance fixture.",
            "no_claims": True,
        },
        "world": {
            "width": 128,
            "height": 80,
            "population": 2560,
            "capacity": 5120,
        },
        "run": {
            "ticks": 25000,
            "batch": 32,
            "checkpoint_ticks": [12500, 25000],
        },
        "dimensions": dimensions,
        "exclude": [{"selection_pressure": 0.0, "mutation_rate": 0.0}],
        "world_template": {
            "initialization": {
                "initial_population": 2560,
                "initial_live_count": 2560,
                "founder_profiles": ["${founder_profile}"],
                "exact_founder_composition": [
                    {"founder_profile_id": "${founder_profile}", "count": 2560}
                ],
                "spatial_placement_profile": "deterministic_seeded_shuffle_v1",
                "standing_variation_profile": "${founder_profile}",
            },
            "controls": controls,
            "seeds": {owner: "${seed}" for owner in sorted(SEED_OWNERS)},
        },
        "replicate_dimension": "seed",
        "observe": _observe(
            total_ticks=25000,
            snapshots=[0, 1000, 2500, 5000, 10000, 12500, 15000, 20000, 25000],
        ),
        "reduce": _reducer(
            list(dimensions),
            "selection_pressure",
            "mutation_rate",
            "MUTATION_PRESSURE_INTERACTION",
        ),
        "expect_worlds": 240,
    }


def test_selected_world_continuation_compiles_absolute_frontier_without_founder_rng() -> None:
    raw = _selected_world_request()
    normalized = validate_eiel_science_request_v1(raw)
    first = compile_science_request_v1(normalized)
    second = compile_science_request_v1(deepcopy(raw))

    assert first.to_dict() == second.to_dict()
    assert first.world_bundle.bundle_content_sha256() == (
        second.world_bundle.bundle_content_sha256()
    )
    assert first.world_bundle.world_count == 1
    world = first.world_bundle.worlds[0]
    semantic = world.semantic_inputs.to_value()
    assert semantic["time"] == {
        "authority": "absolute_tick_v1",
        "start_next_tick": 25,
        "total_ticks": 5,
        "checkpoint_ticks": [28, 30],
        "deliberate_restore_ticks": [25],
    }
    assert semantic["seed_bindings"] == {}
    assert semantic["initialization"]["schema_version"] == (
        "eiel_science_selected_world_initialization_v1"
    )
    assert semantic["initialization"]["state_ref"] == raw["continuation"][
        "state_ref"
    ]
    controls = {
        row["control_id"]: (row["resolved_value"], row["schedule"])
        for row in semantic["control_bindings"]
    }
    assert controls == {
        "environment.resource.multiplier": (1.5, {"kind": "static"}),
        "mutation.rate": (0.25, {"kind": "static"}),
    }
    source_spec = first.world_bundle.compact_source_specs.to_value()[0]
    assert source_spec["registry_bindings"]["founder_registry_sha256"] is None
    assert source_spec["registry_bindings"]["founder_profile_ids"] == []


def _world_by_dimensions(compilation, **expected):
    matches = []
    for world in compilation.world_bundle.worlds:
        dimensions = world.semantic_inputs.to_value()["dimension_bindings"]
        if all(dimensions.get(key) == value for key, value in expected.items()):
            matches.append(world)
    assert len(matches) == 1
    return matches[0]


def _control(world, control_id: str) -> dict:
    rows = world.semantic_inputs.to_value()["control_bindings"]
    return next(row for row in rows if row["control_id"] == control_id)


def test_selection_window_compiles_exact_b1024_with_typed_step_schedules() -> None:
    request = validate_eiel_science_request_v1(_selection_window_request())
    first = compile_science_request_v1(request)
    second = compile_science_request_v1(request)

    assert first.world_bundle.world_count == 1024
    assert first.world_bundle.bundle_content_sha256() == (
        second.world_bundle.bundle_content_sha256()
    )
    plan = first.compilation_receipt["nominal_batch_plan"]
    assert plan["policy"] == "EXACT"
    assert plan["nominal_only"] is True
    assert plan["execution_capability_claimed"] is False
    assert [row["world_count"] for row in plan["batches"]] == [1024]
    assert plan["batches"][0]["world_ids"] == [
        world.world_id for world in first.world_bundle.worlds
    ]
    plan_core = {key: value for key, value in plan.items() if key != "plan_sha256"}
    assert plan["plan_sha256"] == canonical_sha256(plan_core).to_dict()

    onset_zero = _world_by_dimensions(
        first,
        pressure_onset=0,
        pressure_intensity=0.5,
        mutation_enabled=True,
        mutation_step=4,
        seed=1103,
    )
    pressure = _control(onset_zero, "selection.pressure")
    assert pressure["resolved_value"] == 0.5
    assert pressure["schedule"] == {
        "kind": "piecewise_constant_absolute_ticks",
        "segments": [{"start_absolute_tick": 0, "value": 0.5}],
    }
    semantic = onset_zero.semantic_inputs.to_value()
    assert semantic["world"]["world_profile_id"] == "eiel-w32-native-science-v1"
    assert set(semantic["seed_bindings"]) == SEED_OWNERS
    assert set(semantic["seed_bindings"].values()) == {1103}
    assert _control(onset_zero, "mutation.enabled")["resolved_value"] is True
    assert type(_control(onset_zero, "mutation.step")["resolved_value"]) is int
    assert onset_zero.science.to_value() == {
        "test_id": "SELECTION_WINDOW",
        "test_version": "1",
        "study_id": "EIEL-SELECTION-WINDOW-W32-B1024-EXAMPLE-V1",
        "arm_id": onset_zero.science.to_value()["arm_id"],
        "replicate": 1103,
    }

    later = _world_by_dimensions(
        first,
        pressure_onset=2500,
        pressure_intensity=0.5,
        mutation_enabled=True,
        mutation_step=4,
        seed=1103,
    )
    assert _control(later, "selection.pressure")["schedule"] == {
        "kind": "piecewise_constant_absolute_ticks",
        "segments": [
            {"start_absolute_tick": 0, "value": 0.0},
            {"start_absolute_tick": 2500, "value": 0.5},
        ],
    }
    null_pressure = _world_by_dimensions(
        first,
        pressure_onset=2500,
        pressure_intensity=0.0,
        mutation_enabled=True,
        mutation_step=4,
        seed=1103,
    )
    assert _control(null_pressure, "selection.pressure")["schedule"] == {
        "kind": "static"
    }


def test_reproduction_funnel_requirement_binds_native_strategy_contract() -> None:
    raw = _selection_window_request()
    raw["reduce"]["configuration"].update(
        {
            "require_reproduction_funnel_denominators": True,
        }
    )
    request = validate_eiel_science_request_v1(raw)
    compilation = compile_science_request_v1(request)
    world = compilation.world_bundle.worlds[0]
    reducer = world.reducers.to_value()[0]
    configuration = reducer["configuration"]
    binding = configuration["strategy_attribution"]

    assert binding["schema_version"] == SCIENCE_STRATEGY_ATTRIBUTION_SCHEMA_V1
    assert binding["event_kinds"] == list(SCIENCE_STRATEGY_ATTRIBUTION_EVENT_KINDS_V1)
    assert binding["denominator_fields"] == list(
        SCIENCE_STRATEGY_ATTRIBUTION_DENOMINATOR_FIELDS_V1
    )
    assert binding["strategy_ids"] == ["native-broad-diverse-a-v1"]
    assert binding["policy_id_to_strategy_id"] == (
        compilation.compilation_receipt["registry_bindings"]["founder_policy_bindings"]
    )
    assert set(SCIENCE_STRATEGY_ATTRIBUTION_CAPABILITIES_V1).issubset(
        reducer["required_observations"]
    )


def test_direct_genotype_reproductive_attribution_binds_exact_native_contract() -> None:
    raw = _selection_window_request()
    raw["reduce"]["configuration"][
        "require_direct_genotype_reproductive_attribution"
    ] = True

    compilation = compile_science_request_v1(validate_eiel_science_request_v1(raw))
    reducer = compilation.world_bundle.worlds[0].reducers.to_value()[0]
    binding = reducer["configuration"]["strategy_attribution"]
    assert binding["direct_genotype_attribution"] is True
    assert binding["genotype_identity_mode"] == "exact_tracked_genotype_state_id_v1"
    assert binding["genotype_denominator_fields"] == list(
        SCIENCE_STRATEGY_ATTRIBUTION_DENOMINATOR_FIELDS_V1
    )
    assert set(SCIENCE_GENOTYPE_ATTRIBUTION_CAPABILITIES_V1).issubset(
        reducer["required_observations"]
    )


def test_reproductive_attribution_requirements_default_to_unconfigured_reducer() -> None:
    request = validate_eiel_science_request_v1(_selection_window_request())
    compilation = compile_science_request_v1(request)
    reducer = compilation.world_bundle.worlds[0].reducers.to_value()[0]
    assert "strategy_attribution" not in reducer["configuration"]


def test_exact_natural_source_policy_bindings_are_content_bound() -> None:
    policy_hex = "00" * 224
    bindings = _exact_natural_source_policy_bindings(
        [
            {
                "exact_natural_source_states": [
                    {"genome": {"policy_hex": policy_hex}},
                ]
            }
        ]
    )
    expected = "sha256:" + hashlib.sha256(bytes.fromhex(policy_hex)).hexdigest()
    assert bindings == {expected: expected}


def test_w128_exclusion_founder_substitution_and_nominal_multibatch_are_exact() -> None:
    request = validate_eiel_science_request_v1(_mutation_pressure_request())
    compilation = compile_science_request_v1(request)

    assert compilation.world_bundle.world_count == 240
    plan = compilation.compilation_receipt["nominal_batch_plan"]
    assert plan["policy"] == "AT_MOST"
    assert [row["world_count"] for row in plan["batches"]] == [32] * 7 + [16]
    assert plan["execution_capability_claimed"] is False
    assert compilation.compilation_receipt["execution_capability_claimed"] is False

    for world in compilation.world_bundle.worlds:
        semantic = world.semantic_inputs.to_value()
        assert semantic["world"]["world_profile_id"] == (
            "eiel-science-world-shape-w128x80-p2560-c5120-v1"
        )
        dimensions = semantic["dimension_bindings"]
        assert not (
            dimensions["selection_pressure"] == 0.0
            and dimensions["mutation_rate"] == 0.0
        )
        founder = dimensions["founder_profile"]
        initialization = semantic["initialization"]
        assert initialization["founder_profiles"] == [founder]
        assert initialization["exact_founder_composition"] == [
            {"founder_profile_id": founder, "count": 2560}
        ]
        assert initialization["standing_variation_profile"] == founder
        assert set(semantic["seed_bindings"]) == SEED_OWNERS


@pytest.mark.parametrize(
    ("name", "logical_worlds", "batch_sizes"),
    G5_ENGINEERING_FIXTURES,
)
def test_g5_engineering_fixtures_compile_exact_b64_packing_without_claims(
    name: str,
    logical_worlds: int,
    batch_sizes: list[int],
) -> None:
    raw = json.loads((G5_ENGINEERING_ROOT / name).read_text(encoding="utf-8"))
    request = validate_eiel_science_request_v1(raw)
    compilation = compile_science_request_v1(request)

    assert compilation.world_bundle.world_count == logical_worlds
    receipt = compilation.compilation_receipt
    plan = receipt["nominal_batch_plan"]
    assert plan["policy"] == "AT_MOST"
    assert plan["requested_worlds_per_batch"] == 64
    assert plan["logical_world_count"] == logical_worlds
    assert plan["nominal_batch_count"] == len(batch_sizes)
    assert plan["batch_sizes"] == batch_sizes
    assert [row["world_count"] for row in plan["batches"]] == batch_sizes
    assert sum(batch_sizes) == logical_worlds
    assert all(size == 64 for size in batch_sizes[:-1])
    assert batch_sizes[-1] == (64 if logical_worlds == 1024 else 16)
    assert plan["nominal_only"] is True
    assert plan["execution_capability_claimed"] is False
    assert receipt["time_authority"] == "absolute_tick_v1"
    assert receipt["qualification_authority_effect"] is False
    assert b"generation" not in canonical_json_bytes(receipt).lower()

    for world in compilation.world_bundle.worlds:
        semantic = world.semantic_inputs.to_value()
        dimensions = semantic["dimension_bindings"]
        assert semantic["time"] == {
            "authority": "absolute_tick_v1",
            "start_next_tick": 0,
            "total_ticks": 64,
            "checkpoint_ticks": [64],
            "deliberate_restore_ticks": [],
        }
        assert semantic["seed_bindings"]["policy"] == dimensions["condition"]
        assert {
            semantic["seed_bindings"][owner]
            for owner in SEED_OWNERS - {"policy"}
        } == {dimensions["replicate"]}
        assert world.science.to_value()["replicate"] == dimensions["replicate"]
        controls = semantic["control_bindings"]
        assert len(controls) == 1
        assert controls[0]["control_id"] == "mutation.enabled"
        assert controls[0]["resolved_value"] is dimensions["control_enabled"] is True
        assert controls[0]["schedule"] == {"kind": "static"}

    sample = compilation.world_bundle.worlds[0]
    assert sample.acquisition.to_value() == {
        "profile_id": "SCIENCE_STANDARD_V1",
        "event_observations": [
            "birth",
            "birth_parentage",
            "causal_lifecycle",
            "death",
            "lineage_ancestry",
            "mutation",
            "mutation_provenance",
        ],
        "state_observations": [
            "compact.state_snapshots",
            "genome_identity",
            "population_count",
        ],
        "runtime_counters": [
            "birth_count",
            "compact.agent_lifetime_metrics",
            "control_readback",
            "mutation_count",
        ],
        "snapshot_ticks": [0, 64],
        "windows": [],
    }
    reducer = sample.reducers.to_value()[0]
    assert reducer["reducer_id"] == "science.compact.world_atlas.v1"
    assert reducer["configuration"]["decision_family"] == (
        "EXPEDITION_G5_ENGINEERING_PACKING"
    )


def test_compilation_receipt_binds_request_bundle_registries_and_absolute_ticks() -> (
    None
):
    request = validate_eiel_science_request_v1(_mutation_pressure_request())
    compilation = compile_science_request_v1(request)
    receipt = compilation.compilation_receipt
    receipt_core = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }

    assert receipt["request_sha256"] == request.request_sha256().to_dict()
    assert receipt["bundle_content_sha256"] == (
        compilation.world_bundle.bundle_content_sha256().to_dict()
    )
    assert receipt["compiler_component_sha256"] == (
        compilation.world_bundle.compiler_component_sha256.to_dict()
    )
    assert receipt["receipt_sha256"] == canonical_sha256(receipt_core).to_dict()
    assert receipt["time_authority"] == "absolute_tick_v1"
    assert receipt["qualification_authority_effect"] is False
    registries = receipt["registry_bindings"]
    assert registries["founder_profile_ids"] == [
        "native-broad-diverse-a-v1",
        "native-neutral-v1",
    ]
    assert set(registries["founder_policy_bindings"].values()) == {
        "native-broad-diverse-a-v1",
        "native-neutral-v1",
    }
    assert set(registries["control_descriptor_sha256s"]) == set(
        request.to_dict()["world_template"]["controls"]
    )
    assert registries["reducer_descriptor_sha256"]["algorithm"] == "sha256"
    assert b"generation" not in canonical_json_bytes(receipt)


def test_compiler_rejects_partial_placeholder_even_for_mapping_input() -> None:
    request = _selection_window_request()
    request["world_template"]["initialization"]["standing_variation_profile"] = (
        "prefix-${seed}"
    )
    with pytest.raises(ScienceRequestV1CompilerError, match="placeholders"):
        compile_science_request_v1(request)


def test_compilation_is_independent_of_dimension_object_key_order() -> None:
    first_raw = _mutation_pressure_request()
    second_raw = deepcopy(first_raw)
    second_raw["dimensions"] = dict(reversed(list(second_raw["dimensions"].items())))
    first = compile_science_request_v1(validate_eiel_science_request_v1(first_raw))
    second = compile_science_request_v1(validate_eiel_science_request_v1(second_raw))

    assert first.world_bundle.bundle_content_sha256() == (
        second.world_bundle.bundle_content_sha256()
    )
    assert first.compilation_receipt == second.compilation_receipt
