from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
from typing import Any, Mapping

import pytest
import torch

from evolution.phase_f_v2.matrix import build_phase_f_v2_matrix
from perf.cuda_architecture_b_modular_state_abi import (
    MODULAR_STORAGE_PROJECTION_PREFIX,
)
from perf.cuda_architecture_b_phase_e_formal_composition import (
    execute_phase_e_cpu_formal_composition,
)
from perf.eiel_phase_f_forced_combination_fixture_v2 import (
    build_phase_f_fixture_registry_v2,
)
from perf.eiel_phase_f_forced_combination_source_v2 import (
    ACTION_COUNT,
    AGENT_CAPACITY,
    DEVELOPMENT_LATE_START_STEP,
    DEVELOPMENT_MATURE_START_STEP,
    GENERATION_TENSOR_NAMES,
    GRID_HEIGHT,
    GRID_WIDTH,
    OBSERVATION_WIDTH,
    PANEL_AGENT_COUNT,
    PhaseFV2SourceError,
    materialize_phase_f_source_v2,
    validate_phase_f_source_v2,
)


def _assert_tensor_maps_equal(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> None:
    assert set(left) == set(right)
    for name in left:
        assert left[name].dtype == right[name].dtype
        assert tuple(left[name].shape) == tuple(right[name].shape)
        assert torch.equal(left[name], right[name]), name


@lru_cache(maxsize=1)
def _case_seeds() -> dict[str, int]:
    return {
        str(row["case_id"]): int(row["fixture_blueprint"]["construction_seed"])
        for row in build_phase_f_v2_matrix()["cases"]
    }


def _case_seed(case: Mapping[str, Any]) -> int:
    return _case_seeds()[str(case["case_id"])]


@lru_cache(maxsize=1)
def _registry() -> dict[str, Any]:
    return build_phase_f_fixture_registry_v2()


def test_all_120_cases_materialize_repeatably_for_both_profiles() -> None:
    registry = _registry()
    assert registry["case_count"] == 120
    profiles: dict[str, int] = {}
    source_digests: set[str] = set()
    for case in registry["cases"]:
        source_seed = _case_seed(case)
        first = materialize_phase_f_source_v2(case, source_seed=source_seed)
        second = materialize_phase_f_source_v2(case, source_seed=source_seed)
        assert first.source_sha256 == second.source_sha256
        assert first.case_id == case["case_id"]
        assert first.case_sha256 == case["case_sha256"]
        assert first.fixture_sha256 == case["fixture_sha256"]
        assert first.matrix_sha256 == case["matrix_sha256"]
        assert first.profile_id == case["profile_id"]
        assert first.role_action == case["role_action"]
        assert first.source_seed == source_seed
        _assert_tensor_maps_equal(first.generation_tensors, second.generation_tensors)
        assert len(first.formal_oracle_inputs) == PANEL_AGENT_COUNT

        tensors = first.generation_tensors
        projected_names = {
            name
            for name in tensors
            if name.startswith(MODULAR_STORAGE_PROJECTION_PREFIX)
        }
        assert len(projected_names) == 57
        assert set(tensors) - projected_names == set(GENERATION_TENSOR_NAMES)
        assert tuple(tensors["alive_mask"].shape) == (AGENT_CAPACITY,)
        assert int(tensors["alive_mask"].sum()) == PANEL_AGENT_COUNT
        assert tuple(tensors["agent_grid"].shape) == (GRID_HEIGHT, GRID_WIDTH)
        assert tuple(tensors["food_grid"].shape) == (GRID_HEIGHT, GRID_WIDTH)
        assert tuple(tensors["drives"].shape) == (AGENT_CAPACITY, 4)
        assert tuple(tensors["drive_noise_samples_by_tick_slot"].shape) == (
            1,
            AGENT_CAPACITY,
            4,
        )
        assert not bool(torch.any(tensors["drive_noise"]))
        assert not bool(torch.any(tensors["drive_noise_samples_by_tick_slot"]))
        assert first.generation_config.action_count == ACTION_COUNT
        assert first.generation_config.steps == 1
        assert first.generation_config.max_births_per_tick == 0
        assert first.generation_config.generation_turnover_enabled is False
        assert first.generation_config.enable_combat is False
        assert (
            first.generation_config.structural_development_mature_start_step
            == DEVELOPMENT_MATURE_START_STEP
        )
        assert (
            first.generation_config.structural_development_late_start_step
            == DEVELOPMENT_LATE_START_STEP
        )
        assert first.owner_metadata["deaths_enabled"] is False
        assert first.owner_metadata["modular_rng_draw_count"] == 0

        ids = [int(agent["record"]["agent_id"]) for agent in case["agents"]]
        tokens = [int(agent["record"]["agent_slot_token"]) for agent in case["agents"]]
        assert tensors["slot_to_agent_id"][:PANEL_AGENT_COUNT].tolist() == ids
        assert tensors["slot_tokens"][:PANEL_AGENT_COUNT].tolist() == tokens
        assert [value.agent_id for value in first.formal_oracle_inputs] == ids
        assert [
            value.agent_slot_token for value in first.formal_oracle_inputs
        ] == tokens
        for slot, oracle_input in enumerate(first.formal_oracle_inputs):
            assert oracle_input.agent_slot == slot
            assert oracle_input.rng_stream_indices == {"modular": 0}
            assert tuple(oracle_input.modular_observations.shape) == (
                1,
                OBSERVATION_WIDTH,
            )
            assert torch.equal(
                oracle_input.modular_observations,
                oracle_input.policy_observations,
            )
            assert tuple(oracle_input.validity_mask.shape) == (1, ACTION_COUNT)
            assert bool(torch.all(oracle_input.validity_mask))
            assert bool(torch.all(oracle_input.authority_mask))
            for plane_name, formal_plane in oracle_input.planes.items():
                owner_plane = tensors[
                    f"{MODULAR_STORAGE_PROJECTION_PREFIX}{plane_name}"
                ]
                assert torch.equal(formal_plane[0], owner_plane[slot]), plane_name
            expected_step = 0
            variant_id = str(case["agents"][slot]["variant_id"])
            if (case["family_id"], variant_id) in {
                ("f03", "v6"),
                ("f07", "v1"),
            }:
                expected_step = DEVELOPMENT_MATURE_START_STEP
            elif (case["family_id"], variant_id) == ("f07", "v2"):
                expected_step = DEVELOPMENT_LATE_START_STEP
            expected_stage = (
                2
                if expected_step >= DEVELOPMENT_LATE_START_STEP
                else 1
                if expected_step >= DEVELOPMENT_MATURE_START_STEP
                else 0
            )
            assert int(tensors["survival_steps"][slot]) == expected_step
            assert int(oracle_input.stage_indices.item()) == expected_stage

        profiles[first.profile_id] = profiles.get(first.profile_id, 0) + 1
        source_digests.add(first.source_sha256)

    assert profiles == {"b2-12x32-v1": 60, "b2-16x48-v1": 60}
    assert len(source_digests) == 120


def test_mature_and_late_variants_execute_at_the_frozen_stage() -> None:
    registry = _registry()
    expected = {
        "f03": {6: 1, 14: 1},
        "f07": {1: 1, 2: 2, 9: 1, 10: 2},
    }
    for profile_id in ("b2-12x32-v1", "b2-16x48-v1"):
        for family_id, slots in expected.items():
            case = next(
                row
                for row in registry["cases"]
                if row["profile_id"] == profile_id and row["family_id"] == family_id
            )
            source = materialize_phase_f_source_v2(case, source_seed=_case_seed(case))
            for slot, stage in slots.items():
                formal_input = source.formal_oracle_inputs[slot]
                assert int(formal_input.stage_indices.item()) == stage
                result = execute_phase_e_cpu_formal_composition(
                    **formal_input.call_kwargs()
                )
                effective = result.modular_audit_tensors["development_effective_gain"][
                    0
                ]
                leaf_slot = 1
                assert float(effective[leaf_slot]) == 1.0


def test_formal_oracle_inputs_execute_without_rng_or_identity_drift() -> None:
    registry = _registry()
    representatives = {
        profile: next(
            case for case in registry["cases"] if case["profile_id"] == profile
        )
        for profile in ("b2-12x32-v1", "b2-16x48-v1")
    }
    for case in representatives.values():
        source = materialize_phase_f_source_v2(case, source_seed=_case_seed(case))
        for oracle_input in source.formal_oracle_inputs:
            result = execute_phase_e_cpu_formal_composition(
                **oracle_input.call_kwargs()
            )
            assert result.rng_stream_indices_before == {"modular": 0}
            assert result.rng_stream_indices_after == {"modular": 0}
            assert torch.equal(
                result.torch_rng_state_before, result.torch_rng_state_after
            )
            assert (
                result.numpy_rng_state_before["position"]
                == (result.numpy_rng_state_after["position"])
            )
            assert torch.equal(
                result.numpy_rng_state_before["keys"],
                result.numpy_rng_state_after["keys"],
            )
            assert result.python_rng_state_before == result.python_rng_state_after
            assert tuple(result.composed_logits.shape) == (1, ACTION_COUNT)
            assert result.planes_before["agent_id"].item() == oracle_input.agent_id
            assert result.planes_before["agent_slot_token"].item() == (
                oracle_input.agent_slot_token
            )


def test_source_seed_is_preregistered_and_repeat_stable() -> None:
    case = _registry()["cases"][0]
    source_seed = _case_seed(case)
    first = materialize_phase_f_source_v2(case, source_seed=source_seed)
    repeated = materialize_phase_f_source_v2(case, source_seed=source_seed)
    assert first.source_sha256 == repeated.source_sha256
    with pytest.raises(PhaseFV2SourceError, match="construction_seed"):
        materialize_phase_f_source_v2(case, source_seed=source_seed + 1)


def test_source_validation_fails_closed_for_fixture_seed_and_tensor_drift() -> None:
    registry = _registry()
    case = registry["cases"][0]
    source_seed = _case_seed(case)
    source = materialize_phase_f_source_v2(case, source_seed=source_seed)
    validated = validate_phase_f_source_v2(
        source, fixture_case=case, source_seed=source_seed
    )
    assert validated.source_sha256 == source.source_sha256

    altered_case = deepcopy(case)
    altered_case["role_action"] = "not_the_frozen_role_action"
    with pytest.raises(PhaseFV2SourceError, match="fixture case"):
        materialize_phase_f_source_v2(altered_case, source_seed=source_seed)
    with pytest.raises(PhaseFV2SourceError, match="construction_seed"):
        validate_phase_f_source_v2(
            source, fixture_case=case, source_seed=source_seed + 1
        )

    altered_tensors = {
        name: tensor.clone() for name, tensor in source.generation_tensors.items()
    }
    altered_tensors["energy"][0] += 1.0
    altered_source = replace(source, generation_tensors=altered_tensors)
    with pytest.raises(PhaseFV2SourceError, match="generation tensors"):
        validate_phase_f_source_v2(
            altered_source, fixture_case=case, source_seed=source_seed
        )


@pytest.mark.parametrize("bad_seed", [True, -1, 1 << 63, 1.0, "7"])
def test_source_seed_contract_fails_closed(bad_seed: object) -> None:
    case = _registry()["cases"][0]
    with pytest.raises(PhaseFV2SourceError, match="uint63"):
        materialize_phase_f_source_v2(case, source_seed=bad_seed)  # type: ignore[arg-type]
