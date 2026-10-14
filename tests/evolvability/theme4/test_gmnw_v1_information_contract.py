from __future__ import annotations

import json

import numpy as np
import pytest

from env.gridworld_pkg.constants import ACT_MOVE_E, ACT_MOVE_N, ACT_REST
from evolution.gmnw_v1 import (
    GMNW_PRIVATE_STATE_FIELDS,
    GMNW_TASK_OBSERVATION_COLUMNS,
    GMNW_VENV1_PROFILE_CODES,
    GMNW_VENV1_PROFILE_IDS,
    GmnwConfig,
    GmnwConsequenceStatus,
    GmnwOutcomeState,
    GmnwPhase,
    advance_private_task_state,
    apply_decision_consequence,
    append_task_observation,
    balanced_cue_identity,
    decision_observation_equivalence,
    episode_balance_contract,
    failure_isolation_contract,
    gmnw_venv1_profile_config,
    gmnw_venv1_profile_config_hash,
    gmnw_venv1_profile_contract,
    gmnw_venv1_profile_contract_hash,
    initial_private_task_state,
    pressure_for_generation,
    reproduction_opportunity_contract,
    resolve_gmnw_venv1_profile,
    resolve_gmnw_venv1_profile_config,
    reset_episode_reproduction_opportunity,
    task_observation,
    turnover_extinction_contract,
    witness_inheritance_contract,
    write_information_contract_artifacts,
    write_presimulation_contract_artifacts,
)


@pytest.mark.parametrize("cue_alphabet", (2, 4))
def test_gmnw_cues_are_balanced_inside_every_complete_block(
    cue_alphabet: int,
) -> None:
    config = GmnwConfig(cue_alphabet=cue_alphabet, task_seed=101)
    for stream_id in (0, 1, 47):
        for block in range(8):
            cues = [
                balanced_cue_identity(
                    config,
                    stream_id=stream_id,
                    episode_index=block * cue_alphabet + offset,
                )
                for offset in range(cue_alphabet)
            ]
            assert sorted(cues) == list(range(cue_alphabet))


@pytest.mark.parametrize("distractors_enabled", (False, True))
@pytest.mark.parametrize("cue_alphabet", (2, 4))
def test_gmnw_decision_observation_is_byte_identical_across_cues(
    cue_alphabet: int,
    distractors_enabled: bool,
) -> None:
    config = GmnwConfig(
        cue_alphabet=cue_alphabet,
        distractors_enabled=distractors_enabled,
    )
    audit = decision_observation_equivalence(config)

    assert audit["decision_observation_bytes_identical"] is True
    assert len(set(audit["decision_observation_sha256_by_cue"])) == 1
    assert audit["majority_observation_only_accuracy"] == pytest.approx(
        config.chance_accuracy
    )
    assert audit["majority_observation_only_accuracy"] <= audit["leakage_threshold"]


def test_gmnw_phase_machine_requires_the_earlier_cue() -> None:
    config = GmnwConfig(
        cue_alphabet=4,
        cue_ticks=1,
        delay_ticks=4,
        consequence_ticks=1,
        distractors_enabled=True,
        task_seed=202,
    )
    state = initial_private_task_state(config, stream_id=3)
    expected_action = state.correct_action
    assert state.phase is GmnwPhase.CUE
    assert task_observation(state, config)[state.cue_identity] == 1.0

    step = advance_private_task_state(
        state,
        config,
        stream_id=3,
        action=ACT_MOVE_E,
    )
    assert step.action_applied_to_public_world is False
    state = step.state
    assert state.phase is GmnwPhase.DELAY

    for _ in range(4):
        observation = task_observation(state, config)
        assert np.count_nonzero(observation[:4]) == 0
        step = advance_private_task_state(
            state,
            config,
            stream_id=3,
            action=ACT_MOVE_N,
        )
        assert step.action_applied_to_public_world is False
        state = step.state

    assert state.phase is GmnwPhase.DECISION
    assert state.decision_due is True
    observation = task_observation(state, config)
    assert np.count_nonzero(observation[:4]) == 0
    assert np.count_nonzero(observation[8:]) == 0

    step = advance_private_task_state(
        state,
        config,
        stream_id=3,
        action=expected_action,
    )
    assert step.decision_recorded is True
    assert step.decision_correct is True
    assert step.action_applied_to_public_world is False
    assert step.state.consequence_status == int(GmnwConsequenceStatus.CORRECT)

    next_episode = advance_private_task_state(
        step.state,
        config,
        stream_id=3,
        action=ACT_REST,
    ).state
    assert next_episode.phase is GmnwPhase.CUE
    assert next_episode.episode_index == 1
    assert next_episode.task_rng_cursor == 2


def test_immediate_control_exposes_only_the_same_cue_at_decision() -> None:
    delayed = GmnwConfig(cue_alphabet=4, cue_visible_at_decision=False)
    immediate = GmnwConfig(cue_alphabet=4, cue_visible_at_decision=True)
    state = initial_private_task_state(delayed, stream_id=3)
    while state.phase is not GmnwPhase.DECISION:
        state = advance_private_task_state(
            state,
            delayed,
            stream_id=3,
            action=ACT_REST,
        ).state

    delayed_row = task_observation(state, delayed)
    immediate_row = task_observation(state, immediate)
    assert np.count_nonzero(delayed_row[:4]) == 0
    assert immediate_row[state.cue_identity] == 1.0
    assert np.count_nonzero(immediate_row[:4]) == 1
    assert np.array_equal(delayed_row[4:], immediate_row[4:])


def test_gmnw_private_state_is_not_added_to_public_observation() -> None:
    config = GmnwConfig(cue_alphabet=4)
    state = initial_private_task_state(config, stream_id=11)
    public_base = np.arange(9, dtype=np.float32)
    combined = append_task_observation(public_base, state, config)

    assert combined.shape == (9 + len(GMNW_TASK_OBSERVATION_COLUMNS),)
    assert set(GMNW_PRIVATE_STATE_FIELDS).isdisjoint(GMNW_TASK_OBSERVATION_COLUMNS)
    assert np.array_equal(combined[:9], public_base)


def test_gmnw_pressure_schedule_is_frozen_and_has_no_correct_decision_bonus() -> None:
    config = GmnwConfig()
    assert pressure_for_generation(config, 0).tier == "grace"
    assert pressure_for_generation(config, 3).damage_fraction == pytest.approx(0.15)
    assert pressure_for_generation(config, 4).tier == "severe"
    assert pressure_for_generation(config, 7).damage_fraction == pytest.approx(0.45)
    assert pressure_for_generation(config, 8).tier == "lethal"
    assert pressure_for_generation(config, 31).damage_fraction == pytest.approx(1.0)

    initial = GmnwOutcomeState()
    correct = apply_decision_consequence(
        initial,
        decision_correct=True,
        pressure=pressure_for_generation(config, 8),
    )
    assert correct.health == initial.health
    assert correct.reproduction_scale == initial.reproduction_scale
    assert correct.correct_count == 1
    assert correct.damage_total == 0.0


def test_gmnw_config_represents_the_frozen_equal_capability_pressure() -> None:
    config = GmnwConfig(
        grace_damage_fraction=0.01,
        severe_damage_fraction=0.01,
    )

    assert pressure_for_generation(config, 0).damage_fraction == pytest.approx(0.01)
    assert pressure_for_generation(config, 4).damage_fraction == pytest.approx(0.01)


def test_gmnw_venv1_registry_exposes_canonical_ids_codes_and_exact_values() -> None:
    assert GMNW_VENV1_PROFILE_IDS == ("legacy", "P1", "P2", "P3")
    assert dict(GMNW_VENV1_PROFILE_CODES) == {
        "legacy": 0,
        "P1": 1,
        "P2": 2,
        "P3": 3,
    }

    expected_damage = {"P1": 0.0, "P2": 0.25, "P3": 0.5}
    for profile_id, damage_fraction in expected_damage.items():
        profile = resolve_gmnw_venv1_profile(profile_id)
        assert resolve_gmnw_venv1_profile(profile.profile_code) is profile
        assert profile.generation_tier_independent is True
        assert profile.incorrect_reproduction_scale == 0.0
        assert profile.incorrect_lethal_damage_multiplier == damage_fraction
        for generation_index in (0, 4, 8, 31):
            resolved = resolve_gmnw_venv1_profile_config(
                profile_id,
                generation_index=generation_index,
            )
            assert resolved["profile_id"] == profile_id
            assert resolved["profile_code"] == profile.profile_code
            assert resolved["incorrect_reproduction_scale"] == 0.0
            assert resolved["incorrect_damage_fraction"] == damage_fraction
            assert resolved["incorrect_lethal_damage_multiplier"] == damage_fraction
            assert resolved["incorrect_decision_lethal"] is False


def test_gmnw_venv1_legacy_resolution_preserves_existing_schedule_exactly() -> None:
    config = GmnwConfig()
    for generation_index in (0, 3, 4, 7, 8, 31):
        legacy = pressure_for_generation(config, generation_index)
        resolved = resolve_gmnw_venv1_profile_config(
            "legacy",
            generation_index=generation_index,
            config=config,
        )
        assert resolved["profile_id"] == "legacy"
        assert resolved["profile_code"] == 0
        assert resolved["generation_tier_independent"] is False
        assert resolved["generation_tier"] == legacy.tier
        assert resolved["incorrect_reproduction_scale"] == legacy.reproduction_scale
        assert resolved["incorrect_damage_fraction"] == legacy.damage_fraction
        assert resolved["incorrect_decision_lethal"] is legacy.lethal


def test_gmnw_venv1_profiles_scale_the_accepted_one_hit_lethal_damage() -> None:
    accepted = GmnwConfig(lethal_damage_fraction=2.0)
    resolved = resolve_gmnw_venv1_profile_config(
        "P2",
        generation_index=31,
        config=accepted,
    )

    assert resolved["accepted_one_hit_lethal_damage"] == 2.0
    assert resolved["incorrect_lethal_damage_multiplier"] == 0.25
    assert resolved["incorrect_damage_fraction"] == 0.5
    assert resolved["config_hash"] != gmnw_venv1_profile_config_hash("P2")


def test_gmnw_venv1_hashes_are_canonical_and_bind_exact_resolved_config() -> None:
    first_contract = gmnw_venv1_profile_contract()
    second_contract = gmnw_venv1_profile_contract()
    assert first_contract == second_contract
    assert first_contract["contract_hash"] == gmnw_venv1_profile_contract_hash()
    assert first_contract["contract_hash"] == (
        "905c602854699296c7dac19bc9911074e8bb36ef0a09e1105d12b71e65e5c60d"
    )

    p2_config = gmnw_venv1_profile_config("P2")
    assert p2_config["profile_id"] == "P2"
    assert p2_config["config_hash"] == (
        "60cf7954c0b5d5b6187cf1a306ba14613c503a0d26d38709b64ad67293f1624a"
    )
    assert p2_config["resolved_by_generation_tier"] == {
        tier: {
            "incorrect_reproduction_scale": 0.0,
            "incorrect_damage_fraction": 0.25,
        }
        for tier in ("grace", "severe", "lethal")
    }
    p2_generation_0 = resolve_gmnw_venv1_profile_config(
        "P2",
        generation_index=0,
    )
    p2_generation_0_by_code = resolve_gmnw_venv1_profile_config(
        2,
        generation_index=0,
    )
    assert p2_generation_0 == p2_generation_0_by_code
    assert p2_generation_0["config_hash"] == gmnw_venv1_profile_config_hash(
        "P2",
    )
    assert p2_generation_0["config_hash"] == resolve_gmnw_venv1_profile_config(
        "P2",
        generation_index=4,
    )["config_hash"]
    assert p2_generation_0["config_hash"] != resolve_gmnw_venv1_profile_config(
        "P3",
        generation_index=0,
    )["config_hash"]


@pytest.mark.parametrize(
    "profile",
    (None, True, False, -1, 4, 1.0, "", "p1", "P0", "P4", " P1", "P1 "),
)
def test_gmnw_venv1_profile_resolution_fails_closed(profile: object) -> None:
    with pytest.raises(ValueError, match="exact canonical ID"):
        resolve_gmnw_venv1_profile(profile)  # type: ignore[arg-type]


def test_gmnw_incorrect_consequences_escalate_and_episode_reset_is_bounded() -> None:
    config = GmnwConfig()
    initial = GmnwOutcomeState()
    grace = apply_decision_consequence(
        initial,
        decision_correct=False,
        pressure=pressure_for_generation(config, 0),
    )
    assert grace.health == pytest.approx(0.85)
    assert grace.reproduction_scale == pytest.approx(0.5)
    assert grace.alive is True
    assert reset_episode_reproduction_opportunity(grace).reproduction_scale == 1.0

    severe = apply_decision_consequence(
        initial,
        decision_correct=False,
        pressure=pressure_for_generation(config, 4),
    )
    assert severe.health == pytest.approx(0.55)
    assert severe.reproduction_scale == 0.0
    assert severe.alive is True

    lethal = apply_decision_consequence(
        initial,
        decision_correct=False,
        pressure=pressure_for_generation(config, 8),
    )
    assert lethal.health == pytest.approx(0.0)
    assert lethal.reproduction_scale == 0.0
    assert lethal.alive is False


def test_gmnw_selection_contracts_fail_closed_against_hidden_refill_and_aliasing() -> (
    None
):
    turnover = turnover_extinction_contract()
    reproduction = reproduction_opportunity_contract()
    episodes = episode_balance_contract()
    witness = witness_inheritance_contract()
    failure = failure_isolation_contract()

    assert turnover["extinct_lineage_reintroduction"] is False
    assert turnover["generic_founder_template_reuse"] is False
    assert turnover["zero_eligible_parent_result"] == "extinct_no_eligible_parents"
    assert reproduction["multiplicative_compounding_within_episode"] is False
    assert reproduction["generation_boundary_reset"] is False
    assert episodes["new_agent"] == "starts_a_fresh_private_block"
    assert witness["policy_weight_mutation_enabled"] is False
    assert witness["witness_lineage_protection"] is False
    assert failure["mutable_policy_parameter_aliasing"] is False
    assert failure["matrix_acceptance_when_any_cell_fails"] is False


def test_gmnw_contract_artifacts_are_complete_and_machine_readable(tmp_path) -> None:
    paths = write_information_contract_artifacts(tmp_path)

    assert set(paths) == {
        "gmnw_information_contract.md",
        "gmnw_information_contract.json",
        "gmnw_observation_equivalence.json",
        "gmnw_private_state_contract.json",
        "gmnw_chance_baseline.json",
    }
    equivalence = json.loads(
        paths["gmnw_observation_equivalence.json"].read_text(encoding="utf-8")
    )
    assert equivalence["2"]["decision_observation_bytes_identical"] is True
    assert equivalence["4"]["decision_observation_bytes_identical"] is True

    presimulation = write_presimulation_contract_artifacts(tmp_path / "presimulation")
    assert set(presimulation) == {
        "gmnw_consequence_schedule.md",
        "gmnw_consequence_schedule.json",
        "gmnw_reproduction_opportunity_contract.json",
        "gmnw_turnover_extinction_contract.json",
        "gmnw_episode_balance_contract.json",
        "gmnw_witness_inheritance_contract.json",
        "gmnw_failure_isolation_contract.json",
    }
    for path in presimulation.values():
        assert path.is_file()


@pytest.mark.parametrize(
    "kwargs",
    (
        {"cue_alphabet": 3},
        {"cue_ticks": 0},
        {"delay_ticks": 0},
        {"consequence_ticks": 0},
        {"distractors_enabled": 1},
        {"task_seed": -1},
        {"grace_generation_count": 0},
        {"severe_generation_count": 0},
        {"grace_damage_fraction": 0.5},
        {"lethal_damage_fraction": 0.9},
        {"grace_reproduction_scale": 1.0},
    ),
)
def test_gmnw_config_fails_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        GmnwConfig(**kwargs)
