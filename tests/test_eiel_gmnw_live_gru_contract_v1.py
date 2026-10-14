from __future__ import annotations

import json

import pytest

from engine.perf.gmnw_live_gru_contract_v1 import (
    DATA_SPLIT_SCHEMA_V1,
    GmnwLiveGruContractError,
    LIVE_CONTRACT_SCHEMA_V1,
    data_split_v1,
    live_contract_v1,
    validate_bound_payload,
    write_live_contract_artifacts_v1,
)


def test_live_contract_freezes_the_real_86_column_path() -> None:
    contract = live_contract_v1()

    assert contract["observation"]["column_count"] == 86
    assert contract["observation"]["columns"][74:] == [
        "gmnw_cue_0",
        "gmnw_cue_1",
        "gmnw_cue_2",
        "gmnw_cue_3",
        "gmnw_phase_cue",
        "gmnw_phase_delay",
        "gmnw_phase_decision",
        "gmnw_phase_consequence",
        "gmnw_distractor_0",
        "gmnw_distractor_1",
        "gmnw_distractor_2",
        "gmnw_distractor_3",
    ]
    assert contract["task_timing"]["episode_policy_steps"] == "delay_ticks + 3"
    assert contract["task_timing"]["persistent_hidden_crosses_episode_boundaries"]
    assert contract["action_sampling"]["temperature"] == 0.05
    assert contract["actions"]["cue_to_private_action"] == [0, 2, 1, 3]
    assert contract["actions"]["public_action_name"] == "REST"
    assert contract["pressure_schedule"]["grace_incorrect_damage_fraction"] == 0.01
    assert contract["checkpoint"]["gmnw_sidecar_alone_is_complete"] is False
    assert contract["ownership"]["python_tick_ownership"] is False


def test_data_split_is_disjoint_complete_and_hash_bound() -> None:
    split = data_split_v1()
    development = split["development"]["simulation_and_task_seeds"]
    qualification = split["qualification"]["simulation_and_task_seeds"]

    assert set(development).isdisjoint(qualification)
    assert len(split["qualification"]["task_configurations"]) == 8
    assert split["qualification"]["completed_episodes_per_initial_stream"] == 48
    assert len(split["qualification"]["cue_sequence_sha256"]) == 64
    assert validate_bound_payload(
        split,
        schema_version=DATA_SPLIT_SCHEMA_V1,
        hash_field="data_split_sha256",
    ) == split


def test_bound_contract_rejects_drift() -> None:
    contract = live_contract_v1()
    contract["action_sampling"]["temperature"] = 0.1
    with pytest.raises(GmnwLiveGruContractError, match="contract_sha256 mismatch"):
        validate_bound_payload(
            contract,
            schema_version=LIVE_CONTRACT_SCHEMA_V1,
            hash_field="contract_sha256",
        )


def test_writer_roundtrips_both_bound_artifacts(tmp_path) -> None:
    outputs = write_live_contract_artifacts_v1(tmp_path)
    contract = json.loads(outputs["contract"].read_text(encoding="ascii"))
    split = json.loads(outputs["data_split"].read_text(encoding="ascii"))

    assert validate_bound_payload(
        contract,
        schema_version=LIVE_CONTRACT_SCHEMA_V1,
        hash_field="contract_sha256",
    ) == contract
    assert validate_bound_payload(
        split,
        schema_version=DATA_SPLIT_SCHEMA_V1,
        hash_field="data_split_sha256",
    ) == split
