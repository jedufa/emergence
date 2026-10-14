from __future__ import annotations

import copy

import pytest

from engine.evolution import gmnw_v1 as gmnw
from tools.scripts import eiel_gmnw_venv1_matrix_contract_v1 as subject


def test_matrix_is_exact_profile_arm_seed_product_in_frozen_order() -> None:
    cells = subject.cells_v1()

    assert len(cells) == subject.PHYSICAL_CELLS == 72
    assert cells[0].cell_id == "venv1_p1_gd_s101"
    assert cells[5].cell_id == "venv1_p1_gd_s606"
    assert cells[6].cell_id == "venv1_p1_rd_s101"
    assert cells[23].cell_id == "venv1_p1_fi_s606"
    assert cells[24].cell_id == "venv1_p2_gd_s101"
    assert cells[-1].cell_id == "venv1_p3_fi_s606"
    assert [cell.ordinal for cell in cells] == list(range(72))
    assert len({cell.cell_id for cell in cells}) == 72


def test_profiles_are_bound_to_the_canonical_registry() -> None:
    profiles = subject.profile_contracts_v1()

    assert [row["profile_id"] for row in profiles] == ["P1", "P2", "P3"]
    assert [row["profile_code"] for row in profiles] == [1, 2, 3]
    assert {
        row["profile_id"]: row["profile_config"]["config_hash"] for row in profiles
    } == {
        profile_id: gmnw.gmnw_venv1_profile_config_hash(profile_id)
        for profile_id in subject.PROFILE_IDS
    }
    contract = subject.contract_v1()
    assert contract["canonical_profile_registry"][
        "contract_hash"
    ] == gmnw.gmnw_venv1_profile_contract_hash()


def test_arms_freeze_context_visibility_and_policy_semantics() -> None:
    assert [
        (
            arm.arm_id,
            arm.policy_arm,
            arm.context,
            arm.cue_visible_at_decision,
            arm.reset_hidden_each_tick,
        )
        for arm in subject.arms_v1()
    ] == [
        ("G-D", "persistent_gru", "delayed", False, False),
        ("R-D", "reset_gru", "delayed", False, True),
        ("F-D", "matched_mlp", "delayed", False, False),
        ("F-I", "matched_mlp", "immediate", True, False),
    ]


def test_each_accepted_source_has_twelve_exact_non_reseeded_forks() -> None:
    cells = subject.cells_v1()

    assert subject.SOURCE_SEEDS == (101, 202, 303, 404, 505, 606)
    for group, seed in enumerate(subject.SOURCE_SEEDS):
        forks = [cell for cell in cells if cell.source_seed == seed]
        assert len(forks) == subject.FORKS_PER_SOURCE == 12
        assert {cell.source_group for cell in forks} == {group}
        assert {cell.task_seed for cell in forks} == {seed}
        assert {cell.source_id for cell in forks} == {
            f"gmnw-ub100-source-s{seed:03d}"
        }
        assert {cell.fork_reseeded for cell in forks} == {False}
        assert {cell.seed_derivation for cell in forks} == {None}
        assert len({cell.fork_branch_id for cell in forks}) == 12


def test_contract_freezes_g100_t80_checkpoints_schema_and_claim_boundary() -> None:
    value = subject.validate_contract_v1()

    assert value["run_label"] == "gmnw-venv1-b72g100t80"
    assert value["device_binding"] is None
    assert value["execution_mechanics"] == "not_defined_by_this_contract"
    assert value["generations"] == 100
    assert value["ticks_per_generation"] == 80
    assert value["checkpoint_generations"] == [25, 50, 75, 100]
    assert value["result_schema"]["title"] == subject.RESULT_SCHEMA_VERSION
    assert "no selection" in value["claim_boundary"]
    assert "intelligence" in value["claim_boundary"]
    assert value["contract_sha256"] == subject.FROZEN_CONTRACT_SHA256


def test_contract_validation_rejects_rehashed_semantic_drift() -> None:
    changed = copy.deepcopy(subject.contract_v1())
    changed["cells"][0]["fork_reseeded"] = True
    material = dict(changed)
    material.pop("contract_sha256")
    changed["contract_sha256"] = subject.canonical_sha256_v1(material)

    with pytest.raises(
        subject.Venv1MatrixContractError,
        match="frozen hash|differs from the frozen",
    ):
        subject.validate_contract_v1(changed)


def test_contract_validation_rejects_hash_tampering() -> None:
    changed = copy.deepcopy(subject.contract_v1())
    changed["cells"][0]["source_seed"] = 202

    with pytest.raises(subject.Venv1MatrixContractError, match="contract_sha256"):
        subject.validate_contract_v1(changed)
