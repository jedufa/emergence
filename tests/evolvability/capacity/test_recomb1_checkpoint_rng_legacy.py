from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from evolution.inheritance_registry_v1 import (
    ALLOWED_INHERITANCE_MODES,
    INHERITANCE_CHECKPOINT_SCHEMA_VERSION,
    INHERITANCE_REGISTRY_VERSION,
    DIPLOID_INHERITANCE_PROFILE,
    LEGACY_INHERITANCE_PROFILE,
    LINKAGE_BLOCK_IDS,
    LINKED_INHERITANCE_PROFILE,
    LINKED_PROFILE_DRAW_COUNT,
    diploid_flat_linkage_component_contract_v1,
    historical_linked_checkpoint_metadata_v1,
    inheritance_checkpoint_metadata,
    inspect_historical_inheritance_checkpoint_v1,
    validate_production_inheritance_profile,
    validate_inheritance_execution_lane,
)
from simulation.checkpoint_bundle_v2 import (
    CheckpointStrictError,
    _build_metadata,
    _validate_metadata,
    checkpoint_bundle_v2_inspect_historical,
    checkpoint_bundle_v2_load,
    validate_inheritance_checkpoint_compatibility,
)
from simulation.control_plane_pkg.util_json import canonical_json_bytes
from simulation.runner_pkg.runner_base import RunnerBase
from simulation.runner_pkg.runner_config import SimConfig


def _sim(profile: str) -> SimpleNamespace:
    return SimpleNamespace(sim_cfg=SimConfig(inheritance_profile=profile))


def _metadata(profile: str) -> dict:
    return _build_metadata(
        _sim(profile),
        generation_index=4,
        strict_replay_mode=False,
        world_hash="w",
        population_hash="p",
        rng_snapshot_hash="r",
        runner_state_hash="s",
        bank_state_hash="b",
        resident_runtime_state_hash="rr",
        native_runtime_checkpoint_hash="n",
        event_log_hash="e",
        wal_hash="wal",
    )


def _metadata_without_inheritance_field() -> dict:
    return _build_metadata(
        SimpleNamespace(sim_cfg=SimpleNamespace()),
        generation_index=4,
        strict_replay_mode=False,
        world_hash="w",
        population_hash="p",
        rng_snapshot_hash="r",
        runner_state_hash="s",
        bank_state_hash="b",
        resident_runtime_state_hash="rr",
        native_runtime_checkpoint_hash="n",
        event_log_hash="e",
        wal_hash="wal",
    )


def test_default_profile_is_legacy_and_is_serialized_in_run_config() -> None:
    cfg = SimConfig()
    assert cfg.inheritance_profile == LEGACY_INHERITANCE_PROFILE
    assert cfg.__dict__["inheritance_profile"] == LEGACY_INHERITANCE_PROFILE


def test_profile_validation_happens_before_rng_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    constructed = False

    def _forbidden_rng(*_args, **_kwargs):
        nonlocal constructed
        constructed = True
        raise AssertionError("RNGManager must not be constructed")

    monkeypatch.setattr("simulation.runner_pkg.runner_base.RNGManager", _forbidden_rng)
    bad_cfg = SimpleNamespace(inheritance_profile="not_a_frozen_profile")
    with pytest.raises(ValueError, match="inheritance"):
        RunnerBase(
            SimpleNamespace(),
            SimpleNamespace(),
            bad_cfg,
            SimpleNamespace(),
            "/tmp/recomb1_invalid_profile",
        )
    assert constructed is False


def test_retired_profile_rejects_every_production_execution_lane() -> None:
    linked = SimpleNamespace(
        inheritance_profile=LINKED_INHERITANCE_PROFILE,
        device="auto",
        bank_device="auto",
    )
    for policy_device, bank_device in (
        ("cpu", "cpu"),
        ("cuda:0", "cpu"),
        ("cpu", "cuda:0"),
        ("cuda:0", "cuda:0"),
    ):
        with pytest.raises(
            RuntimeError, match="P12_RETIRED_INHERITANCE_PROFILE"
        ):
            validate_inheritance_execution_lane(
                linked,
                policy_device=policy_device,
                bank_device=bank_device,
            )

    with pytest.raises(ValueError, match="P12_RETIRED_INHERITANCE_PROFILE"):
        validate_production_inheritance_profile(LINKED_INHERITANCE_PROFILE)


def test_genome_v2_uses_its_separately_qualified_native_cuda_lane() -> None:
    binding = {
        "schema_version": "genome_v2_checkpoint_authority_binding_v1",
        "provider_id": "eiel-production-provider-v1",
        "runtime_profile_id": "eiel-production-runtime-v1",
        "global_genome_profile_id": "genome_upgrade_v2",
        "source_commit": "0" * 40,
        "executable_package_core_sha256": "0" * 64,
        "package_id_sha256": "1" * 64,
        "package_receipt_sha256": "2" * 64,
        "artifact_sha256": "4" * 64,
        "manifest_sha256": "5" * 64,
        "state_schema_sha256": "6" * 64,
        "mechanic_contract_sha256": "7" * 64,
        "cancellation_abi_sha256": "8" * 64,
        "observability_schema_sha256": "9" * 64,
        "capacity_layout_contract_sha256": "a" * 64,
        "capability_contract_sha256": "b" * 64,
        "execution_route_sha256": "3" * 64,
    }
    genome_v2 = SimpleNamespace(
        inheritance_profile=DIPLOID_INHERITANCE_PROFILE,
        genome_upgrade_v2_enable=True,
        genome_v2_checkpoint_authority_binding=tuple(sorted(binding.items())),
        device="cuda",
        bank_device="cuda",
        native_runtime_enable=True,
    )
    validate_inheritance_execution_lane(
        genome_v2,
        policy_device="cuda:0",
        bank_device="cuda:0",
    )

    wrong_profile = SimpleNamespace(
        inheritance_profile=LINKED_INHERITANCE_PROFILE,
        genome_upgrade_v2_enable=True,
    )
    with pytest.raises(
        RuntimeError, match="P12_RETIRED_INHERITANCE_PROFILE"
    ):
        validate_inheritance_execution_lane(wrong_profile)

    for invalid_binding in (
        None,
        {**binding, "schema_version": "legacy-binding"},
        {**binding, "provider_id": "legacy-provider"},
        {**binding, "runtime_profile_id": "legacy-runtime"},
        {**binding, "global_genome_profile_id": "legacy-genome"},
        {**binding, "package_id_sha256": "not-qualified"},
        {**binding, "execution_route_sha256": "not-qualified"},
    ):
        unqualified = SimpleNamespace(
            inheritance_profile=DIPLOID_INHERITANCE_PROFILE,
            genome_upgrade_v2_enable=True,
            genome_v2_checkpoint_authority_binding=(
                ()
                if invalid_binding is None
                else tuple(sorted(invalid_binding.items()))
            ),
        )
        with pytest.raises(
            RuntimeError, match="GENOME_UPGRADE_V2_UNQUALIFIED_NATIVE_LANE"
        ):
            validate_inheritance_execution_lane(unqualified)


def test_legacy_metadata_is_omitted_and_historical_metadata_is_frozen() -> None:
    legacy = _metadata(LEGACY_INHERITANCE_PROFILE)
    contract = historical_linked_checkpoint_metadata_v1()

    assert "inheritance" not in legacy
    with pytest.raises(ValueError, match="P12_RETIRED_INHERITANCE_PROFILE"):
        _metadata(LINKED_INHERITANCE_PROFILE)
    with pytest.raises(ValueError, match="P12_RETIRED_INHERITANCE_PROFILE"):
        inheritance_checkpoint_metadata(
            SimConfig(inheritance_profile=LINKED_INHERITANCE_PROFILE)
        )
    assert contract["schema_version"] == INHERITANCE_CHECKPOINT_SCHEMA_VERSION
    assert contract["registry_version"] == INHERITANCE_REGISTRY_VERSION
    assert contract["profile"] == LINKED_INHERITANCE_PROFILE
    assert contract["draws_per_offspring"] == LINKED_PROFILE_DRAW_COUNT == 7
    assert tuple(block["id"] for block in contract["blocks"]) == tuple(LINKAGE_BLOCK_IDS)
    assert contract["migration_rng_draws"] == 0
    assert (
        contract["mutation_after_inheritance_version"]
        == "existing_mutation_schedule_after_linkage_v1"
    )
    assert contract["supported_lane"] == "cpu_python_reference_only_in_recomb1"

    expected_fields = {
        "behavioral_strategy_v1": [
            "exploration_bias",
            "aggression_bias",
            "action_temperature",
        ],
        "regulation_prediction_v1": [
            "drive_inertia",
            "drive_noise",
            "drive_logit_gain",
            "pred_lr",
            "drive_sp_hunger",
            "drive_sp_pain",
            "drive_sp_fatigue",
            "drive_sp_fear",
            "exp_prior_strength",
            "exp_prior_noise",
            "drive_birth_noise",
        ],
        "life_history_v1": ["reproduction_threshold"],
        "mutation_rate_v1": ["mutation_rate"],
        "memory_controller_architecture_v1": [
            "hidden_size",
            "memory_dim",
            "use_gru",
            "policy_controller",
        ],
        "resource_handling_v1": ["resource_handling_axis"],
        "family_head_strategy_v1": ["family_head_residuals"],
    }
    for block in contract["blocks"]:
        assert block["fields"] == expected_fields[block["id"]]
        expected_mode = (
            "whole_record_donor"
            if block["id"] in {"memory_controller_architecture_v1", "family_head_strategy_v1"}
            else "parent_copy"
        )
        assert block["inheritance_mode"] == expected_mode

    structural = contract["structural_linkage_rule"]
    assert structural == {
        "id": "homology_connected_component_v1",
        "inheritance_mode": "homology_record_donor",
        "unit": (
            "each connected component of matched native homologs joined by matched ordinary "
            "connections or developmental dependency edges present in either parent"
        ),
        "draw_boundary": "mutation.c2.recombination_lineage",
        "draw_formula": "ctx.recomb_draws=component_count",
        "draws_per_component": 1,
        "unmatched_rule": "existing module and connection union with caps; zero donor draws",
        "pre_draw_reject_code": "DET_C2_RECOMB1_UNASSIGNED_MATCHED_CONNECTION",
        "draw_dependent_conflict_rule": (
            "consume full component schedule once then canonical validation fails closed without redraw"
        ),
    }

    registry_without_digest = {
        key: value
        for key, value in contract.items()
        if key not in {"schema_version", "profile", "registry_metadata_sha256"}
    }
    canonical_registry = json.dumps(
        registry_without_digest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    assert contract["registry_metadata_sha256"] == hashlib.sha256(
        canonical_registry
    ).hexdigest()

    assert tuple(contract["allowed_inheritance_modes"]) == ALLOWED_INHERITANCE_MODES
    loci = contract["loci"]
    assert loci
    assert len({row["semantic_locus_id"] for row in loci}) == len(loci)
    assert {field for fields in expected_fields.values() for field in fields}.issubset(
        {row["field"] for row in loci}
    )
    assert {
        "structural.module_record",
        "structural.input_routing_record",
        "structural.output_routing_record",
        "structural.developmental_record",
        "structural.dormancy_record",
        "structural.connection_record",
    }.issubset({row["semantic_locus_id"] for row in loci})
    for row in loci:
        assert row["inheritance_mode"] in ALLOWED_INHERITANCE_MODES
        assert row["linkage_group"]
        assert row["checkpoint_schema"]
        assert row["mutation_rule"]
        assert row["compatibility_version"]
        assert "bounds" in row


def test_legacy_checkpoint_identifier_omits_inheritance_material() -> None:
    metadata = _metadata(LEGACY_INHERITANCE_PROFILE)
    metadata_without_field = _metadata_without_inheritance_field()
    old_material = {
        "schema_version": metadata["schema_version"],
        "generation_index": metadata["generation_index"],
        "branch_id": metadata["branch_id"],
        "campaign_id": metadata["campaign_id"],
        "code_commit_hash": metadata["code_commit_hash"],
        "python_version": metadata["python_version"],
        "numpy_version": metadata["numpy_version"],
        "torch_version": metadata["torch_version"],
        "world_hash": metadata["world_hash"],
        "population_hash": metadata["population_hash"],
        "rng_snapshot_hash": metadata["rng_snapshot_hash"],
        "event_log_hash": metadata["event_log_hash"],
        "wal_hash": metadata["wal_hash"],
        "runner_state_hash": metadata["runner_state_hash"],
        "bank_state_hash": metadata["bank_state_hash"],
        "resident_runtime_state_hash": metadata["resident_runtime_state_hash"],
        "native_runtime_checkpoint_hash": metadata["native_runtime_checkpoint_hash"],
        "authoritative_state_identity": metadata["authoritative_state_identity"],
        "capacity_epoch_metadata": None,
        "structural_identity_metadata": metadata["structural_identity"],
    }
    assert "inheritance" not in metadata
    assert "inheritance" not in metadata_without_field
    assert metadata["checkpoint_id"] == metadata_without_field["checkpoint_id"]
    assert metadata["checkpoint_id"] == hashlib.sha256(
        canonical_json_bytes(old_material)
    ).hexdigest()


def test_retired_checkpoint_is_readable_for_analysis_but_never_resumable() -> None:
    legacy = _sim(LEGACY_INHERITANCE_PROFILE)
    repaired = _sim(LINKED_INHERITANCE_PROFILE)
    repaired_contract = historical_linked_checkpoint_metadata_v1()

    # Old/absent metadata is the legacy envelope.
    validate_inheritance_checkpoint_compatibility(
        legacy, {"metadata": {"strict_replay_mode": False}}
    )
    with pytest.raises(
        CheckpointStrictError, match="P12_RETIRED_INHERITANCE_PROFILE"
    ):
        validate_inheritance_checkpoint_compatibility(
            repaired, {"metadata": {"strict_replay_mode": False}}
        )

    with pytest.raises(
        CheckpointStrictError, match="P12_RETIRED_INHERITANCE_PROFILE"
    ):
        validate_inheritance_checkpoint_compatibility(
            repaired,
            {
                "metadata": {
                    "strict_replay_mode": False,
                    "inheritance": repaired_contract,
                }
            },
        )
    with pytest.raises(CheckpointStrictError, match="profile mismatch"):
        validate_inheritance_checkpoint_compatibility(
            legacy,
            {
                "metadata": {
                    "strict_replay_mode": False,
                    "inheritance": repaired_contract,
                }
            },
        )

    mismatched = dict(repaired_contract)
    mismatched["draws_per_offspring"] = 6
    malformed_load_metadata = {
        "schema_version": "checkpoint_bundle_v2",
        "inheritance": mismatched,
    }
    with pytest.raises(
        CheckpointStrictError, match="P12_RETIRED_INHERITANCE_PROFILE"
    ):
        _validate_metadata(malformed_load_metadata, strict=False)
    with pytest.raises(
        CheckpointStrictError, match="P12_RETIRED_INHERITANCE_PROFILE"
    ):
        validate_inheritance_checkpoint_compatibility(
            repaired,
            {
                "metadata": {
                    "strict_replay_mode": False,
                    "inheritance": mismatched,
                }
            },
        )

    disposition = inspect_historical_inheritance_checkpoint_v1(
        repaired_contract
    )
    assert disposition["decision"] == "P12-A"
    assert disposition["historical_analysis_readable"] is True
    assert disposition["production_execution_allowed"] is False
    assert disposition["production_resume_allowed"] is False
    assert disposition["migration_supported"] is False
    assert disposition["migration_rng_draws"] == 0
    with pytest.raises(ValueError, match="contract mismatch"):
        inspect_historical_inheritance_checkpoint_v1(mismatched)


def test_historical_bundle_inspector_reads_metadata_only(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "historical-linked-bundle"
    bundle.mkdir()
    metadata = _metadata(LEGACY_INHERITANCE_PROFILE)
    metadata["inheritance"] = historical_linked_checkpoint_metadata_v1()
    (bundle / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True),
        encoding="utf-8",
    )
    (bundle / "world.json").write_text("not-json", encoding="utf-8")
    (bundle / "rng_snapshot_v2.json").write_text("not-json", encoding="utf-8")

    inspection = checkpoint_bundle_v2_inspect_historical(str(bundle))

    assert inspection["disposition"]["authority_status"] == "historical"
    assert inspection["disposition"]["access_mode"] == "read_only"
    assert inspection["disposition"]["resume_status"] == "non_resumable"
    assert inspection["disposition"]["state_loaded"] is False
    assert inspection["disposition"]["rng_loaded"] is False
    with pytest.raises(
        CheckpointStrictError,
        match="P12_RETIRED_INHERITANCE_PROFILE",
    ):
        checkpoint_bundle_v2_load(str(bundle), strict=False)


def test_retired_flat_linkage_mechanic_is_subordinate_to_diploid_genome() -> None:
    contract = diploid_flat_linkage_component_contract_v1()

    assert contract["standalone_profile_id"] is None
    assert contract["retired_source_profile"] == LINKED_INHERITANCE_PROFILE
    assert contract["representation"]["draw_count"] == LINKED_PROFILE_DRAW_COUNT
    assert tuple(contract["representation"]["linkage_blocks"]) == LINKAGE_BLOCK_IDS
    assert contract["route_qualification"] == (
        "cannot_qualify_or_execute_standalone"
    )
    assert contract["native_cuda_ownership"] == (
        "qualified_genome_v2_package_and_execution_route_only"
    )


def test_rng_governance_has_fixed_recomb1_boundary() -> None:
    with open("governance/rng_budget_contract.json", "rb") as handle:
        raw = handle.read()
    contract = json.loads(raw)
    boundary = contract["boundaries"]["mutation.reproduction.inheritance_linkage"]
    assert boundary["expected_delta"] == {"mutation.py": 7}

    with open("governance/rng_budget_contract.lock.json", "r", encoding="utf-8") as handle:
        lock = json.load(handle)
    assert lock["contract_sha256"] == hashlib.sha256(raw).hexdigest()
    assert lock["contract_version"] == contract["contract_version"]
