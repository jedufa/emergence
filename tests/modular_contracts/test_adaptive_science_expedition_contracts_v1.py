from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from emergence_contracts.adaptive_science_expedition_v1 import (
    DecisionV1,
    ExpeditionManifestV1,
    EvolutionaryAtlasV1,
    GeneratedChildContinuationV1,
    JournalRecordV1,
    ObservationFrameV1,
    RetainedParentBankV1,
    StateRefV1,
)
from emergence_contracts.canonical import canonical_sha256
from emergence_contracts.errors import ContractError
from emergence_contracts.payloads import CanonicalPayloadV1


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = (
    REPO_ROOT
    / "_spec"
    / "adaptive_science_expedition"
    / "v1"
    / "adaptive_science_expedition_contracts_v1.schema.json"
)


def _digest(label: str) -> dict[str, str]:
    return canonical_sha256({"label": label}).to_dict()


def _payload(version: str, value: object) -> dict[str, object]:
    return {"schema_version": version, "payload": value}


def _payload_digest(payload: dict[str, object]) -> dict[str, str]:
    return CanonicalPayloadV1.from_dict(payload).content_digest().to_dict()


def _artifact(label: str, *, size_bytes: int = 128) -> dict[str, object]:
    return {
        "schema_version": "run_artifact_ref_v1",
        "artifact_id": f"artifact-{label}",
        "artifact_kind": "native_checkpoint",
        "artifact_schema_version": "native_checkpoint_v1",
        "locator": f"opaque://lifecycle/{label}",
        "content_digest": _digest(f"artifact-{label}"),
        "size_bytes": size_bytes,
    }


def _state(*, state_class: str = "BRANCHABLE_EXACT") -> dict[str, object]:
    artifact = _artifact("state")
    exact = state_class in {"BRANCHABLE_EXACT", "REPLAY_DERIVABLE"}
    return {
        "schema_version": "eiel_adaptive_science_state_ref_v1",
        "expedition_id": "expedition-a",
        "world_trajectory_id": "trajectory-a",
        "logical_world_id": 7,
        "next_absolute_tick": 480,
        "state_class": state_class,
        "semantic_world_sha256": _digest("world"),
        "control_history_sha256": _digest("controls"),
        "source_object_sha256": artifact["content_digest"],
        "source_artifact_ref": artifact,
        "exact_state_sha256": _digest("exact-state") if exact else None,
        "checkpoint_occurrence_sha256": _digest("checkpoint-occurrence") if exact else None,
        "rng_state_sha256": _digest("rng") if exact else None,
        "event_prefix_sha256": _digest("event-prefix") if exact else None,
        "parent_occurrence_sha256": _digest("parent-occurrence") if exact else None,
        "replay_recipe_sha256": (
            _digest("replay-recipe") if state_class == "REPLAY_DERIVABLE" else None
        ),
    }


def _objects() -> tuple[dict[str, dict[str, object]], ExpeditionManifestV1]:
    profile = _payload(
        "adaptive_policy_profile_v1",
        {"opaque_policy_configuration": {"quota_profile": "fixture"}},
    )
    manifest_raw: dict[str, object] = {
        "schema_version": "eiel_adaptive_science_expedition_manifest_v1",
        "expedition_id": "expedition-a",
        "question_description": "Map a bounded response surface",
        "no_claims": True,
        "hypothesis_ids": [],
        "required_science_instrument": "EIEL_SCIENCE_INSTRUMENT_V1",
        "founder_basis_refs": [_artifact("founder")],
        "search_space": _payload("adaptive_search_space_v1", {"controls": []}),
        "epoch_ticks": 120,
        "maximum_absolute_tick": 2400,
        "maximum_frontiers": 20,
        "maximum_actions_per_frontier": 4,
        "wall_time_budget_seconds": None,
        "observation_profile_id": "fixture-observation-profile",
        "observation_descriptor_set_sha256": _digest("descriptor-set"),
        "policy_binding": {
            "policy_id": "fixture-adaptive-policy",
            "policy_version": "fixture-v1",
            "implementation_sha256": _digest("policy-implementation"),
            "profile_sha256": _payload_digest(profile),
            "profile": profile,
        },
        "outer_search_seed": (1 << 64) - 1,
        "maximum_atlas_entries": 4,
        "maximum_retained_parent_entries": 2,
        "maximum_retained_parent_bytes": 512,
        "artifact_lifecycle_policy": _payload(
            "eiel_adaptive_expedition_lifecycle_policy_v1",
            {
                "child_contract_class": "SCIENCE_COMPACT",
                "lifecycle_authority": "canonical_run_artifact_lifecycle",
                "outer_contract": {
                    "schema_version": "eiel_run_artifact_lifecycle_contract_v1",
                    "lifecycle_profile": "SCIENCE_COMPACT",
                    "contract_sha256": _digest("outer-lifecycle")["hex"],
                },
                "retention_authority_created": False,
            },
        ),
        "stop_conditions": ["integrity_failure", "operator_stop"],
    }
    manifest = ExpeditionManifestV1.from_dict(manifest_raw)
    state_raw = _state()
    state = StateRefV1.from_dict(state_raw)
    facts = _payload(
        "fixture_policy_facts_v1",
        {"policy_owned_dimensions": {"arbitrary": "value"}},
    )
    observation_raw: dict[str, object] = {
        "schema_version": "eiel_adaptive_science_observation_frame_v1",
        "expedition_id": "expedition-a",
        "world_trajectory_id": "trajectory-a",
        "logical_world_id": 7,
        "next_absolute_tick": 480,
        "semantic_world_sha256": _digest("world"),
        "control_history_sha256": _digest("controls"),
        "observation_profile_id": "fixture-observation-profile",
        "observation_profile_sha256": _digest("observation-profile"),
        "observation_descriptor_set_sha256": _digest("descriptor-set"),
        "reducer_implementation_sha256": _digest("reducer"),
        "observations": _payload("fixture_observations_v1", {"rows": []}),
        "control_readback": _payload("fixture_control_readback_v1", {"controls": {}}),
        "policy_facing_facts": facts,
        "descriptor_values": _payload("fixture_descriptors_v1", {"axes": {}}),
        "calibration": _payload("fixture_calibration_v1", {"status": "opaque"}),
        "source_state_ref": state_raw,
        "previous_frame_sha256": None,
    }
    observation = ObservationFrameV1.from_dict(observation_raw)
    proposal = _payload(
        "fixture_policy_proposal_v1",
        {
            "candidate_summary": {},
            "quota_allocation": {},
            "atlas_delta": [],
            "parent_bank_delta": [],
        },
    )
    decision_raw: dict[str, object] = {
        "schema_version": "eiel_adaptive_science_decision_v1",
        "expedition_id": "expedition-a",
        "frontier_ordinal": 3,
        "manifest_sha256": manifest.manifest_sha256().to_dict(),
        "observation_frontier_sha256": observation.observation_frame_sha256().to_dict(),
        "atlas_before_sha256": _digest("atlas-before"),
        "policy_implementation_sha256": _digest("policy-implementation"),
        "policy_profile_sha256": _payload_digest(profile),
        "outer_rng_before_sha256": _digest("outer-rng-before"),
        "outer_rng_after_sha256": _digest("outer-rng-after"),
        "policy_proposal": proposal,
        "policy_proposal_sha256": _payload_digest(proposal),
        "action_rows": [
            {
                "action_id": "action-a",
                "action_kind": "REPLICATE",
                "quota_bucket": "confirmation",
                "parent_state_ref_sha256": state.state_ref_sha256().to_dict(),
                "effective_next_absolute_tick": 480,
                "next_epoch_ticks": 120,
                "reason_codes": ["fixture-reason"],
                "action_payload": _payload("fixture_action_v1", {"seed_policy": "opaque"}),
            }
        ],
        "integrity_warnings": [],
    }
    decision = DecisionV1.from_dict(decision_raw)
    child_request = _payload(
        "eiel_science_request_v1",
        {"request_id": "fresh-founder-child"},
    )
    child_raw: dict[str, object] = {
        "schema_version": "eiel_adaptive_science_generated_child_continuation_v1",
        "expedition_id": "expedition-a",
        "manifest_sha256": manifest.manifest_sha256().to_dict(),
        "decision_sha256": decision.decision_sha256().to_dict(),
        "action_id": "action-a",
        "action_kind": "REPLICATE",
        "realization_kind": "FOUNDER_BASIS_REQUEST",
        "parent_state_ref": None,
        "founder_basis_sha256": manifest.founder_basis_sha256().to_dict(),
        "child_trajectory_id": "trajectory-child",
        "child_run_id": "run-child-new-occurrence",
        "retry_of_run_id": None,
        "effective_next_absolute_tick": 0,
        "control_schedule_sha256": _digest("child-control-schedule"),
        "child_request": child_request,
        "child_request_sha256": _payload_digest(child_request),
    }
    child = GeneratedChildContinuationV1.from_dict(child_raw)
    atlas_raw: dict[str, object] = {
        "schema_version": "eiel_adaptive_science_evolutionary_atlas_v1",
        "expedition_id": "expedition-a",
        "manifest_sha256": manifest.manifest_sha256().to_dict(),
        "revision": 0,
        "through_frontier_ordinal": 3,
        "descriptor_set_sha256": _digest("descriptor-set"),
        "maximum_entries": 4,
        "entries": [
            {
                "atlas_entry_id": "atlas-entry-a",
                "world_trajectory_id": "trajectory-a",
                "observation_frame_sha256": observation.observation_frame_sha256().to_dict(),
                "state_ref_sha256": state.state_ref_sha256().to_dict(),
                "niche_key": _payload("fixture_niche_v1", {"niche": "a"}),
                "descriptor_values": _payload("fixture_descriptors_v1", {"axes": {}}),
                "policy_facing_facts": facts,
                "representative_roles": ["future-parent"],
                "parent_entry_ids": [],
            }
        ],
        "previous_atlas_sha256": None,
    }
    atlas = EvolutionaryAtlasV1.from_dict(atlas_raw)
    bank_raw: dict[str, object] = {
        "schema_version": "eiel_adaptive_science_retained_parent_bank_v1",
        "expedition_id": "expedition-a",
        "manifest_sha256": manifest.manifest_sha256().to_dict(),
        "revision": 0,
        "maximum_entries": 2,
        "maximum_total_bytes": 512,
        "current_entry_count": 1,
        "current_total_bytes": 128,
        "entries": [
            {
                "state_ref": state_raw,
                "selected_at_frontier_ordinal": 3,
                "reason_codes": ["future-parent"],
                "retained_bytes": 128,
            }
        ],
        "previous_bank_sha256": None,
    }
    bank = RetainedParentBankV1.from_dict(bank_raw)
    journal_payload = _payload("fixture_journal_event_v1", {"settled": True})
    journal_raw: dict[str, object] = {
        "schema_version": "eiel_adaptive_science_journal_record_v1",
        "expedition_id": "expedition-a",
        "record_sequence": 0,
        "frontier_ordinal": 3,
        "record_kind": "FRONTIER_SETTLED",
        "manifest_sha256": manifest.manifest_sha256().to_dict(),
        "subject_sha256": decision.decision_sha256().to_dict(),
        "payload": journal_payload,
        "payload_sha256": _payload_digest(journal_payload),
        "previous_record_sha256": None,
    }
    journal = JournalRecordV1.from_dict(journal_raw)
    return (
        {
            "manifest": manifest_raw,
            "observation": observation_raw,
            "state": state_raw,
            "decision": decision_raw,
            "child": child_raw,
            "journal": journal_raw,
            "atlas": atlas_raw,
            "bank": bank_raw,
        },
        manifest,
    )


_PARSERS = {
    "manifest": ExpeditionManifestV1.from_dict,
    "observation": ObservationFrameV1.from_dict,
    "state": StateRefV1.from_dict,
    "decision": DecisionV1.from_dict,
    "child": GeneratedChildContinuationV1.from_dict,
    "journal": JournalRecordV1.from_dict,
    "atlas": EvolutionaryAtlasV1.from_dict,
    "bank": RetainedParentBankV1.from_dict,
}


def test_all_durable_contracts_round_trip_and_validate_against_strict_schema() -> None:
    objects, manifest = _objects()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    for name, raw in objects.items():
        restored = _PARSERS[name](raw)
        assert restored.to_dict() == raw
        jsonschema.validate(raw, schema)
    bank = RetainedParentBankV1.from_dict(objects["bank"])
    bank.validate_against_manifest(manifest)


@pytest.mark.parametrize("name", sorted(_PARSERS))
def test_unknown_and_generation_shaped_fields_fail_closed(name: str) -> None:
    objects, _ = _objects()
    mutated = copy.deepcopy(objects[name])
    mutated["generation"] = 3
    with pytest.raises(ContractError):
        _PARSERS[name](mutated)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(mutated, schema)


def test_state_ref_does_not_promote_observation_bytes_to_branch_authority() -> None:
    observation_only = _state(state_class="OBSERVATION_ONLY")
    state = StateRefV1.from_dict(observation_only)
    assert not state.supports_exact_continuation
    promoted = copy.deepcopy(observation_only)
    promoted["state_class"] = "BRANCHABLE_EXACT"
    with pytest.raises(ContractError):
        StateRefV1.from_dict(promoted)
    wrong_source = copy.deepcopy(_state())
    wrong_source["source_object_sha256"] = _digest("not-the-wrapped-artifact")
    with pytest.raises(ContractError):
        StateRefV1.from_dict(wrong_source)


def test_founder_child_restarts_at_zero_and_exact_child_cannot_precede_parent() -> None:
    objects, _ = _objects()
    founder = objects["child"]
    assert GeneratedChildContinuationV1.from_dict(founder).effective_next_absolute_tick == 0
    bad_founder = copy.deepcopy(founder)
    bad_founder["effective_next_absolute_tick"] = 480
    with pytest.raises(ContractError):
        GeneratedChildContinuationV1.from_dict(bad_founder)
    bad_exact = copy.deepcopy(founder)
    bad_exact["realization_kind"] = "EXACT_STATE_REQUEST"
    bad_exact["effective_next_absolute_tick"] = 479
    with pytest.raises(ContractError):
        GeneratedChildContinuationV1.from_dict(bad_exact)


def test_policy_and_bank_digests_and_hard_accounting_are_not_advisory() -> None:
    objects, manifest = _objects()
    wrong_profile = copy.deepcopy(objects["manifest"])
    wrong_profile["policy_binding"]["profile"]["payload"] = {"changed": True}  # type: ignore[index]
    with pytest.raises(ContractError):
        ExpeditionManifestV1.from_dict(wrong_profile)
    wrong_proposal = copy.deepcopy(objects["decision"])
    wrong_proposal["policy_proposal"]["payload"] = {"changed": True}  # type: ignore[index]
    with pytest.raises(ContractError):
        DecisionV1.from_dict(wrong_proposal)
    wrong_count = copy.deepcopy(objects["bank"])
    wrong_count["current_entry_count"] = 0
    with pytest.raises(ContractError):
        RetainedParentBankV1.from_dict(wrong_count)
    wrong_cap = copy.deepcopy(objects["bank"])
    wrong_cap["maximum_total_bytes"] = 127
    with pytest.raises(ContractError):
        RetainedParentBankV1.from_dict(wrong_cap)
    bank = RetainedParentBankV1.from_dict(objects["bank"])
    changed_manifest = copy.deepcopy(objects["manifest"])
    changed_manifest["maximum_retained_parent_bytes"] = 1024
    with pytest.raises(ContractError):
        bank.validate_against_manifest(ExpeditionManifestV1.from_dict(changed_manifest))
    bank.validate_against_manifest(manifest)


def test_core_contract_contains_no_first_policy_science_vocabulary() -> None:
    source = (
        REPO_ROOT / "emergence_contracts" / "adaptive_science_expedition_v1.py"
    ).read_text(encoding="utf-8")
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    forbidden = (
        "pressure_readback",
        "pressure_engagement",
        "sibling_divergence",
        "evolvability_proxy",
        "scientific_state",
    )
    assert all(term not in source for term in forbidden)
    assert all(term not in schema for term in forbidden)
