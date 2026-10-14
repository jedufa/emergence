from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from eiel_composition.adaptive_expedition.observation_adapter import (
    AuthenticatedObservationError,
    project_authenticated_science_request_observations_v1,
)
from eiel_composition.adaptive_expedition import observation_adapter as adapter_module
from eiel_composition.adaptive_expedition.lifecycle import (
    default_expedition_lifecycle_policy_value_v1,
)
from emergence_contracts.adaptive_science_expedition_v1 import (
    ExpeditionManifestV1,
    StateClassV1,
)
from emergence_contracts.canonical import canonical_json_bytes, canonical_sha256
from emergence_contracts.payloads import CanonicalPayloadV1
from emergence_evidence.science_campaign_atlas_v1 import (
    SCIENCE_CAMPAIGN_ATLAS_RESULT_SCHEMA_V1,
    SCIENCE_CAMPAIGN_ATLAS_VERIFICATION_RECEIPT_SCHEMA_V1,
)
from emergence_evidence.science_compact_world_atlas_v1 import (
    COMPACT_WORLD_ATLAS_CONTRACT_VERSION,
    COMPACT_WORLD_ATLAS_OUTPUT_CONTRACT_ID,
    COMPACT_WORLD_ATLAS_REDUCER_ID,
    compact_world_atlas_reducer_descriptor_v1,
)
from engine.perf.cuda_full_evolution_v2.workhorse import (
    BACKEND_FUSED_CTA_TEMPORAL_V2,
    SUPPORTED_WORKHORSE_SHAPES_V2,
    FullEvolutionCudaWorkhorseContractV2,
    WORKHORSE_SELECTED_WORLD_OBSERVATION_MODE_V1,
    workhorse_scientific_control_contract_v1,
)


def _digest(label: str) -> dict[str, str]:
    return canonical_sha256({"fixture": label}).to_dict()


def _canonical_hex(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value))


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _payload(schema_version: str, value: object) -> dict[str, object]:
    return {"schema_version": schema_version, "payload": value}


def _artifact(label: str) -> dict[str, object]:
    return {
        "schema_version": "run_artifact_ref_v1",
        "artifact_id": f"artifact-{label}",
        "artifact_kind": "founder_basis",
        "artifact_schema_version": "founder_basis_v1",
        "locator": f"opaque://founder/{label}",
        "content_digest": _digest(f"artifact-{label}"),
        "size_bytes": 128,
    }


def _expedition_manifest() -> ExpeditionManifestV1:
    profile = _payload(
        "fixture_policy_profile_v1",
        {"version": "1", "opaque_configuration": {}},
    )
    profile_digest = CanonicalPayloadV1.from_dict(profile).content_digest().to_dict()
    return ExpeditionManifestV1.from_dict(
        {
            "schema_version": "eiel_adaptive_science_expedition_manifest_v1",
            "expedition_id": "expedition-observation-fixture",
            "question_description": "Exercise authenticated observation projection",
            "no_claims": True,
            "hypothesis_ids": [],
            "required_science_instrument": "EIEL_SCIENCE_INSTRUMENT_V1",
            "founder_basis_refs": [_artifact("a")],
            "search_space": _payload("fixture_search_space_v1", {"controls": []}),
            "epoch_ticks": 40,
            "maximum_absolute_tick": 400,
            "maximum_frontiers": 10,
            "maximum_actions_per_frontier": 4,
            "wall_time_budget_seconds": None,
            "observation_profile_id": "fixture-authenticated-atlas-profile-v1",
            "observation_descriptor_set_sha256": _digest("descriptor-set"),
            "policy_binding": {
                "policy_id": "fixture-policy-v1",
                "policy_version": "1",
                "implementation_sha256": _digest("policy-implementation"),
                "profile_sha256": profile_digest,
                "profile": profile,
            },
            "outer_search_seed": 7,
            "maximum_atlas_entries": 32,
            "maximum_retained_parent_entries": 4,
            "maximum_retained_parent_bytes": 4096,
            "artifact_lifecycle_policy": _payload(
                "eiel_adaptive_expedition_lifecycle_policy_v1",
                default_expedition_lifecycle_policy_value_v1(),
            ),
            "stop_conditions": ["integrity_failure"],
        }
    )


def _world_result(*, include_resident_capacity: bool = True) -> dict[str, object]:
    body: dict[str, object] = {
        "world_id": "world-fixture-0000",
        "science": {
            "test_id": "OPAQUE-TEST",
            "test_version": "1",
            "study_id": "OPAQUE-STUDY",
            "arm_id": "opaque-arm",
            "replicate": 11,
        },
        "dimension_bindings": {"opaque_control": 3},
        "semantic_world_sha256": _digest("semantic-world"),
        "acquisition_sha256": _digest("acquisition"),
        "birth_count": 4,
        "death_count": 2,
        "maximum_lineage_depth": 3,
        "milestones": [
            {
                "absolute_tick": 0,
                "active_agent_count": 2,
                "world_state": {"population_count": 2},
            },
            {
                "absolute_tick": 40,
                "active_agent_count": 4,
                "world_state": {"population_count": 4},
                "genome_reconstruction": {
                    "active_agent_count": 4,
                    "maximum_active_lineage_depth": 3,
                },
            },
        ],
        "integrity": {
            "dispatch_count": 1,
            "event_count": 6,
            "snapshot_count": 2,
            "dropped_record_count": 0,
            "dispatch_tick_count": 40,
            "genome_state_identity_checks": 6,
        },
    }
    if include_resident_capacity:
        body["resident_capacity"] = 512
    return {**body, "result_sha256": canonical_sha256(body).to_dict()}


def _retained_science_root(
    tmp_path: Path, *, include_resident_capacity: bool = True
) -> Path:
    root = tmp_path / "run-fixture-0001"
    root.mkdir()
    request_sha256 = canonical_sha256({"science_request": "fixture"})
    bundle_sha256 = _digest("bundle")
    batch_result_sha256 = _digest("batch-atlas-result")
    capture_index_sha256 = _digest("capture-index")
    descriptor = compact_world_atlas_reducer_descriptor_v1()
    world = _world_result(include_resident_capacity=include_resident_capacity)
    result_body: dict[str, object] = {
        "schema_version": SCIENCE_CAMPAIGN_ATLAS_RESULT_SCHEMA_V1,
        "request_sha256": request_sha256.to_dict(),
        "bundle_content_sha256": bundle_sha256,
        "reducer_id": COMPACT_WORLD_ATLAS_REDUCER_ID,
        "reducer_contract_version": str(COMPACT_WORLD_ATLAS_CONTRACT_VERSION),
        "reducer_descriptor_sha256": descriptor.descriptor_sha256().to_dict(),
        "output_contract_id": COMPACT_WORLD_ATLAS_OUTPUT_CONTRACT_ID,
        "logical_world_count": 1,
        "physical_batch_count": 1,
        "physical_composition_used": False,
        "source_batch_result_sha256s": [batch_result_sha256],
        "world_results": [world],
    }
    result = {
        **result_body,
        "result_sha256": canonical_sha256(result_body).to_dict(),
    }
    semantic_order_sha256 = canonical_sha256(
        [
            {
                "world_id": world["world_id"],
                "semantic_world_sha256": world["semantic_world_sha256"],
                "acquisition_sha256": world["acquisition_sha256"],
            }
        ]
    )
    verification_body: dict[str, object] = {
        "schema_version": SCIENCE_CAMPAIGN_ATLAS_VERIFICATION_RECEIPT_SCHEMA_V1,
        "request_sha256": request_sha256.to_dict(),
        "bundle_content_sha256": bundle_sha256,
        "reducer_id": COMPACT_WORLD_ATLAS_REDUCER_ID,
        "reducer_contract_version": str(COMPACT_WORLD_ATLAS_CONTRACT_VERSION),
        "reducer_descriptor_sha256": descriptor.descriptor_sha256().to_dict(),
        "output_contract_id": COMPACT_WORLD_ATLAS_OUTPUT_CONTRACT_ID,
        "aggregate_result_sha256": result["result_sha256"],
        "semantic_world_order_sha256": semantic_order_sha256.to_dict(),
        "logical_world_count": 1,
        "physical_batch_count": 1,
        "batch_results": [
            {
                "physical_batch_index": 0,
                "batch_result_sha256": batch_result_sha256,
                "batch_bundle_content_sha256": bundle_sha256,
                "capture_index_sha256": capture_index_sha256,
                "logical_world_count": 1,
                "world_ids": [world["world_id"]],
            }
        ],
        "verification": {
            "batch_result_digests_authenticated": True,
            "world_result_digests_authenticated": True,
            "complete_disjoint_world_coverage": True,
            "semantic_world_order_canonical": True,
        },
        "trajectory_effects_performed": False,
        "authority_effect": "none",
    }
    verification = {
        **verification_body,
        "receipt_sha256": canonical_sha256(verification_body).to_dict(),
    }
    result_path = root / "campaign_atlas_result.json"
    verification_path = root / "campaign_atlas_verification_receipt.json"
    _write(result_path, result)
    _write(verification_path, verification)

    batch_path = root / "batch_result_0000.json"
    batch_body = {
        "schema_version": "fixture_science_batch_result_v1",
        "ok": True,
        "status": "completed",
        "result_class": "SUCCESS",
    }
    _write(batch_path, batch_body)
    plan_sha256 = _canonical_hex({"plan": "fixture"})
    run_receipt_body: dict[str, object] = {
        "schema_version": "eiel_science_request_v1_run_receipt_v1",
        "ok": True,
        "status": "COMPLETED",
        "result_class": "SCIENCE_CAMPAIGN_COMPLETED",
        "campaign_id": "campaign-fixture",
        "request_sha256": request_sha256.to_dict(),
        "runtime_root": str(root),
        "materialization_root": str(tmp_path / "materialization-fixture"),
        "materialization_plan_receipt_sha256": _canonical_hex(
            {"materialization": "fixture"}
        ),
        "authority_sha256": _canonical_hex({"authority": "fixture"}),
        "completed_batches": [
            {
                "batch_index": 0,
                "path": str(batch_path),
                "file_sha256": hashlib.sha256(batch_path.read_bytes()).hexdigest(),
                "result_sha256": _digest("batch-result"),
                "ok": True,
                "status": "completed",
                "result_class": "SUCCESS",
                "runtime_root": str(root / "batches/batch-0000/attempt-0001"),
                "materialized_world_ids": [world["world_id"]],
                "child_lifecycle": {"terminal_status": "completed"},
            }
        ],
        "aggregate": {
            "result": {
                "path": str(result_path),
                "file_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            },
            "verification_receipt": {
                "path": str(verification_path),
                "file_sha256": hashlib.sha256(
                    verification_path.read_bytes()
                ).hexdigest(),
            },
            "result_sha256": result["result_sha256"],
            "verification_receipt_sha256": verification["receipt_sha256"],
        },
        "batch_execution_authority": "canonical_launchpad_per_batch",
        "lifecycle": {
            "plan_sha256": plan_sha256,
            "run_passport_path": str(root / "run_passport.json"),
            "artifact_disposition_manifest_path": str(
                root / "artifact_disposition_manifest.json"
            ),
            "storage_disposition_receipt_path": str(
                root / "storage_disposition_receipt.json"
            ),
        },
        "reduction_completed": True,
        "aggregation_completed": True,
        "closeout_completed": True,
        "closeout_completion_condition": (
            "authoritative only when the terminal lifecycle artifacts validate"
        ),
        "execution_started": True,
        "prequal_started": False,
        "qualification_started": False,
        "qualification_authority_effect": False,
    }
    run_receipt = {
        **run_receipt_body,
        "receipt_sha256": canonical_sha256(run_receipt_body).to_dict(),
    }
    run_receipt_path = root / "science_campaign_run_receipt.json"
    _write(run_receipt_path, run_receipt)

    lifecycle_sha256 = _canonical_hex({"lifecycle": "fixture"})
    passport_body: dict[str, object] = {
        "schema_version": "eiel_run_passport_v1",
        "campaign_id": "campaign-fixture",
        "run_id": root.name,
        "plan_sha256": plan_sha256,
        "qualification_key_sha256": None,
        "source_identity": {"request_sha256": request_sha256.to_dict()},
        "config_identity": _canonical_hex({"config": "fixture"}),
        "seed_identity": _canonical_hex({"seed": "fixture"}),
        "provider_identity": {"provider_id": "fixture"},
        "checkpoint_catalog": [],
        "terminal_status": "completed",
        "terminal_digest": None,
        "lifecycle_profile": "SCIENCE_COMPACT",
        "lifecycle_contract_sha256": lifecycle_sha256,
        "disposition_manifest_path": str(root / "artifact_disposition_manifest.json"),
        "disposition_receipt_path": str(root / "storage_disposition_receipt.json"),
    }
    passport = {
        **passport_body,
        "passport_sha256": _canonical_hex(passport_body),
    }
    passport_path = root / "run_passport.json"
    _write(passport_path, passport)

    retained = (
        (result_path, "R4", "science.compact.campaign_atlas"),
        (verification_path, "R5", "lifecycle.control"),
        (run_receipt_path, "R5", "lifecycle.control"),
        (batch_path, "R4", "science.compact.batch_result"),
        (passport_path, "R5", "lifecycle.control"),
    )
    artifacts = [
        {
            "path": path.name,
            "artifact_family": family,
            "retention_class": retention,
            "owner_writer": "fixture",
            "release_predicate": "audit closeout",
            "artifact_type": "file",
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "exact_object_identity": True,
            "release_eligible": False,
        }
        for path, retention, family in retained
    ]
    manifest_body: dict[str, object] = {
        "schema_version": "eiel_artifact_disposition_manifest_v1",
        "campaign_id": "campaign-fixture",
        "run_id": root.name,
        "plan_sha256": plan_sha256,
        "runtime_root": str(root),
        "terminal_status": "completed",
        "lifecycle_profile": "SCIENCE_COMPACT",
        "lifecycle_contract_sha256": lifecycle_sha256,
        "inventory": {"mode": "producer_declared", "complete": True},
        "checkpoint_catalog": [],
        "artifacts": artifacts,
        "replacement_artifacts": [],
        "retention": {
            "audit_hold_status": "active",
            "qualification_hold_status": "not_required",
            "unknown_artifact_policy": "RX_BLOCKS_AUTO_DELETE",
            "unknown_artifact_scope": [],
        },
        "summary": {
            "artifact_count": len(artifacts),
            "produced_bytes": sum(int(row["bytes"]) for row in artifacts),
            "unresolved_rx_bytes": 0,
        },
        "prune": {"automatic_prune_allowed": False},
    }
    manifest = {
        **manifest_body,
        "manifest_sha256": _canonical_hex(manifest_body),
    }
    manifest_path = root / "artifact_disposition_manifest.json"
    _write(manifest_path, manifest)
    storage_body: dict[str, object] = {
        "schema_version": "eiel_storage_disposition_receipt_v1",
        "campaign_id": "campaign-fixture",
        "run_id": root.name,
        "lifecycle_contract_sha256": lifecycle_sha256,
        "manifest_sha256": manifest["manifest_sha256"],
        "plan_sha256": plan_sha256,
        "bytes_before": manifest_body["summary"]["produced_bytes"],  # type: ignore[index]
        "bytes_retained": manifest_body["summary"]["produced_bytes"],  # type: ignore[index]
        "bytes_released": 0,
        "bytes_unresolved_RX": 0,
        "removed": [],
        "authorization_reason": "dry_run_only_no_destructive_cleanup_authority",
        "replacement_references": [],
        "retention_references": [str(manifest_path)],
        "timestamp_utc": "2026-08-28T00:00:00+00:00",
        "result": "dry_run",
    }
    storage = {
        **storage_body,
        "receipt_sha256": _canonical_hex(storage_body),
    }
    _write(root / "storage_disposition_receipt.json", storage)
    return root


def _reseal_outer_files(root: Path) -> None:
    result_path = root / "campaign_atlas_result.json"
    verification_path = root / "campaign_atlas_verification_receipt.json"
    run_receipt_path = root / "science_campaign_run_receipt.json"
    receipt = _read(run_receipt_path)
    aggregate = receipt["aggregate"]
    assert isinstance(aggregate, dict)
    result = _read(result_path)
    verification = _read(verification_path)
    aggregate["result"] = {
        "path": str(result_path),
        "file_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
    }
    aggregate["verification_receipt"] = {
        "path": str(verification_path),
        "file_sha256": hashlib.sha256(verification_path.read_bytes()).hexdigest(),
    }
    aggregate["result_sha256"] = result["result_sha256"]
    aggregate["verification_receipt_sha256"] = verification["receipt_sha256"]
    receipt_body = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt_body).to_dict()
    _write(run_receipt_path, receipt)

    manifest_path = root / "artifact_disposition_manifest.json"
    manifest = _read(manifest_path)
    artifacts = manifest["artifacts"]
    assert isinstance(artifacts, list)
    changed = {result_path.name, verification_path.name, run_receipt_path.name}
    for raw in artifacts:
        assert isinstance(raw, dict)
        if raw["path"] in changed:
            path = root / str(raw["path"])
            raw["bytes"] = path.stat().st_size
            raw["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = manifest["summary"]
    assert isinstance(summary, dict)
    summary["produced_bytes"] = sum(int(row["bytes"]) for row in artifacts)
    manifest_body = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    manifest["manifest_sha256"] = _canonical_hex(manifest_body)
    _write(manifest_path, manifest)

    storage_path = root / "storage_disposition_receipt.json"
    storage = _read(storage_path)
    storage["manifest_sha256"] = manifest["manifest_sha256"]
    storage["bytes_before"] = summary["produced_bytes"]
    storage["bytes_retained"] = summary["produced_bytes"]
    storage_body = {
        key: value for key, value in storage.items() if key != "receipt_sha256"
    }
    storage["receipt_sha256"] = _canonical_hex(storage_body)
    _write(storage_path, storage)


def _retained_selected_world_host_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    root = _retained_science_root(tmp_path)
    result_path = root / "campaign_atlas_result.json"
    verification_path = root / "campaign_atlas_verification_receipt.json"
    result_path.unlink()
    verification_path.unlink()

    attempt_root = root / "batches" / "batch-0000" / "attempt-0001"
    host_root = attempt_root / "batches" / "workhorse-selected"
    host_root.mkdir(parents=True)
    observation_path = host_root / "workhorse-selected-world-host-observation.json"
    source_result_path = host_root / "workhorse-indexed-continuation-result.json"
    shape = SUPPORTED_WORKHORSE_SHAPES_V2[0]
    contract = FullEvolutionCudaWorkhorseContractV2(
        shape_id=shape.shape_id,
        backend_id=BACKEND_FUSED_CTA_TEMPORAL_V2,
        B=1,
        temporal_k=1,
        total_ticks=5,
        start_next_tick=25,
    )
    logical = {
        "science_continuation_identity_sha256": "1" * 64,
        "science_continuation_contract_sha256": "2" * 64,
        "source_control_history_sha256": "3" * 64,
        "target_next_tick": 30,
    }
    scientific_control = workhorse_scientific_control_contract_v1(
        effective_next_absolute_tick=25,
        resource_multiplier=1.0,
        mutation_rate=0.125,
    )
    continuation_result = {
        "continuation_contract": {
            "science_continuation_identity": logical,
            "physical_continuation_plan": {
                "target_contract": contract.to_dict(),
            },
            "scientific_control_contract": scientific_control,
        }
    }
    observation_body: dict[str, object] = {
        "schema_version": "eiel_workhorse_selected_world_host_observation_v1",
        "observation_mode": WORKHORSE_SELECTED_WORLD_OBSERVATION_MODE_V1,
        "claim_class": (
            "AUTHENTICATED_SCIENTIFIC_OBSERVATION_NO_EXECUTION_OR_QUALIFICATION_AUTHORITY"
        ),
        "time_authority": "absolute_tick_v1",
        "observation_boundary": (
            "HOST_PROJECTION_AFTER_ACKNOWLEDGED_TERMINAL_CHECKPOINT_COMMIT"
        ),
        "execution_authority_granted": False,
        "qualification_authority_granted": False,
        "named_phenotype_reward_authority_granted": False,
        "cell_id": "world-selected-0001",
        "source_continuation": {
            "execution_result_path": str(source_result_path),
        },
        "terminal_checkpoint": {
            "world_state_set_sha256": "4" * 64,
            "checkpoint_at_tick": 30,
            "checkpoint_occurrence_sha256": "5" * 64,
            "checkpoint_occurrence_projection_sha256": "6" * 64,
        },
        "control_readback": {
            "control_contract_sha256": scientific_control[
                "control_contract_sha256"
            ],
            "applied_next_absolute_tick": 25,
            "first_permitted_divergence_tick": 25,
            "ordered_native_fields": [
                "mutation_threshold_u16",
                "resource_multiplier_q16",
            ],
            "requested_native_rows": [[8192, 65536]],
            "readback_native_rows": [[8192, 65536]],
            "control_application_sha256": "8" * 64,
        },
        "event_observation_coverage": {
            "coverage": "ACKNOWLEDGED_COUNTS_AND_DIGESTS_ONLY",
            "event_kind_rows_available": False,
            "event_kind_rows_unavailable_reason": (
                "TECHNICAL_DIGEST_CONSUMER_RELEASED_RAW_PACKED_EVENTS"
            ),
            "birth_death_causal_rows_claimed": False,
            "total_event_count": 4,
            "segments": [],
            "coverage_sha256": "9" * 64,
        },
        "world_observations": [
            {
                "world_uid": 1902,
                "next_absolute_tick": 30,
                "rng_key": 302,
                "rng_epoch": 4,
                "active_population": 2,
                "alive_count": 2,
                "occupied_site_count": 2,
                "cumulative_counters": {
                    "ticks_completed": 30,
                    "birth_count": 3,
                    "death_count": 1,
                    "mutation_count": 2,
                    "recombination_count": 1,
                },
                "site_field_totals": {"resource_total": 8192},
                "active_agent_aggregates": {
                    "energy": {"count": 2, "sum": 210, "minimum": 90, "maximum": 120},
                    "health": {"count": 2, "sum": 430, "minimum": 200, "maximum": 230},
                    "age": {"count": 2, "sum": 11, "minimum": 3, "maximum": 8},
                },
                "active_lineage_summary": {
                    "active_lineage_sha256": "a" * 64,
                    "unique_parent_pair_count": 2,
                    "active_founder_count": 1,
                },
                "active_genotype_summary": {
                    "genotype_multiset_sha256": "b" * 64,
                    "unique_genotype_count": 2,
                    "locus_value_histograms": [],
                },
                "canonical_world_state_sha256": "c" * 64,
            }
        ],
        "integrity": {
            "world_count": 1,
            "dropped_record_count": 0,
            "checkpoint_state_reopened_exact": True,
            "event_acknowledgements_reopened_exact": True,
            "control_readback_authenticated": True,
            "runtime_observation_mode_remains_off": True,
            "host_boundary_adapter_is_observation_only": True,
        },
    }
    observation_body["observation_content_sha256"] = _canonical_hex(
        observation_body
    )
    _write(observation_path, observation_body)
    observation_path.chmod(0o444)
    file_sha256 = hashlib.sha256(observation_path.read_bytes()).hexdigest()
    binding = {
        "observation_mode": WORKHORSE_SELECTED_WORLD_OBSERVATION_MODE_V1,
        "path": str(observation_path.resolve()),
        "file_sha256": file_sha256,
        "content_sha256": observation_body["observation_content_sha256"],
        "file_bytes": observation_path.stat().st_size,
        "checkpoint_occurrence_sha256": "5" * 64,
        "checkpoint_occurrence_projection_sha256": "6" * 64,
        "terminally_reopened": True,
        "runtime_observation_mode": "OFF",
        "host_adapter_observation_only": True,
    }
    terminal = {
        "operation": "INDEXED_ACS_CONTINUATION",
        "host_boundary_observation": binding,
    }
    batch_result_body: dict[str, object] = {
        "schema_version": "eiel_science_instrument_execution_result_v1",
        "engineering_only": True,
        "scientific_execution": False,
        "ok": True,
        "status": "completed",
        "result_class": "SUCCESS",
        "front_door_result": {
            "workhorse_terminalized_operation": terminal,
            "workhorse_terminalized_indexed_continuation": terminal,
        },
        "compact_reducer": {
            "enabled": False,
            "status": "NOT_REQUESTED_BY_BATCH",
        },
        "selected_world_host_observation": binding,
    }
    batch_result = {
        **batch_result_body,
        "result_sha256": canonical_sha256(batch_result_body).hex,
    }
    batch_path = root / "batch_result_0000.json"
    _write(batch_path, batch_result)

    run_receipt_path = root / "science_campaign_run_receipt.json"
    receipt = _read(run_receipt_path)
    completed = receipt["completed_batches"]
    assert isinstance(completed, list) and isinstance(completed[0], dict)
    completed[0].update(
        {
            "path": str(batch_path),
            "file_sha256": hashlib.sha256(batch_path.read_bytes()).hexdigest(),
            "result_sha256": batch_result["result_sha256"],
            "runtime_root": str(attempt_root),
            "materialized_world_ids": ["world-selected-0001"],
            "selected_world_host_observation": binding,
        }
    )
    host_row = {
        "physical_batch_index": 0,
        "world_id": "world-selected-0001",
        "host_boundary_observation": binding,
    }
    receipt["aggregate"] = None
    receipt["host_observation_closeout"] = {
        "mode": "AUTHENTICATED_SELECTED_WORLD_HOST_BOUNDARY_V1",
        "world_count": 1,
        "physical_batch_count": 1,
        "observations": [host_row],
        "runtime_observation_mode": "OFF",
        "host_adapter_observation_only": True,
        "atlas_generated": False,
        "causal_event_rows_available": False,
        "individual_lifetime_histories_available": False,
    }
    receipt["reduction_completed"] = False
    receipt["aggregation_completed"] = False
    receipt["observation_closeout_completed"] = True
    receipt_body = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt_body).to_dict()
    _write(run_receipt_path, receipt)

    manifest_path = root / "artifact_disposition_manifest.json"
    manifest = _read(manifest_path)
    artifacts = manifest["artifacts"]
    assert isinstance(artifacts, list)
    artifacts[:] = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, dict)
        and artifact["path"]
        not in {result_path.name, verification_path.name}
    ]
    for artifact in artifacts:
        assert isinstance(artifact, dict)
        if artifact["path"] in {batch_path.name, run_receipt_path.name}:
            path = root / str(artifact["path"])
            artifact["bytes"] = path.stat().st_size
            artifact["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    artifacts.append(
        {
            "path": str(observation_path.relative_to(root)),
            "artifact_family": (
                "science.workhorse_selected_world_host_observation_v1"
            ),
            "retention_class": "R4",
            "owner_writer": "fixture",
            "release_predicate": "audit closeout",
            "artifact_type": "file",
            "bytes": observation_path.stat().st_size,
            "sha256": file_sha256,
            "exact_object_identity": True,
            "release_eligible": False,
        }
    )
    summary = manifest["summary"]
    assert isinstance(summary, dict)
    summary["artifact_count"] = len(artifacts)
    summary["produced_bytes"] = sum(int(row["bytes"]) for row in artifacts)
    manifest_body = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    manifest["manifest_sha256"] = _canonical_hex(manifest_body)
    _write(manifest_path, manifest)
    storage_path = root / "storage_disposition_receipt.json"
    storage = _read(storage_path)
    storage["manifest_sha256"] = manifest["manifest_sha256"]
    storage["bytes_before"] = summary["produced_bytes"]
    storage["bytes_retained"] = summary["produced_bytes"]
    storage_body = {
        key: value for key, value in storage.items() if key != "receipt_sha256"
    }
    storage["receipt_sha256"] = _canonical_hex(storage_body)
    _write(storage_path, storage)

    def validate_observation(path: object, **_kwargs: object) -> dict[str, object]:
        assert Path(str(path)).resolve() == observation_path.resolve()
        return {
            **deepcopy(observation_body),
            "observation_path": str(observation_path.resolve()),
            "observation_file_sha256": hashlib.sha256(
                observation_path.read_bytes()
            ).hexdigest(),
            "observation_file_bytes": observation_path.stat().st_size,
        }

    monkeypatch.setattr(
        adapter_module,
        "validate_workhorse_selected_world_observation_v1",
        validate_observation,
    )
    monkeypatch.setattr(
        adapter_module,
        "validate_workhorse_indexed_continuation_result_v2",
        lambda _path: deepcopy(continuation_result),
    )
    return root


def test_projects_only_observation_authority_from_authenticated_retained_output(
    tmp_path: Path,
) -> None:
    root = _retained_science_root(tmp_path)
    manifest = _expedition_manifest()

    projection = project_authenticated_science_request_observations_v1(
        root, manifest=manifest
    )

    assert len(projection.frames) == len(projection.state_refs) == 1
    frame = projection.frames[0]
    state = projection.state_refs[0]
    assert frame.next_absolute_tick == state.next_absolute_tick == 40
    assert state.state_class is StateClassV1.OBSERVATION_ONLY
    assert state.supports_exact_continuation is False
    assert state.exact_state_sha256 is None
    assert state.checkpoint_occurrence_sha256 is None
    assert state.rng_state_sha256 is None
    assert state.event_prefix_sha256 is None
    assert state.parent_occurrence_sha256 is None
    assert state.replay_recipe_sha256 is None
    assert state.source_artifact_ref.artifact_kind == "science_campaign_atlas_result"
    assert state.source_artifact_ref.locator.endswith("campaign_atlas_result.json")
    assert set(frame.observations.to_value()) == {
        "world_id",
        "world_result_sha256",
        "terminal_absolute_tick",
        "top_level_scalar_observations",
        "terminal_scalar_observations",
        "integrity",
        "exact_detail_source",
    }
    facts = frame.policy_facing_facts.to_value()
    assert facts["birth_count"] == 4
    assert facts["maximum_lineage_depth"] == 3
    assert facts["active_count"] == 4
    assert facts["resident_capacity"] == 512
    assert facts["capacity"] == 512
    assert facts["terminal_absolute_tick"] == 40
    assert facts["integrity_ok"] is True
    assert frame.descriptor_values.to_value()["status"] == (
        "NOT_INTERPRETED_BY_GENERIC_ADAPTER"
    )
    assert frame.calibration.to_value()["adapter_applied_calibration"] is False
    assert frame.control_readback.to_value()["branch_authority"] is False


def test_historical_row_without_capacity_remains_readable_and_fail_closed(
    tmp_path: Path,
) -> None:
    root = _retained_science_root(tmp_path, include_resident_capacity=False)

    projection = project_authenticated_science_request_observations_v1(
        root, manifest=_expedition_manifest()
    )

    facts = projection.frames[0].policy_facing_facts.to_value()
    assert "resident_capacity" not in facts
    assert "capacity" not in facts


def test_runtime_root_and_exact_receipt_path_project_byte_identically(
    tmp_path: Path,
) -> None:
    root = _retained_science_root(tmp_path)
    manifest = _expedition_manifest()

    from_root = project_authenticated_science_request_observations_v1(
        root, manifest=manifest
    )
    from_receipt = project_authenticated_science_request_observations_v1(
        root / "science_campaign_run_receipt.json", manifest=manifest
    )

    assert from_root == from_receipt
    assert from_root.projection_sha256() == from_receipt.projection_sha256()


def test_rejects_run_receipt_digest_tamper(tmp_path: Path) -> None:
    root = _retained_science_root(tmp_path)
    receipt_path = root / "science_campaign_run_receipt.json"
    receipt = _read(receipt_path)
    receipt["ok"] = False
    _write(receipt_path, receipt)

    with pytest.raises(AuthenticatedObservationError, match="canonical digest"):
        project_authenticated_science_request_observations_v1(
            root, manifest=_expedition_manifest()
        )


def test_rejects_exact_aggregate_file_hash_tamper(tmp_path: Path) -> None:
    root = _retained_science_root(tmp_path)
    result_path = root / "campaign_atlas_result.json"
    result = _read(result_path)
    result["physical_composition_used"] = True
    _write(result_path, result)

    with pytest.raises(AuthenticatedObservationError, match="file SHA-256"):
        project_authenticated_science_request_observations_v1(
            root, manifest=_expedition_manifest()
        )


def test_rejects_internally_tampered_result_even_when_outer_files_are_resealed(
    tmp_path: Path,
) -> None:
    root = _retained_science_root(tmp_path)
    result_path = root / "campaign_atlas_result.json"
    result = _read(result_path)
    worlds = result["world_results"]
    assert isinstance(worlds, list) and isinstance(worlds[0], dict)
    worlds[0]["birth_count"] = 999
    _write(result_path, result)
    _reseal_outer_files(root)

    with pytest.raises(AuthenticatedObservationError, match="canonical digest differs"):
        project_authenticated_science_request_observations_v1(
            root, manifest=_expedition_manifest()
        )


def test_rejects_request_identity_mismatch_with_valid_internal_digests(
    tmp_path: Path,
) -> None:
    root = _retained_science_root(tmp_path)
    result_path = root / "campaign_atlas_result.json"
    result = _read(result_path)
    result["request_sha256"] = _digest("foreign-request")
    result_body = {
        key: value for key, value in result.items() if key != "result_sha256"
    }
    result["result_sha256"] = canonical_sha256(result_body).to_dict()
    _write(result_path, result)
    verification_path = root / "campaign_atlas_verification_receipt.json"
    verification = _read(verification_path)
    verification["aggregate_result_sha256"] = result["result_sha256"]
    verification_body = {
        key: value for key, value in verification.items() if key != "receipt_sha256"
    }
    verification["receipt_sha256"] = canonical_sha256(verification_body).to_dict()
    _write(verification_path, verification)
    _reseal_outer_files(root)

    with pytest.raises(AuthenticatedObservationError, match="internal identity"):
        project_authenticated_science_request_observations_v1(
            root, manifest=_expedition_manifest()
        )


def test_rejects_nonterminal_lifecycle_even_when_rehashed(tmp_path: Path) -> None:
    root = _retained_science_root(tmp_path)
    manifest_path = root / "artifact_disposition_manifest.json"
    manifest = _read(manifest_path)
    manifest["terminal_status"] = "failed"
    manifest_body = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    manifest["manifest_sha256"] = _canonical_hex(manifest_body)
    _write(manifest_path, manifest)
    storage_path = root / "storage_disposition_receipt.json"
    storage = _read(storage_path)
    storage["manifest_sha256"] = manifest["manifest_sha256"]
    storage_body = {
        key: value for key, value in storage.items() if key != "receipt_sha256"
    }
    storage["receipt_sha256"] = _canonical_hex(storage_body)
    _write(storage_path, storage)

    with pytest.raises(AuthenticatedObservationError, match="terminal lifecycle"):
        project_authenticated_science_request_observations_v1(
            root, manifest=_expedition_manifest()
        )


def test_refuses_symlinked_runtime_source(tmp_path: Path) -> None:
    root = _retained_science_root(tmp_path)
    alias = tmp_path / "run-alias"
    alias.symlink_to(root, target_is_directory=True)

    with pytest.raises(AuthenticatedObservationError, match="symbolic link"):
        project_authenticated_science_request_observations_v1(
            alias, manifest=_expedition_manifest()
        )


def test_refuses_symlinked_retained_aggregate_file(tmp_path: Path) -> None:
    root = _retained_science_root(tmp_path)
    result_path = root / "campaign_atlas_result.json"
    preserved = tmp_path / "preserved-result.json"
    preserved.write_bytes(result_path.read_bytes())
    result_path.unlink()
    result_path.symlink_to(preserved)

    with pytest.raises(AuthenticatedObservationError, match="symbolic link"):
        project_authenticated_science_request_observations_v1(
            root, manifest=_expedition_manifest()
        )


def test_projects_selected_world_host_facts_without_atlas_or_causal_synthesis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _retained_selected_world_host_root(tmp_path, monkeypatch)
    manifest = _expedition_manifest()

    projection = project_authenticated_science_request_observations_v1(
        root, manifest=manifest
    )
    resumed = project_authenticated_science_request_observations_v1(
        root / "science_campaign_run_receipt.json", manifest=manifest
    )

    assert projection == resumed
    assert projection.projection_sha256() == resumed.projection_sha256()
    assert projection.aggregate_result_file_sha256 is None
    assert projection.verification_receipt_file_sha256 is None
    assert len(projection.host_observation_file_sha256s) == 1
    assert len(projection.frames) == 1
    frame = projection.frames[0]
    state = frame.source_state_ref
    assert state.state_class is StateClassV1.OBSERVATION_ONLY
    assert state.supports_exact_continuation is False
    assert state.exact_state_sha256 is None
    assert state.checkpoint_occurrence_sha256 is None
    observations = frame.observations.to_value()
    assert observations["population"] == {
        "active": 2,
        "alive": 2,
        "occupied_sites": 2,
        "capacity": 2048,
    }
    assert observations["cumulative_counters"]["birth_count"] == 3
    assert observations["active_lineage_summary"]["active_founder_count"] == 1
    assert observations["active_genotype_summary"]["unique_genotype_count"] == 2
    assert observations["control_readback"]["runtime_observation_mode"] == "OFF"
    assert observations["state_identity"]["canonical_world_state_sha256"] == (
        "c" * 64
    )
    unavailable = observations["unavailable_facts"]
    assert unavailable["event_kind_rows_available"] is False
    assert unavailable["birth_death_causal_rows_available"] is False
    assert unavailable["individual_lifetime_histories_available"] is False
    facts = frame.policy_facing_facts.to_value()
    assert facts["active_count"] == 2
    assert facts["capacity"] == 2048
    assert facts["birth_count"] == 3
    assert facts["death_count"] == 1
    assert frame.control_readback.to_value()["controls"] == {
        "environment.resource.multiplier": 1.0,
        "mutation.rate": 0.125,
    }
    assert "maximum_lineage_depth" not in facts
    assert "atlas" not in observations
    assert frame.descriptor_values.to_value()["missing_profile_facts"] == [
        "maximum_lineage_depth"
    ]


def test_selected_world_host_projection_rejects_tampered_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _retained_selected_world_host_root(tmp_path, monkeypatch)
    observation_path = next(root.rglob("workhorse-selected-world-host-observation.json"))
    observation_path.chmod(0o644)
    observation = _read(observation_path)
    observation["cell_id"] = "tampered-world"
    _write(observation_path, observation)

    with pytest.raises(AuthenticatedObservationError, match="observation"):
        project_authenticated_science_request_observations_v1(
            root, manifest=_expedition_manifest()
        )


def test_selected_world_host_projection_rejects_missing_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _retained_selected_world_host_root(tmp_path, monkeypatch)
    observation_path = next(root.rglob("workhorse-selected-world-host-observation.json"))
    observation_path.unlink()

    with pytest.raises(AuthenticatedObservationError, match="unavailable"):
        project_authenticated_science_request_observations_v1(
            root, manifest=_expedition_manifest()
        )
