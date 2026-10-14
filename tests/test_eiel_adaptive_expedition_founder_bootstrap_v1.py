from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from eiel_composition import (
    ABSOLUTE_TICK_OCCURRENCE_COMMIT_NAME,
    ABSOLUTE_TICK_OCCURRENCE_DESCRIPTOR_NAME,
    absolute_tick_checkpoint_bundle_file_participant_bindings_v1,
    materialize_absolute_tick_checkpoint_occurrence_v1,
)
from eiel_composition.adaptive_expedition import continuation_adapter as branch
from emergence_contracts.adaptive_science_expedition_v1 import ExpeditionManifestV1
from emergence_contracts import (
    ABSOLUTE_TICK_CHECKPOINT_FORMAT_ID,
    ABSOLUTE_TICK_CHECKPOINT_TIME_AUTHORITY,
    AttemptId,
    CheckpointId,
    ContentDigest,
    RunId,
    canonical_json_bytes,
)
from emergence_contracts.adaptive_science_expedition_v1 import ObservationFrameV1
from emergence_contracts.canonical import canonical_sha256
from engine.campaigns import cuda_full_evolution_v2_workhorse_provider as provider
from engine.campaigns.compiler import compile_campaign
from engine.perf.cuda_full_evolution_v2.workhorse.contract import (
    BACKEND_FUSED_CTA_TEMPORAL_V2,
    FullEvolutionCudaWorkhorseContractV2,
)
from engine.perf.cuda_full_evolution_v2.workhorse import indexed_checkpoint
from engine.perf.cuda_full_evolution_v2.workhorse.scientific_control import (
    workhorse_active_scientific_control_state_v1,
)
from eiel_composition.adaptive_expedition import founder_bootstrap as bootstrap
from eiel_composition.adaptive_expedition.operator_run import (
    _materialize_initial_exact_observation_sources_v1,
)
from eiel_composition.adaptive_expedition.policies.adaptive_illumination_v1 import (
    AdaptiveIlluminationPolicyV1,
)
from tests.test_eiel_adaptive_expedition_coordinator_v1 import _manifest
from tests.test_eiel_adaptive_expedition_general_closed_loop_v1 import (
    _workhorse_parent,
)
from tests.test_eiel_cuda_full_evolution_v2_workhorse_indexed_checkpoint import (
    _physical_checkpoint,
)
from tests.test_eiel_cuda_full_evolution_v2_workhorse_launchpad import (
    _operation_fixture,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _exact_founder_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_root: Path,
) -> tuple[dict[str, object], FullEvolutionCudaWorkhorseContractV2, dict[str, object]]:
    fixture = _operation_fixture(
        tmp_path / "compiled-source",
        monkeypatch,
        backend_id=BACKEND_FUSED_CTA_TEMPORAL_V2,
        acs_slots=2,
        B=2,
    )
    spec = deepcopy(fixture["spec"])
    contract = fixture["contract"]
    operation = spec["analysis"]["extension"]["contract"]
    operation["schema_version"] = provider.OPERATION_SCHEMA_V3
    operation["exact_founder_bootstrap"] = (
        provider.workhorse_exact_founder_bootstrap_contract_v1()
    )
    spec["analysis"]["extension"]["contract_sha256"] = provider.canonical_sha256(
        operation
    )
    spec["output"]["runtime_root"] = str(run_root.resolve())
    spec["execution"]["checkpoint_ticks"] = [1, 6]
    spec["execution"]["technical_gate"]["checkpoint_ticks"] = [1, 6]
    spec["artifact_lifecycle"]["checkpoints"] = {
        "authority": "absolute_tick",
        "checkpoint_ticks": [1, 6],
        "roles": [
            {
                "identity": "checkpoint-tick-000000000001",
                "checkpoint_tick": 1,
                "role": "intermediate",
                "retention_class": "R4",
            },
            {
                "identity": "checkpoint-tick-000000000006",
                "checkpoint_tick": 6,
                "role": "terminal",
                "retention_class": "R4",
            },
        ],
    }
    compiled = compile_campaign(
        spec,
        registry=provider.cuda_full_evolution_workhorse_v2_profile_registry_v1(
            shape_id=contract.shape_id,
            operation="EXECUTION_ROW",
        ),
        planning_resource_accounting=(
            provider.cuda_full_evolution_workhorse_v2_signed_planning_resource_accounting_v2
        ),
    ).payload()
    plan = provider.materialize_workhorse_exact_founder_plan_v1(compiled)
    assert provider.materialize_workhorse_exact_founder_plan_v1(plan) == plan
    return plan, contract, fixture


def _workhorse_absolute_bundle(
    root: Path,
    *,
    contract: FullEvolutionCudaWorkhorseContractV2,
    tick: int,
    state_bytes: bytes,
    last_sink_receipt_sha256: str,
    artifact_path: str,
    artifact_sha256: str,
    manifest_sha256: str,
) -> Path:
    root.mkdir(parents=True)
    state_sha256 = hashlib.sha256(state_bytes).hexdigest()
    state_bytes_per_world = len(state_bytes) // contract.B
    runtime_body = {
        "schema_version": ABSOLUTE_TICK_CHECKPOINT_FORMAT_ID,
        "time_authority": ABSOLUTE_TICK_CHECKPOINT_TIME_AUTHORITY,
        "contract": contract.to_dict(),
        "canonical_state_shape": [contract.B, state_bytes_per_world],
        "canonical_state_nbytes": len(state_bytes),
        "canonical_state_sha256": state_sha256,
        "canonical_state_layout_sha256": canonical_sha256(
            {
                "dtype": "uint8",
                "shape": [contract.B, state_bytes_per_world],
                "nbytes": len(state_bytes),
                "content_sha256": state_sha256,
            }
        ).hex,
        "initial_next_absolute_tick": 0,
        "advanced_ticks": tick,
        "acknowledged_event_next_tick": tick,
        "last_sink_receipt_sha256": last_sink_receipt_sha256,
        "world_uids": [210_000, 210_001],
        "rng_keys": [83, 83],
        "rng_semantic_epoch": 4,
        "artifact_sha256": artifact_sha256,
        "manifest_sha256": manifest_sha256,
        "device_identity": {"uuid": "GPU-WORKHORSE-HOST-FIXTURE"},
        "population_fixture_id": None,
    }
    runtime_checkpoint = {
        **runtime_body,
        "metadata_seal_sha256": canonical_sha256(runtime_body).hex,
    }
    checkpoint_id = f"workhorse-row-checkpoint-tick-{tick:012d}"
    metadata = {
        "schema_version": "eiel_absolute_tick_checkpoint_metadata_v1",
        "checkpoint_format_id": ABSOLUTE_TICK_CHECKPOINT_FORMAT_ID,
        "time_authority": ABSOLUTE_TICK_CHECKPOINT_TIME_AUTHORITY,
        "checkpoint_id": checkpoint_id,
        "absolute_next_tick": tick,
        "artifact_path": artifact_path,
        "runtime_checkpoint": runtime_checkpoint,
    }
    metadata_bytes = canonical_json_bytes(metadata)
    completion = {
        "schema_version": "eiel_absolute_tick_checkpoint_bundle_complete_v1",
        "status": "complete",
        "checkpoint_format_id": ABSOLUTE_TICK_CHECKPOINT_FORMAT_ID,
        "time_authority": ABSOLUTE_TICK_CHECKPOINT_TIME_AUTHORITY,
        "checkpoint_id": checkpoint_id,
        "absolute_next_tick": tick,
        "canonical_state_file": "canonical_state.bin",
        "canonical_state_nbytes": len(state_bytes),
        "canonical_state_sha256": state_sha256,
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
    }
    (root / "canonical_state.bin").write_bytes(state_bytes)
    (root / "metadata.json").write_bytes(metadata_bytes)
    (root / "bundle_complete.json").write_bytes(canonical_json_bytes(completion))
    return root


def _source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    tamper_publisher_controls: bool = False,
):
    manifest = _manifest(AdaptiveIlluminationPolicyV1())
    manifest_path = tmp_path / "expedition-manifest.json"
    _write(manifest_path, manifest.to_dict())
    contract = FullEvolutionCudaWorkhorseContractV2(
        shape_id="w64h64p1024c2048",
        backend_id=BACKEND_FUSED_CTA_TEMPORAL_V2,
        B=2,
        temporal_k=2,
        total_ticks=25,
        start_next_tick=0,
    )
    plan_body = {
        "analysis": {
            "extension": {
                "contract": {
                    "schema_version": "eiel_cuda_full_evolution_v2_workhorse_operation_v2",
                    "operation": "EXECUTION_ROW",
                    "contract": contract.to_dict(),
                }
            }
        },
        "execution": {
            "acs_checkpoint_slots": 2,
            "checkpoint_ticks": [2, 25],
        },
    }
    plan = {**plan_body, "plan_sha256": canonical_sha256(plan_body).hex}
    run_root = tmp_path / "run"
    batch_root = run_root / "batches" / "batch-0000"
    plan_path = run_root / "execution_plan.json"
    batch_path = batch_root / "manifest.json"
    _write(plan_path, plan)
    _write(
        batch_path,
        {
            "schema_version": "eiel_campaign_batch_manifest_v1",
            "status": "completed",
            "provider_id": "eiel-full-evolution-cuda-workhorse-v2-provider-v1",
            "batch_id": "batch-0000",
            "plan_sha256": plan["plan_sha256"],
        },
    )
    cache: dict[str, object] = {}

    def publisher(
        _batch_manifest_path: str | Path,
        destination: str | Path,
        *,
        world_uid: int,
        expedition_id: str,
        world_trajectory_id: str,
    ):
        assert Path(_batch_manifest_path).resolve() == batch_path.resolve()
        assert world_uid == 9102
        if "result" not in cache:
            state, authority, resolved = _workhorse_parent(
                tmp_path / "provider-material",
                monkeypatch,
                expedition_id=expedition_id,
                world_trajectory_id=world_trajectory_id,
                destination=Path(destination),
            )
            if tamper_publisher_controls:
                resolved = replace(
                    resolved,
                    active_scientific_control_state=(
                        workhorse_active_scientific_control_state_v1(
                            captured_next_absolute_tick=25,
                            resource_multiplier=1.0,
                            mutation_rate=0.25,
                        )
                    ),
                )
            cache["result"] = (state, authority, resolved)
        return cache["result"]

    monkeypatch.setattr(
        bootstrap,
        "materialize_workhorse_founder_exact_state_ref_v1",
        publisher,
    )
    return manifest, manifest_path, batch_path, plan_path, tmp_path / "frame", cache


def test_real_compiled_founder_plan_publishes_exact_observation_without_publisher_stub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "canonical-run"
    plan, contract, fixture = _exact_founder_plan(
        tmp_path,
        monkeypatch,
        run_root=run_root,
    )
    batch_id = str(plan["physical_batches"][0]["batch_id"])
    batch_root = run_root / "batches" / batch_id
    batch_root.mkdir(parents=True)
    plan_path = run_root / "execution_plan.json"
    _write(plan_path, plan)

    world_uids = [210_000, 210_001]
    rng_keys = [83, 83]
    event_receipts: list[dict[str, object]] = []
    for start_tick in range(6):
        end_tick = start_tick + 1
        rng_identity = {
            "schema_version": "eiel_workhorse_rng_identity_v2",
            "rng_epoch": 4,
            "world_uids": world_uids,
            "rng_keys": rng_keys,
        }
        dispatch_identity = canonical_sha256(
            {
                "schema_version": "eiel_workhorse_dispatch_fixture_v1",
                "start_next_absolute_tick": start_tick,
                "end_next_absolute_tick": end_tick,
                "world_uids": world_uids,
            }
        ).hex
        receipt_body = canonical_sha256(
            {
                "schema_version": "eiel_workhorse_sink_fixture_v1",
                "dispatch_identity_sha256": dispatch_identity,
                "rng_identity": rng_identity,
            }
        ).hex
        receipt_path = batch_root / "events" / f"receipt-{end_tick:012d}.json"
        _write(
            receipt_path,
            {
                "schema_version": "eiel_workhorse_sink_fixture_v1",
                "receipt_body_sha256": receipt_body,
                "dispatch_identity_sha256": dispatch_identity,
                "rng_identity": rng_identity,
            },
        )
        event_receipts.append(
            {
                "receipt_path": str(receipt_path.resolve()),
                "receipt_file_sha256": hashlib.sha256(
                    receipt_path.read_bytes()
                ).hexdigest(),
                "world_uids": world_uids,
                "start_next_absolute_tick": start_tick,
                "end_next_absolute_tick": end_tick,
                "receipt_body_sha256": receipt_body,
                "dispatch_identity_sha256": dispatch_identity,
                "rng_identity": rng_identity,
                "rng_identity_sha256": canonical_sha256(rng_identity).hex,
            }
        )

    genesis_metadata, _genesis_manifest, genesis_state_path = _physical_checkpoint(
        batch_root / "physical-checkpoints" / "checkpoint-tick-000000000001",
        world_uids=world_uids,
        rng_keys=rng_keys,
        temporal_k=1,
        tick=1,
    )
    terminal_metadata, checkpoint_manifest, terminal_state_path = (
        _physical_checkpoint(
            batch_root / "physical-checkpoints" / "checkpoint-tick-000000000006",
            world_uids=world_uids,
            rng_keys=rng_keys,
            temporal_k=1,
            tick=6,
        )
    )
    artifact_binding = fixture["artifact_binding"]
    terminal_metadata["artifact"] = {
        "identity_sha256": artifact_binding["identity"]["identity_sha256"],
        "sealed_artifact_identity_sha256": artifact_binding[
            "sealed_artifact_identity_sha256"
        ],
    }
    terminal_metadata["attempt_lineage_sha256"] = canonical_sha256(
        {"run_id": "run-founder-exact-v1", "attempt_id": "attempt-founder-exact-v1"}
    ).hex
    terminal_metadata["dispatch_identity_sha256"] = event_receipts[-1][
        "dispatch_identity_sha256"
    ]
    terminal_metadata["event_sink_ack"] = {
        "receipt_path": event_receipts[-1]["receipt_path"],
        "receipt_file_sha256": event_receipts[-1]["receipt_file_sha256"],
    }
    terminal_body = {
        key: value
        for key, value in terminal_metadata.items()
        if key != "bundle_body_sha256"
    }
    terminal_metadata["bundle_body_sha256"] = provider.canonical_sha256(
        terminal_body
    )

    def inspect_physical_checkpoint(
        path: str | Path,
        **_kwargs: object,
    ) -> tuple[dict[str, object], Path]:
        if Path(path).resolve() != checkpoint_manifest.resolve():
            raise RuntimeError("fixture exposes only its authenticated terminal")
        return deepcopy(terminal_metadata), terminal_state_path

    monkeypatch.setattr(
        indexed_checkpoint,
        "inspect_workhorse_checkpoint_bundle_v2",
        inspect_physical_checkpoint,
    )
    indexed = indexed_checkpoint.write_workhorse_indexed_checkpoint_set_v1(
        checkpoint_manifest,
        batch_root / "indexed-checkpoints" / "checkpoint-tick-000000000006",
    )

    artifact_path = str(fixture["artifact_binding"]["path"])
    artifact_sha256 = str(fixture["artifact_binding"]["sha256"])
    artifact_manifest_sha256 = str(fixture["artifact_binding"]["manifest_sha256"])
    absolute_parent = batch_root / "workhorse-absolute-checkpoints"
    genesis_bundle = _workhorse_absolute_bundle(
        absolute_parent / "checkpoint-tick-000000000001",
        contract=contract,
        tick=1,
        state_bytes=genesis_state_path.read_bytes(),
        last_sink_receipt_sha256=str(event_receipts[0]["receipt_body_sha256"]),
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
        manifest_sha256=artifact_manifest_sha256,
    )
    terminal_bundle = _workhorse_absolute_bundle(
        absolute_parent / "checkpoint-tick-000000000006",
        contract=contract,
        tick=6,
        state_bytes=terminal_state_path.read_bytes(),
        last_sink_receipt_sha256=str(event_receipts[-1]["receipt_body_sha256"]),
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
        manifest_sha256=artifact_manifest_sha256,
    )
    run_id = RunId("run-founder-exact-v1")
    attempt_id = AttemptId("attempt-founder-exact-v1")
    semantic_spec = ContentDigest(
        "sha256", str(plan["cells"][0]["science_world_sha256"]["hex"])
    )
    implementation = canonical_sha256(
        {
            "provider_id": provider.PROVIDER_ID,
            "materialization": plan["science_materialization"],
        }
    )
    owner_digest = canonical_sha256(
        {"owner": "cuda_full_evolution_v2_workhorse_provider"}
    )
    occurrence_parent = batch_root / "workhorse-checkpoint-occurrences"
    genesis, _genesis_commit = materialize_absolute_tick_checkpoint_occurrence_v1(
        genesis_bundle,
        occurrence_parent / "checkpoint-tick-000000000001",
        checkpoint_id=CheckpointId("workhorse-row-checkpoint-tick-000000000001"),
        run_id=run_id,
        attempt_id=attempt_id,
        transaction_id="founder-exact-checkpoint-000000000001",
        absolute_next_tick=1,
        semantic_spec_digest=semantic_spec,
        implementation_digest=implementation,
        participant_bindings=(
            absolute_tick_checkpoint_bundle_file_participant_bindings_v1(
                genesis_bundle,
                owner_id="cuda_full_evolution_v2_workhorse_provider",
                owner_digest=owner_digest,
            )
        ),
        parent_checkpoint_digest=None,
        committed_at_utc="2026-08-29T12:00:00Z",
    )
    terminal, _terminal_commit = materialize_absolute_tick_checkpoint_occurrence_v1(
        terminal_bundle,
        occurrence_parent / "checkpoint-tick-000000000006",
        checkpoint_id=CheckpointId("workhorse-row-checkpoint-tick-000000000006"),
        run_id=run_id,
        attempt_id=attempt_id,
        transaction_id="founder-exact-checkpoint-000000000006",
        absolute_next_tick=6,
        semantic_spec_digest=semantic_spec,
        implementation_digest=implementation,
        participant_bindings=(
            absolute_tick_checkpoint_bundle_file_participant_bindings_v1(
                terminal_bundle,
                owner_id="cuda_full_evolution_v2_workhorse_provider",
                owner_digest=owner_digest,
            )
        ),
        parent_checkpoint_digest=genesis.occurrence_digest(),
        committed_at_utc="2026-08-29T12:00:01Z",
    )
    terminal_occurrence_root = (
        occurrence_parent / "checkpoint-tick-000000000006"
    ).resolve()
    descriptor_path = (
        terminal_occurrence_root / ABSOLUTE_TICK_OCCURRENCE_DESCRIPTOR_NAME
    )
    commit_path = terminal_occurrence_root / ABSOLUTE_TICK_OCCURRENCE_COMMIT_NAME
    projection_core = {
        "schema_version": "eiel_workhorse_row_checkpoint_occurrence_projection_v2",
        "claim_class": "TECHNICAL_ONLY",
        "science_authority_granted": False,
        "qualification_authority_granted": False,
        "time_authority": "absolute_tick_v1",
        "checkpoint_at_tick": 6,
        "checkpoint_manifest_path": str(checkpoint_manifest.resolve()),
        "checkpoint_manifest_file_sha256": hashlib.sha256(
            checkpoint_manifest.read_bytes()
        ).hexdigest(),
        "checkpoint_bundle_body_sha256": terminal_metadata[
            "bundle_body_sha256"
        ],
        "checkpoint_identity_sha256": terminal_metadata["checkpoint_identity"][
            "checkpoint_identity_sha256"
        ],
        "canonical_state_sha256": terminal_metadata["canonical_state"]["sha256"],
        "canonical_parent_binding_sha256": canonical_sha256(
            {"run_id": str(run_id), "attempt_id": str(attempt_id)}
        ).hex,
        "operation_contract_sha256": plan["analysis"]["extension"][
            "contract_sha256"
        ],
        "absolute_tick_bundle_path": str(terminal_bundle.resolve()),
        "absolute_tick_bundle_digest": terminal.checkpoint_bundle_digest.to_dict(),
        "occurrence_root": str(terminal_occurrence_root),
        "occurrence_digest": terminal.occurrence_digest().to_dict(),
        "occurrence_descriptor_path": str(descriptor_path),
        "occurrence_descriptor_file_sha256": hashlib.sha256(
            descriptor_path.read_bytes()
        ).hexdigest(),
        "occurrence_commit_path": str(commit_path),
        "occurrence_commit_file_sha256": hashlib.sha256(
            commit_path.read_bytes()
        ).hexdigest(),
        "parent_checkpoint_digest": genesis.occurrence_digest().to_dict(),
        "canonical_visible": True,
        "restore_admitted_exact": False,
    }
    projection = {
        **projection_core,
        "projection_content_sha256": canonical_sha256(projection_core).hex,
    }
    execution_result_path = batch_root / "workhorse-execution-result.json"
    _write(execution_result_path, {"outcome": "COMPLETED"})
    monkeypatch.setattr(
        branch,
        "validate_workhorse_canonical_row_execution_result_v2",
        lambda _path: {
            "event_sink_receipts": deepcopy(event_receipts),
            "checkpoint_binding": {"checkpoint_ticks": [1, 6]},
        },
    )
    batch_path = batch_root / "manifest.json"
    _write(
        batch_path,
        {
            "schema_version": "eiel_campaign_batch_manifest_v1",
            "status": "completed",
            "provider_id": provider.PROVIDER_ID,
            "batch_id": batch_id,
            "plan_sha256": plan["plan_sha256"],
            "result": {
                "execution_result_path": str(execution_result_path.resolve()),
                "execution_result_file_sha256": hashlib.sha256(
                    execution_result_path.read_bytes()
                ).hexdigest(),
            },
            "checkpoint": {
                "acs_checkpoint_receipt": {
                    "checkpoint_at_tick": 6,
                    "checkpoint_occurrence_projection": projection,
                    "indexed_checkpoint": {
                        "manifest_path": indexed["manifest_path"],
                    },
                }
            },
        },
    )
    expedition = _manifest(AdaptiveIlluminationPolicyV1())
    expedition_path = tmp_path / "expedition-manifest.json"
    _write(expedition_path, expedition.to_dict())

    frame, artifact_ref, frame_path, ref_path = (
        bootstrap.materialize_workhorse_founder_exact_observation_frame_v1(
            manifest_path=expedition_path,
            batch_manifest_path=batch_path,
            destination_root=tmp_path / "exact-founder-frame",
            world_uid=210_000,
        )
    )
    assert frame.source_state_ref.logical_world_id == 210_000
    assert frame.source_state_ref.next_absolute_tick == 6
    assert frame.source_state_ref.semantic_world_sha256 == semantic_spec
    assert frame.source_state_ref.parent_occurrence_sha256 == (
        genesis.occurrence_digest()
    )
    assert frame.source_state_ref.checkpoint_occurrence_sha256 == (
        terminal.occurrence_digest()
    )
    assert frame.control_readback.to_value()["controls"] == {
        "environment.resource.multiplier": 1.0,
        "mutation.rate": 0.125,
    }
    assert frame.policy_facing_facts.to_value()["capacity"] == 2048
    assert artifact_ref.content_digest == frame.observation_frame_sha256()
    assert frame_path.read_bytes() == canonical_json_bytes(frame.to_dict())
    assert ref_path.is_file()


def test_exact_founder_builder_is_canonical_idempotent_and_operator_consumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, manifest_path, batch_path, _plan_path, root, _cache = _source(
        tmp_path, monkeypatch
    )
    first = bootstrap.materialize_workhorse_founder_exact_observation_frame_v1(
        manifest_path=manifest_path,
        batch_manifest_path=batch_path,
        destination_root=root,
        world_uid=9102,
    )
    second = bootstrap.materialize_workhorse_founder_exact_observation_frame_v1(
        manifest_path=manifest_path,
        batch_manifest_path=batch_path,
        destination_root=root,
        world_uid=9102,
    )
    frame, ref, frame_path, ref_path = first
    assert second[0].to_dict() == frame.to_dict()
    assert second[1].to_dict() == ref.to_dict()
    assert frame_path.read_bytes() == canonical_json_bytes(frame.to_dict())
    assert ref_path.read_bytes() == canonical_json_bytes(ref.to_dict())
    facts = frame.policy_facing_facts.to_value()
    assert facts["active_count"] == 0
    assert facts["capacity"] == 2048
    assert facts["birth_count"] == 0
    assert facts["death_count"] == 0
    assert facts["integrity_ok"] is True
    assert "maximum_lineage_depth" not in facts
    assert facts["unavailable_authenticated_facts"] == ["maximum_lineage_depth"]
    assert frame.descriptor_values.to_value()["status"] == (
        "NOT_INTERPRETED_BY_GENERIC_ADAPTER"
    )
    assert frame.source_state_ref.supports_exact_continuation is True
    assert ref.content_digest == frame.observation_frame_sha256()
    assert ref.size_bytes == len(canonical_json_bytes(frame.to_dict()))

    final = manifest.to_dict()
    final["founder_basis_refs"].append(ref.to_dict())
    bound = ExpeditionManifestV1.from_dict(final)
    sources = _materialize_initial_exact_observation_sources_v1(
        manifest=bound,
        control_root=tmp_path / "operator-control",
    )
    assert len(sources) == 1
    retained = ObservationFrameV1.from_dict(json.loads(Path(sources[0]).read_bytes()))
    assert retained.source_state_ref.exact_state_sha256 == (
        frame.source_state_ref.exact_state_sha256
    )
    assert retained.source_state_ref.rng_state_sha256 == (
        frame.source_state_ref.rng_state_sha256
    )
    assert retained.source_state_ref.event_prefix_sha256 == (
        frame.source_state_ref.event_prefix_sha256
    )
    assert Path(retained.source_state_ref.source_artifact_ref.locator).is_relative_to(
        tmp_path / "durable" / "founder-basis"
    )


@pytest.mark.parametrize("tamper", ("STATE", "ARTIFACT_REF", "MANIFEST_BINDING"))
def test_exact_founder_builder_fails_closed_on_retained_artifact_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    _manifest_value, manifest_path, batch_path, _plan_path, root, cache = _source(
        tmp_path, monkeypatch
    )
    _frame, _ref, _frame_path, ref_path = (
        bootstrap.materialize_workhorse_founder_exact_observation_frame_v1(
            manifest_path=manifest_path,
            batch_manifest_path=batch_path,
            destination_root=root,
            world_uid=9102,
        )
    )
    if tamper == "STATE":
        _state, _authority, resolved = cache["result"]
        state_path = Path(resolved.source_object_path)
        state_path.chmod(0o644)
        payload = bytearray(state_path.read_bytes())
        payload[-1] ^= 1
        state_path.write_bytes(payload)
    elif tamper == "ARTIFACT_REF":
        ref_path.chmod(0o644)
        value = json.loads(ref_path.read_text(encoding="ascii"))
        value["size_bytes"] += 1
        _write(ref_path, value)
    else:
        value = json.loads(manifest_path.read_text(encoding="ascii"))
        value["observation_profile_id"] = "CHANGED_AFTER_FRAME_PUBLICATION"
        manifest_path.chmod(0o644)
        _write(manifest_path, value)
    with pytest.raises(bootstrap.FounderBootstrapError):
        bootstrap.materialize_workhorse_founder_exact_observation_frame_v1(
            manifest_path=manifest_path,
            batch_manifest_path=batch_path,
            destination_root=root,
            world_uid=9102,
        )


def test_exact_founder_builder_rejects_plan_and_publisher_control_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _manifest_value, manifest_path, batch_path, plan_path, root, _cache = _source(
        tmp_path / "plan", monkeypatch
    )
    plan = json.loads(plan_path.read_text(encoding="ascii"))
    plan["execution"]["checkpoint_ticks"] = [3, 25]
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    plan["plan_sha256"] = canonical_sha256(body).hex
    plan_path.chmod(0o644)
    _write(plan_path, plan)
    batch = json.loads(batch_path.read_text(encoding="ascii"))
    batch["plan_sha256"] = plan["plan_sha256"]
    batch_path.chmod(0o644)
    _write(batch_path, batch)
    with pytest.raises(bootstrap.FounderBootstrapError, match="ACS2 bootstrap"):
        bootstrap.materialize_workhorse_founder_exact_observation_frame_v1(
            manifest_path=manifest_path,
            batch_manifest_path=batch_path,
            destination_root=root,
            world_uid=9102,
        )

    _manifest_value, manifest_path, batch_path, _plan_path, root, _cache = _source(
        tmp_path / "controls",
        monkeypatch,
        tamper_publisher_controls=True,
    )
    with pytest.raises(bootstrap.FounderBootstrapError, match="inconsistent authority"):
        bootstrap.materialize_workhorse_founder_exact_observation_frame_v1(
            manifest_path=manifest_path,
            batch_manifest_path=batch_path,
            destination_root=root,
            world_uid=9102,
        )
