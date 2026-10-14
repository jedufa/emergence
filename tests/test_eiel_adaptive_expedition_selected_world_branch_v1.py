from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from eiel_composition.adaptive_expedition import continuation_adapter as branch
from eiel_composition import science_cuda_v2_carrier_v1 as carrier
from eiel_composition.science_request_v1_compiler import compile_science_request_v1
from emergence_contracts.adaptive_science_expedition_v1 import (
    StateClassV1,
    StateRefV1,
)
from emergence_contracts.artifacts import RunArtifactRefV1
from emergence_contracts.canonical import canonical_sha256
from emergence_contracts.checkpoint_occurrence import (
    ABSOLUTE_TICK_CHECKPOINT_FORMAT_ID,
    ABSOLUTE_TICK_CHECKPOINT_TIME_AUTHORITY,
    AbsoluteTickCheckpointOccurrenceV1,
    AbsoluteTickCheckpointOccurrenceCommitV1,
    AbsoluteTickCheckpointParticipantOccurrenceV1,
)
from emergence_contracts.ids import (
    ArtifactId,
    AttemptId,
    CheckpointId,
    ContentDigest,
    ParticipantId,
    RunId,
    SchemaVersion,
)
from engine.campaigns import acs_world_sharded_checkpoint_v1 as sharded
from engine.campaigns import cuda_full_evolution_v2_workhorse_provider as workhorse_provider
from engine.campaigns.artifact_lifecycle import (
    CHECKPOINT_TRANSIT,
    lifecycle_profile_contract_v1,
)
from engine.campaigns.providers import (
    ACS_NATIVE_RUNTIME_EXTENSION_ARTIFACT_ID_V1,
    ACS_NATIVE_RUNTIME_MANIFEST_ARTIFACT_ID_V1,
)
from engine.native_runtime.checkpoint_digests import (
    checkpoint_source_projection_authority_manifest,
)
from engine.perf.cuda_architecture_b_boundary import (
    EVENT_COLUMN_DTYPES,
    EVENT_COLUMN_NAMES,
)
from engine.perf.cuda_architecture_b_event_sink_v1 import (
    append_event_sink_columns_v1,
    event_sink_prefix_identity_v1,
)
from engine.perf.cuda_full_evolution_v2.workhorse import indexed_checkpoint
from engine.perf.cuda_full_evolution_v2.workhorse.contract import (
    BACKEND_PHASE_GRAPH_V2,
    FullEvolutionCudaWorkhorseContractV2,
    workhorse_shape_v2,
)
from engine.perf.cuda_full_evolution_v2.workhorse.scientific_control import (
    workhorse_active_scientific_control_state_v1,
)
from tests import test_acs_world_sharded_checkpoint_v1 as world_fixture
from tests.test_eiel_cuda_full_evolution_v2_workhorse_indexed_checkpoint import (
    _physical_checkpoint,
)
from tests.test_eiel_science_request_v1_compiler import _selected_world_request
from eiel_composition.science_world_bundle_v1 import (
    admit_science_world_bundle_v1,
)


def _empty_event_columns() -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.empty((0,), dtype=EVENT_COLUMN_DTYPES[name])
        for name in EVENT_COLUMN_NAMES
    )


def _occurrence(*, tick: int, parent_digest: ContentDigest) -> AbsoluteTickCheckpointOccurrenceV1:
    run_id = RunId("parent-run-v1")
    attempt_id = AttemptId("parent-attempt-v1")
    transaction_id = "parent-checkpoint-transaction-v1"
    return AbsoluteTickCheckpointOccurrenceV1(
        checkpoint_format_id=ABSOLUTE_TICK_CHECKPOINT_FORMAT_ID,
        time_authority=ABSOLUTE_TICK_CHECKPOINT_TIME_AUTHORITY,
        checkpoint_id=CheckpointId("parent-checkpoint-v1"),
        run_id=run_id,
        attempt_id=attempt_id,
        transaction_id=transaction_id,
        absolute_next_tick=tick,
        semantic_spec_digest=canonical_sha256({"semantic": "parent"}),
        implementation_digest=canonical_sha256({"implementation": "fixture"}),
        participant_occurrences=(
            AbsoluteTickCheckpointParticipantOccurrenceV1(
                participant_id=ParticipantId("absolute-tick-checkpoint-file:state"),
                run_id=run_id,
                attempt_id=attempt_id,
                transaction_id=transaction_id,
                absolute_next_tick=tick,
                owner_id="fixture-owner",
                owner_digest=canonical_sha256({"owner": "fixture"}),
                state_digest=canonical_sha256({"state": "fixture"}),
            ),
        ),
        checkpoint_bundle_digest=canonical_sha256({"bundle": "fixture"}),
        completion_marker_digest=canonical_sha256({"completion": "fixture"}),
        parent_checkpoint_digest=parent_digest,
    )


def _exact_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    swap_selected_event_prefix: bool = False,
) -> tuple[
    StateRefV1,
    dict[str, object],
    dict[str, object],
    Path,
    Path,
    str,
]:
    run_root = tmp_path / "parent-run"
    run_root.mkdir()
    cells = 2
    event_path = run_root / "events" / "events.bin"
    append_event_sink_columns_v1(
        event_path,
        cell_index=0,
        columns=_empty_event_columns(),
        sequence_base=0,
        expected_cells=cells,
    )
    event_refs = [
        event_sink_prefix_identity_v1(
            event_path,
            cell_index=index,
            expected_rows=0,
        )
        for index in range(cells)
    ]
    original_recipe = world_fixture._recipe

    def exact_recipe(
        cell_ids: tuple[str, ...],
        descriptors: list[dict[str, object]],
    ) -> dict[str, object]:
        value = original_recipe(cell_ids, descriptors)
        value["checkpoint_projection_inputs"]["tw_rngv2_logical_world_uids"] = [
            7001 + index for index in range(cells)
        ]
        value["direct_result_scalars"].update(
            {
                "event_history_frontier_by_cell": [0] * cells,
                "event_history_prefix_references_by_cell": (
                    [event_refs[0], event_refs[0]]
                    if swap_selected_event_prefix
                    else event_refs
                ),
            }
        )
        value["source_configuration_sha256"] = hashlib.sha256(
            world_fixture._canonical({"configuration": "fixture-v1"})
        ).hexdigest()
        value.pop("recipe_sha256")
        value["recipe_sha256"] = hashlib.sha256(
            world_fixture._canonical(value)
        ).hexdigest()
        return value

    monkeypatch.setattr(world_fixture, "_recipe", exact_recipe)
    source, receipt, expected = world_fixture._packed_fixture(run_root, cells=cells)
    result = world_fixture._write_set(
        source,
        receipt,
        run_root / "world-sets",
        cells=cells,
    )
    selected_world_id = str(expected["cell_ids"][1])
    loaded = sharded.load_selected_world_from_set_v1(
        result["manifest_path"],
        world_id=selected_world_id,
    )
    manifest = loaded["set_manifest"]
    row = next(
        item for item in manifest["objects"] if item["cell_id"] == selected_world_id
    )
    rng_sha256 = checkpoint_source_projection_authority_manifest(
        loaded["source_projection"]
    )["source_rng_sha256"]
    parent_digest = canonical_sha256({"occurrence": "grandparent"})
    occurrence = _occurrence(tick=960, parent_digest=parent_digest)
    semantic_digest = canonical_sha256({"science-world": selected_world_id})
    recipe = loaded["scalar_metadata"]["settlement_recipe"]
    control_prefix = {
        "schema_version": branch.CONTROL_HISTORY_PREFIX_SCHEMA,
        "selected_world_id": selected_world_id,
        "logical_world_id": 7002,
        "end_next_absolute_tick": 960,
        "semantic_world_sha256": semantic_digest.to_dict(),
        "world_mechanics_sha256": ContentDigest(
            "sha256", row["world_mechanics_sha256"]
        ).to_dict(),
        "source_configuration_sha256": ContentDigest(
            "sha256", recipe["source_configuration_sha256"]
        ).to_dict(),
        "applied_control_prefix": {
            "control_readback_receipt_sha256s": ["c" * 64],
            "prefix_end_next_absolute_tick": 960,
        },
    }
    object_path = Path(loaded["path"])
    state_ref = StateRefV1(
        expedition_id="expedition-fixture-v1",
        world_trajectory_id="trajectory-parent-v1",
        logical_world_id=7002,
        next_absolute_tick=960,
        state_class=StateClassV1.BRANCHABLE_EXACT,
        semantic_world_sha256=semantic_digest,
        control_history_sha256=canonical_sha256(control_prefix),
        source_object_sha256=ContentDigest("sha256", row["artifact_sha256"]),
        source_artifact_ref=RunArtifactRefV1(
            artifact_id=ArtifactId("world-object-fixture-v1"),
            artifact_kind=branch.WORLD_ARTIFACT_KIND,
            artifact_schema_version=SchemaVersion(sharded.WORLD_SCHEMA_VERSION),
            locator=str(object_path),
            content_digest=ContentDigest("sha256", row["artifact_sha256"]),
            size_bytes=object_path.stat().st_size,
        ),
        exact_state_sha256=ContentDigest("sha256", row["world_state_sha256"]),
        checkpoint_occurrence_sha256=occurrence.occurrence_digest(),
        rng_state_sha256=ContentDigest("sha256", rng_sha256),
        event_prefix_sha256=ContentDigest(
            "sha256",
            recipe["direct_result_scalars"][
                "event_history_prefix_references_by_cell"
            ][0]["prefix_sha256"],
        ),
        parent_occurrence_sha256=parent_digest,
        replay_recipe_sha256=None,
    )

    # The existing world-sharded module separately tests the real compatibility
    # restorer with a production-shape settlement recipe.  This fixture keeps
    # the adapter unit bounded while returning the same authenticated source
    # projection used to derive its StateRef RNG.
    monkeypatch.setattr(
        branch,
        "restore_world_checkpoint_projection_v1",
        lambda *_args, **_kwargs: {"projection": loaded["source_projection"]},
    )

    return (
        state_ref,
        occurrence.to_dict(),
        control_prefix,
        run_root,
        event_path,
        str(result["manifest_path"]),
    )


def test_exact_selected_world_binds_object_occurrence_rng_event_and_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_ref, occurrence, controls, run_root, _events, manifest = _exact_fixture(
        tmp_path,
        monkeypatch,
    )
    resolved = branch._inspect_selected_world_material_v1(
        state_ref=state_ref,
        checkpoint_set_manifest_path=manifest,
        selected_world_id="cell-001",
        checkpoint_occurrence=occurrence,
        control_history_prefix=controls,
        parent_run_root=run_root,
    )
    with pytest.raises(
        branch.SelectedWorldBranchCarrierUnavailable,
        match="no committed producer occurrence",
    ):
        branch.resolve_selected_world_state_v1(
            state_ref=state_ref,
            checkpoint_set_manifest_path=manifest,
            selected_world_id="cell-001",
            checkpoint_occurrence=occurrence,
            control_history_prefix=controls,
            parent_run_root=run_root,
        )

    assert resolved.physical_cell_index == 1
    assert resolved.runtime_logical_world_uid == state_ref.logical_world_id
    assert resolved.next_absolute_tick == 960
    assert resolved.source_object_sha256 == state_ref.source_object_sha256
    assert resolved.exact_state_sha256 == state_ref.exact_state_sha256
    assert resolved.rng_state_sha256 == state_ref.rng_state_sha256
    assert resolved.event_prefix_sha256 == state_ref.event_prefix_sha256
    assert resolved.checkpoint_occurrence_sha256 == (
        state_ref.checkpoint_occurrence_sha256
    )
    assert resolved.parent_occurrence_sha256 == state_ref.parent_occurrence_sha256
    assert resolved.control_history_sha256 == state_ref.control_history_sha256

    suffix = {
        "schema_version": branch.CONTROL_SUFFIX_SCHEMA,
        "effective_next_absolute_tick": 960,
        "controls": [
            {
                "control_id": "pressure-fixture-v1",
                "onset_tick": 960,
                "payload": {"value": 0.75},
            }
        ],
    }
    assert branch.validate_control_suffix_v1(
        suffix,
        parent_next_absolute_tick=960,
        effective_next_absolute_tick=960,
    ) == canonical_sha256(suffix)
    with pytest.raises(branch.SelectedWorldBranchCarrierUnavailable):
        branch.prepare_selected_world_branch_v1(
            resolved,
            child_run_id=RunId("child-run-v1"),
            child_trajectory_id="trajectory-child-v1",
            effective_next_absolute_tick=960,
            control_suffix=suffix,
        )


def test_workhorse_selected_parent_materializes_public_refs_and_reopens_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, checkpoint_manifest, state_path = _physical_checkpoint(
        tmp_path / "physical-workhorse-parent",
        world_uids=[9101, 9102],
        rng_keys=[71, 72],
        temporal_k=2,
    )
    monkeypatch.setattr(
        indexed_checkpoint,
        "inspect_workhorse_checkpoint_bundle_v2",
        lambda _path, **_kwargs: (deepcopy(metadata), state_path),
    )
    indexed = indexed_checkpoint.write_workhorse_indexed_checkpoint_set_v1(
        checkpoint_manifest,
        tmp_path / "indexed-workhorse-parent",
    )
    parent_digest = canonical_sha256({"occurrence": "prior-workhorse-parent"})
    occurrence = _occurrence(tick=25, parent_digest=parent_digest)
    occurrence_root = tmp_path / "workhorse-occurrence"
    occurrence_root.mkdir()
    descriptor_path = occurrence_root / "absolute-tick-occurrence.json"
    commit_path = occurrence_root / "absolute-tick-occurrence-commit.json"
    descriptor_path.write_text(json.dumps(occurrence.to_dict(), sort_keys=True))
    commit = AbsoluteTickCheckpointOccurrenceCommitV1(
        occurrence_digest=occurrence.occurrence_digest(),
        checkpoint_bundle_digest=occurrence.checkpoint_bundle_digest,
        checkpoint_format_id=occurrence.checkpoint_format_id,
        committed_at_utc="2026-08-28T12:00:00Z",
    )
    commit_path.write_text(json.dumps(commit.to_dict(), sort_keys=True))
    projection_core = {
        "schema_version": "eiel_workhorse_row_checkpoint_occurrence_projection_v2",
        "claim_class": "TECHNICAL_ONLY",
        "science_authority_granted": False,
        "qualification_authority_granted": False,
        "time_authority": "absolute_tick_v1",
        "checkpoint_at_tick": 25,
        "checkpoint_manifest_path": str(checkpoint_manifest),
        "checkpoint_manifest_file_sha256": "1" * 64,
        "checkpoint_bundle_body_sha256": "2" * 64,
        "checkpoint_identity_sha256": indexed["source_physical_checkpoint"][
            "checkpoint_identity_sha256"
        ],
        "canonical_state_sha256": "3" * 64,
        "canonical_parent_binding_sha256": "4" * 64,
        "operation_contract_sha256": "5" * 64,
        "absolute_tick_bundle_path": str(tmp_path / "absolute-bundle"),
        "absolute_tick_bundle_digest": occurrence.checkpoint_bundle_digest.to_dict(),
        "occurrence_root": str(occurrence_root),
        "occurrence_digest": occurrence.occurrence_digest().to_dict(),
        "occurrence_descriptor_path": str(descriptor_path),
        "occurrence_descriptor_file_sha256": hashlib.sha256(
            descriptor_path.read_bytes()
        ).hexdigest(),
        "occurrence_commit_path": str(commit_path),
        "occurrence_commit_file_sha256": hashlib.sha256(
            commit_path.read_bytes()
        ).hexdigest(),
        "parent_checkpoint_digest": parent_digest.to_dict(),
        "canonical_visible": True,
        "restore_admitted_exact": False,
    }
    projection = {
        **projection_core,
        "projection_content_sha256": canonical_sha256(projection_core).hex,
    }
    active_controls = workhorse_active_scientific_control_state_v1(
        captured_next_absolute_tick=25,
        resource_multiplier=1.0,
        mutation_rate=0.125,
    )
    state_ref, authority_ref, resolved = (
        branch.materialize_workhorse_selected_world_state_ref_v1(
            indexed["manifest_path"],
            tmp_path / "retained-workhorse-parent",
            world_uid=9102,
            checkpoint_occurrence_projection=projection,
            expedition_id="expedition-workhorse-v1",
            world_trajectory_id="trajectory-workhorse-parent-v1",
            semantic_world_sha256="6" * 64,
            control_history_sha256="7" * 64,
            rng_state_sha256="8" * 64,
            event_prefix_sha256="9" * 64,
            active_scientific_control_state=active_controls,
        )
    )
    assert resolved.state_ref == state_ref
    assert resolved.runtime_logical_world_uid == 9102
    assert resolved.parent_run_id == occurrence.run_id
    assert resolved.active_scientific_control_state == active_controls
    assert branch.resolve_workhorse_selected_world_continuation_v1(
        state_ref=state_ref,
        authority_receipt_ref=authority_ref,
    ) == resolved
    # A retained Parent Bank package is lifecycle-independent from the raw
    # child occurrence.  Release/move the source occurrence and prove that
    # exact reopen still succeeds from the package-local descriptor/commit.
    retained_root = tmp_path / "retained-workhorse-parent"
    retained_authority = json.loads(
        (retained_root / branch.WORKHORSE_SELECTED_WORLD_AUTHORITY_FILE_V1).read_text()
    )
    assert Path(retained_authority["occurrence_root"]).is_dir()
    assert Path(retained_authority["occurrence_descriptor_path"]).is_file()
    assert Path(retained_authority["occurrence_commit_path"]).is_file()
    occurrence_root.rename(tmp_path / "released-workhorse-occurrence")
    assert not occurrence_root.exists()
    assert branch.resolve_workhorse_selected_world_continuation_v1(
        state_ref=state_ref,
        authority_receipt_ref=authority_ref,
    ) == resolved
    suffix = {
        "schema_version": branch.CONTROL_SUFFIX_SCHEMA,
        "effective_next_absolute_tick": 25,
        "controls": [
            {
                "control_id": "environment.resource.multiplier",
                "onset_tick": 25,
                "payload": {"value": 1.5},
            }
        ],
    }
    prepared = branch.prepare_selected_world_branch_v1(
        resolved,
        child_run_id=RunId("child-workhorse-run-v1"),
        child_trajectory_id="trajectory-workhorse-child-v1",
        effective_next_absolute_tick=25,
        control_suffix=suffix,
    )
    assert prepared.child_run_id != resolved.parent_run_id
    assert prepared.first_permitted_divergence_tick == 25

    wrong_authority = replace(
        authority_ref,
        content_digest=canonical_sha256({"authority": "wrong"}),
    )
    with pytest.raises(
        branch.SelectedWorldBranchError,
        match="authority reference differs",
    ):
        branch.resolve_workhorse_selected_world_continuation_v1(
            state_ref=state_ref,
            authority_receipt_ref=wrong_authority,
        )


def test_science_carrier_lowers_exact_selected_parent_as_fresh_b1_workhorse_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, checkpoint_manifest, state_path = _physical_checkpoint(
        tmp_path / "carrier-physical-parent",
        world_uids=[9101, 9102],
        rng_keys=[71, 72],
        temporal_k=2,
    )
    monkeypatch.setattr(
        indexed_checkpoint,
        "inspect_workhorse_checkpoint_bundle_v2",
        lambda _path, **_kwargs: (deepcopy(metadata), state_path),
    )
    indexed = indexed_checkpoint.write_workhorse_indexed_checkpoint_set_v1(
        checkpoint_manifest,
        tmp_path / "carrier-indexed-parent",
    )
    projection_core = {
        "schema_version": "eiel_workhorse_row_checkpoint_occurrence_projection_v2",
        "claim_class": "TECHNICAL_ONLY",
        "science_authority_granted": False,
        "qualification_authority_granted": False,
        "time_authority": "absolute_tick_v1",
        "checkpoint_at_tick": 25,
        "checkpoint_manifest_path": str(checkpoint_manifest),
        "checkpoint_manifest_file_sha256": "1" * 64,
        "checkpoint_bundle_body_sha256": "2" * 64,
        "checkpoint_identity_sha256": indexed["source_physical_checkpoint"][
            "checkpoint_identity_sha256"
        ],
        "canonical_state_sha256": "3" * 64,
        "canonical_parent_binding_sha256": "4" * 64,
        "operation_contract_sha256": "5" * 64,
        "absolute_tick_bundle_path": str(tmp_path / "absolute-bundle"),
        "absolute_tick_bundle_digest": {"algorithm": "sha256", "hex": "6" * 64},
        "occurrence_root": str(tmp_path / "occurrence"),
        "occurrence_digest": {"algorithm": "sha256", "hex": "8" * 64},
        "occurrence_descriptor_path": str(tmp_path / "occurrence" / "descriptor.json"),
        "occurrence_descriptor_file_sha256": "a" * 64,
        "occurrence_commit_path": str(tmp_path / "occurrence" / "commit.json"),
        "occurrence_commit_file_sha256": "b" * 64,
        "parent_checkpoint_digest": {"algorithm": "sha256", "hex": "9" * 64},
        "canonical_visible": True,
        "restore_admitted_exact": False,
    }
    projection = {
        **projection_core,
        "projection_content_sha256": canonical_sha256(projection_core).hex,
    }
    selected = indexed_checkpoint.materialize_workhorse_selected_world_checkpoint_v1(
        indexed["manifest_path"],
        tmp_path / "carrier-retained-parent",
        world_uid=9102,
        checkpoint_occurrence_projection=projection,
        semantic_world_sha256="c" * 64,
        control_history_sha256="d" * 64,
        rng_state_sha256="e" * 64,
        event_prefix_sha256="f" * 64,
        active_scientific_control_state=workhorse_active_scientific_control_state_v1(
            captured_next_absolute_tick=25,
            resource_multiplier=1.0,
            mutation_rate=0.125,
        ),
    )
    selected_manifest = json.loads(Path(selected["manifest_path"]).read_text())
    selected_manifest_path = Path(selected["manifest_path"])
    source_ref = RunArtifactRefV1(
        artifact_id=ArtifactId("carrier-selected-parent-v1"),
        artifact_kind=branch.WORKHORSE_SELECTED_WORLD_ARTIFACT_KIND,
        artifact_schema_version=SchemaVersion(
            indexed_checkpoint.WORKHORSE_SELECTED_WORLD_CHECKPOINT_SCHEMA_V1
        ),
        locator=str(selected_manifest_path),
        content_digest=ContentDigest(
            "sha256", hashlib.sha256(selected_manifest_path.read_bytes()).hexdigest()
        ),
        size_bytes=selected_manifest_path.stat().st_size,
    )
    authority_path = Path(selected["authority_receipt_path"])
    authority_ref = RunArtifactRefV1(
        artifact_id=ArtifactId("carrier-selected-authority-v1"),
        artifact_kind=branch.WORKHORSE_OCCURRENCE_PROJECTION_ARTIFACT_KIND,
        artifact_schema_version=SchemaVersion(projection["schema_version"]),
        locator=str(authority_path),
        content_digest=ContentDigest(
            "sha256", hashlib.sha256(authority_path.read_bytes()).hexdigest()
        ),
        size_bytes=authority_path.stat().st_size,
    )
    state_ref = StateRefV1(
        expedition_id="carrier-expedition-v1",
        world_trajectory_id="carrier-parent-trajectory-v1",
        logical_world_id=9102,
        next_absolute_tick=25,
        state_class=StateClassV1.BRANCHABLE_EXACT,
        semantic_world_sha256=ContentDigest("sha256", "c" * 64),
        control_history_sha256=ContentDigest("sha256", "d" * 64),
        source_object_sha256=source_ref.content_digest,
        source_artifact_ref=source_ref,
        exact_state_sha256=ContentDigest(
            "sha256", selected_manifest["selected_world"]["state_sha256"]
        ),
        checkpoint_occurrence_sha256=ContentDigest("sha256", "8" * 64),
        rng_state_sha256=ContentDigest("sha256", "e" * 64),
        event_prefix_sha256=ContentDigest("sha256", "f" * 64),
        parent_occurrence_sha256=ContentDigest("sha256", "9" * 64),
        replay_recipe_sha256=None,
    )
    request = _selected_world_request()
    request["world"] = {
        "width": 64,
        "height": 64,
        "population": 1024,
        "capacity": 2048,
    }
    request["continuation"]["state_ref"] = state_ref.to_dict()
    request["continuation"]["authority_receipt_ref"] = authority_ref.to_dict()
    compilation = compile_science_request_v1(request)
    bundle = compilation.world_bundle
    requested_observations: set[str] = set()
    for compiled_world in bundle.worlds:
        acquisition = compiled_world.acquisition.to_value()
        requested_observations.update(acquisition["event_observations"])
        requested_observations.update(acquisition["state_observations"])
        requested_observations.update(acquisition["runtime_counters"])
        for reducer in compiled_world.reducers.to_value():
            requested_observations.update(reducer["required_observations"])
    admission = admit_science_world_bundle_v1(
        bundle,
        supported_observations=tuple(sorted(requested_observations)),
        supported_profiles=("SCIENCE_STANDARD_V1",),
        maximum_total_ticks=30,
    )
    carrier_plan = carrier.plan_science_cuda_v2_execution_v1(
        bundle,
        admission,
        physical_batch_capacity=1,
        temporal_k=2,
    )

    shape = workhorse_shape_v2("w64h64p1024c2048")
    artifact_path = tmp_path / "workhorse.so"
    artifact_path.write_bytes(b"carrier-workhorse")
    artifact = {
        "path": str(artifact_path.resolve()),
        "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "manifest_path": str((tmp_path / "workhorse.manifest.json").resolve()),
        "manifest_file_sha256": "0" * 64,
        "manifest_sha256": "1" * 64,
        "sealed_artifact_identity_sha256": "2" * 64,
        "source_inventory_sha256": "3" * 64,
        "identity": {
            "shape_id": shape.shape_id,
            "backend_id": BACKEND_PHASE_GRAPH_V2,
        },
    }
    receipt_paths = {}
    for name in ("envelope", "micro", "semantic"):
        path = tmp_path / f"{name}.json"
        path.write_text("{}\n")
        receipt_paths[name] = path.resolve()
    device = {"uuid": "GPU-CARRIER-TEST", "total_memory_bytes": 24 * 1024**3}
    envelope = {
        "receipt_path": str(receipt_paths["envelope"]),
        "receipt_file_sha256": hashlib.sha256(
            receipt_paths["envelope"].read_bytes()
        ).hexdigest(),
        "receipt_content_sha256": "4" * 64,
        "artifact": artifact,
        "shape_id": shape.shape_id,
        "backend_id": BACKEND_PHASE_GRAPH_V2,
        "device_identity": device,
        "expected_rows": [
            {
                "B": 1,
                "temporal_k": temporal_k,
                "planned_admission": {
                    "admitted": True,
                    "required_bytes": 128 * 1024**2,
                },
            }
            for temporal_k in (1, 2)
        ],
        "capacity_basis": {
            "planning_available_bytes": 8 * 1024**3,
            "effective_reserve_bytes": 512 * 1024**2,
        },
    }
    micro = {
        "receipt_path": str(receipt_paths["micro"]),
        "receipt_file_sha256": hashlib.sha256(
            receipt_paths["micro"].read_bytes()
        ).hexdigest(),
        "receipt_content_sha256": "5" * 64,
    }
    semantic_gate = {
        "receipt_path": str(receipt_paths["semantic"]),
        "receipt_file_sha256": hashlib.sha256(
            receipt_paths["semantic"].read_bytes()
        ).hexdigest(),
        "semantic_gate_receipt_sha256": "6" * 64,
    }
    artifact_binding = {
        key: artifact[key]
        for key in ("path", "sha256", "sealed_artifact_identity_sha256")
    }
    base_contract = FullEvolutionCudaWorkhorseContractV2(
        shape_id=shape.shape_id,
        backend_id=BACKEND_PHASE_GRAPH_V2,
        B=1,
        temporal_k=2,
        total_ticks=5,
    )
    base_operation = {
        "schema_version": workhorse_provider.OPERATION_SCHEMA_V2,
        "operation": workhorse_provider.EXECUTION_ROW_OPERATION_V2,
        "contract": base_contract.to_dict(),
        "artifact": artifact_binding,
        "envelope": {
            "path": envelope["receipt_path"],
            "file_sha256": envelope["receipt_file_sha256"],
            "content_sha256": envelope["receipt_content_sha256"],
        },
        "authoritative_micro_receipt": {
            "path": micro["receipt_path"],
            "file_sha256": micro["receipt_file_sha256"],
            "content_sha256": micro["receipt_content_sha256"],
        },
        "semantic_gate_receipt": {
            "path": semantic_gate["receipt_path"],
            "file_sha256": semantic_gate["receipt_file_sha256"],
            "content_sha256": semantic_gate["semantic_gate_receipt_sha256"],
        },
        "science_continuation_contract_sha256": "7" * 64,
        "source_checkpoint_identity_sha256": None,
        "measurement_schedule": {},
    }
    acs_extension = tmp_path / "acs-native.so"
    acs_manifest = tmp_path / "acs-native.so.manifest.json"
    acs_extension.write_bytes(b"acs-native")
    acs_manifest.write_text("{}\n")
    lifecycle = lifecycle_profile_contract_v1(
        CHECKPOINT_TRANSIT,
        total_max_bytes=4 * 1024**3,
        checkpoint_schedule=(5,),
        checkpoint_authority="absolute_tick",
    )
    base_plan = {
        "schema_version": "eiel_campaign_plan_v1",
        "campaign_id": "workhorse-template-v1",
        "description": "Workhorse template",
        "no_claims": "TECHNICAL_ONLY",
        "resolved_profile": shape.profile_id,
        "cells": [],
        "physical_batches": [],
        "comparisons": [],
        "execution": {
            "budgets": {
                "vram_bytes": 8 * 1024**3,
                "minimum_free_gpu_bytes": 512 * 1024**2,
                "disk_bytes": 4 * 1024**3,
                "host_bytes": 2 * 1024**3,
                "wall_seconds": 300,
            },
            "technical_gate": {},
        },
        "artifact_lifecycle": lifecycle,
        "artifacts": {
            ACS_NATIVE_RUNTIME_EXTENSION_ARTIFACT_ID_V1: {
                "path": str(acs_extension.resolve()),
                "sha256": hashlib.sha256(acs_extension.read_bytes()).hexdigest(),
            },
            ACS_NATIVE_RUNTIME_MANIFEST_ARTIFACT_ID_V1: {
                "path": str(acs_manifest.resolve()),
                "sha256": hashlib.sha256(acs_manifest.read_bytes()).hexdigest(),
            },
        },
        "resource_estimate": {},
        "analysis": {
            "required_metrics": [],
            "optional_metrics": [],
            "rules": [],
            "extension": {
                "id": workhorse_provider.OPERATION_EXTENSION_ID,
                "contract": base_operation,
                "contract_sha256": workhorse_provider.canonical_sha256(
                    base_operation
                ),
            },
        },
    }
    monkeypatch.setattr(
        carrier,
        "resolve_workhorse_selected_world_continuation_v1",
        lambda **_kwargs: SimpleNamespace(
            manifest_path=str(selected_manifest_path),
            parent_run_id=RunId("carrier-parent-run-v1"),
            checkpoint_occurrence_sha256=state_ref.checkpoint_occurrence_sha256,
            parent_occurrence_sha256=state_ref.parent_occurrence_sha256,
        ),
    )
    monkeypatch.setattr(carrier, "workhorse_artifact_binding_v2", lambda _path: artifact)
    monkeypatch.setattr(
        carrier,
        "validate_workhorse_device_envelope_calibration_receipt_v2",
        lambda _path: envelope,
    )
    monkeypatch.setattr(
        carrier,
        "validate_authoritative_micro_denominator_v2",
        lambda _path, **_kwargs: micro,
    )
    monkeypatch.setattr(
        carrier,
        "validate_workhorse_semantic_gate_receipt_v2",
        lambda _path, **_kwargs: semantic_gate,
    )
    monkeypatch.setattr(
        workhorse_provider,
        "workhorse_artifact_binding_v2",
        lambda _path: artifact,
    )
    monkeypatch.setattr(
        workhorse_provider,
        "validate_workhorse_device_envelope_calibration_receipt_v2",
        lambda _path: envelope,
    )

    with pytest.raises(
        carrier.ScienceWorldBundleError,
        match="technical-only until its authenticated observation adapter exists",
    ):
        carrier.lower_science_cuda_v2_batch_to_campaign_plan_v1(
            base_plan,
            bundle,
            admission,
            carrier_plan,
            physical_batch_index=0,
            observation_mode=carrier.SCIENCE_CUDA_V2_OBSERVATION_OFF_V1,
            science_use_class=(
                carrier.SCIENCE_CUDA_V2_USE_CLASS_SCIENTIFIC_EXECUTION_V1
            ),
        )

    lowered = carrier.lower_science_cuda_v2_batch_to_campaign_plan_v1(
        base_plan,
        bundle,
        admission,
        carrier_plan,
        physical_batch_index=0,
        observation_mode=carrier.SCIENCE_CUDA_V2_OBSERVATION_OFF_V1,
        science_use_class=(
            carrier.SCIENCE_CUDA_V2_USE_CLASS_ENGINEERING_DEMONSTRATION_V1
        ),
    )
    operation = workhorse_provider._operation_contract(lowered)
    continuation = operation["continuation_contract"]
    assert operation["operation"] == (
        workhorse_provider.INDEXED_ACS_CONTINUATION_OPERATION_V2
    )
    assert operation["contract"] == {
        **base_contract.to_dict(),
        "B": 1,
        "total_ticks": 5,
        "start_next_tick": 25,
    }
    assert lowered["execution"]["start_next_tick"] == 25
    assert lowered["execution"]["total_ticks"] == 30
    assert lowered["execution"]["checkpoint_ticks"] == [28, 30]
    assert lowered["execution"]["acs_checkpoint_slots"] == 2
    assert lowered["cells"][0]["seeds"] == {}
    assert lowered["cells"][0]["native_rng_world_uid"] is None
    assert [
        row["world_uid"]
        for row in continuation["repack_binding"][
            "selected_worlds_in_physical_order"
        ]
    ] == [9102]
    assert selected_manifest["source_indexed_checkpoint"]["source_physical_slot"] == 1
    assert continuation["repack_binding"]["selected_worlds_in_physical_order"][0][
        "target_physical_slot"
    ] == 0
    assert continuation["scientific_control_contract"][
        "effective_next_absolute_tick"
    ] == 25
    assert continuation["scientific_control_contract"]["controls"] == {
        "environment.resource.multiplier": 1.5,
        "mutation.rate": 0.25,
    }
    assert lowered["science_materialization"]["parent_run_id"] == (
        "carrier-parent-run-v1"
    )
    assert lowered["science_materialization"]["science_authority_granted"] is False
    assert all(
        row["retention_class"] == "R4"
        for row in lowered["artifact_lifecycle"]["checkpoints"]["roles"]
    )


def test_snapshots_and_tampered_event_prefix_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_ref, occurrence, controls, run_root, events, manifest = _exact_fixture(
        tmp_path,
        monkeypatch,
    )
    observation_only = replace(
        state_ref,
        state_class=StateClassV1.OBSERVATION_ONLY,
        exact_state_sha256=None,
        checkpoint_occurrence_sha256=None,
        rng_state_sha256=None,
        event_prefix_sha256=None,
        parent_occurrence_sha256=None,
    )
    with pytest.raises(branch.SelectedWorldBranchError, match="not a direct exact"):
        branch.resolve_selected_world_state_v1(
            state_ref=observation_only,
            checkpoint_set_manifest_path=manifest,
            selected_world_id="cell-001",
            checkpoint_occurrence=occurrence,
            control_history_prefix=controls,
            parent_run_root=run_root,
        )

    raw = bytearray(events.read_bytes())
    raw[0] ^= 0xFF
    events.write_bytes(raw)
    with pytest.raises(branch.SelectedWorldBranchError, match="event prefix"):
        branch.resolve_selected_world_state_v1(
            state_ref=state_ref,
            checkpoint_set_manifest_path=manifest,
            selected_world_id="cell-001",
            checkpoint_occurrence=occurrence,
            control_history_prefix=controls,
            parent_run_root=run_root,
        )


def test_selected_world_rejects_another_cells_authentic_event_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_ref, occurrence, controls, run_root, _events, manifest = _exact_fixture(
        tmp_path,
        monkeypatch,
        swap_selected_event_prefix=True,
    )
    with pytest.raises(branch.SelectedWorldBranchError, match="different physical cell"):
        branch.resolve_selected_world_state_v1(
            state_ref=state_ref,
            checkpoint_set_manifest_path=manifest,
            selected_world_id="cell-001",
            checkpoint_occurrence=occurrence,
            control_history_prefix=controls,
            parent_run_root=run_root,
        )


def test_selected_world_identity_chain_fails_closed_on_each_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_ref, occurrence, controls, run_root, _events, manifest = _exact_fixture(
        tmp_path,
        monkeypatch,
    )
    arguments = {
        "checkpoint_set_manifest_path": manifest,
        "selected_world_id": "cell-001",
        "checkpoint_occurrence": occurrence,
        "control_history_prefix": controls,
        "parent_run_root": run_root,
    }

    with pytest.raises(branch.SelectedWorldBranchError, match="logical-world"):
        branch.resolve_selected_world_state_v1(
            state_ref=replace(state_ref, logical_world_id=7001),
            **arguments,
        )

    wrong_rng = canonical_sha256({"rng": "wrong"})
    with pytest.raises(branch.SelectedWorldBranchError, match="RNG identity"):
        branch.resolve_selected_world_state_v1(
            state_ref=replace(state_ref, rng_state_sha256=wrong_rng),
            **arguments,
        )

    wrong_source = canonical_sha256({"world-object": "wrong"})
    wrong_artifact = replace(
        state_ref.source_artifact_ref,
        content_digest=wrong_source,
    )
    with pytest.raises(branch.SelectedWorldBranchError, match="source/exact-state"):
        branch.resolve_selected_world_state_v1(
            state_ref=replace(
                state_ref,
                source_object_sha256=wrong_source,
                source_artifact_ref=wrong_artifact,
            ),
            **arguments,
        )

    changed_occurrence = {
        **occurrence,
        "semantic_spec_digest": canonical_sha256(
            {"semantic": "different occurrence"}
        ).to_dict(),
    }
    with pytest.raises(branch.SelectedWorldBranchError, match="occurrence digest"):
        branch.resolve_selected_world_state_v1(
            state_ref=state_ref,
            **{**arguments, "checkpoint_occurrence": changed_occurrence},
        )

    changed_controls = {
        **controls,
        "applied_control_prefix": {
            "control_readback_receipt_sha256s": ["d" * 64],
            "prefix_end_next_absolute_tick": 960,
        },
    }
    with pytest.raises(branch.SelectedWorldBranchError, match="control-history"):
        branch.resolve_selected_world_state_v1(
            state_ref=state_ref,
            **{**arguments, "control_history_prefix": changed_controls},
        )


def test_changed_suffix_cannot_precede_branch_or_be_a_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_ref, occurrence, controls, run_root, _events, manifest = _exact_fixture(
        tmp_path,
        monkeypatch,
    )
    resolved = branch._inspect_selected_world_material_v1(
        state_ref=state_ref,
        checkpoint_set_manifest_path=manifest,
        selected_world_id="cell-001",
        checkpoint_occurrence=occurrence,
        control_history_prefix=controls,
        parent_run_root=run_root,
    )
    early = {
        "schema_version": branch.CONTROL_SUFFIX_SCHEMA,
        "effective_next_absolute_tick": 960,
        "controls": [{"control_id": "x", "onset_tick": 959, "payload": {}}],
    }
    with pytest.raises(branch.SelectedWorldBranchError, match="onset must equal"):
        branch.prepare_selected_world_branch_v1(
            resolved,
            child_run_id=RunId("child-run-v1"),
            child_trajectory_id="trajectory-child-v1",
            effective_next_absolute_tick=960,
            control_suffix=early,
        )
    delayed = {
        "schema_version": branch.CONTROL_SUFFIX_SCHEMA,
        "effective_next_absolute_tick": 961,
        "controls": [{"control_id": "x", "onset_tick": 961, "payload": {}}],
    }
    with pytest.raises(branch.SelectedWorldBranchError, match="must equal"):
        branch.prepare_selected_world_branch_v1(
            resolved,
            child_run_id=RunId("child-run-v1"),
            child_trajectory_id="trajectory-child-v1",
            effective_next_absolute_tick=961,
            control_suffix=delayed,
        )
    no_onset = {
        "schema_version": branch.CONTROL_SUFFIX_SCHEMA,
        "effective_next_absolute_tick": 960,
        "controls": [{"control_id": "x", "tick": 960, "payload": {}}],
    }
    with pytest.raises(branch.SelectedWorldBranchError, match="unknown or missing"):
        branch.prepare_selected_world_branch_v1(
            resolved,
            child_run_id=RunId("child-run-v1"),
            child_trajectory_id="trajectory-child-v1",
            effective_next_absolute_tick=960,
            control_suffix=no_onset,
        )
    hidden_timing = {
        "schema_version": branch.CONTROL_SUFFIX_SCHEMA,
        "effective_next_absolute_tick": 960,
        "controls": [
            {
                "control_id": "x",
                "onset_tick": 960,
                "payload": {"alternate_schedule_tick": 959},
            }
        ],
    }
    with pytest.raises(branch.SelectedWorldBranchError, match="hidden temporal"):
        branch.prepare_selected_world_branch_v1(
            resolved,
            child_run_id=RunId("child-run-v1"),
            child_trajectory_id="trajectory-child-v1",
            effective_next_absolute_tick=960,
            control_suffix=hidden_timing,
        )
    valid = {
        "schema_version": branch.CONTROL_SUFFIX_SCHEMA,
        "effective_next_absolute_tick": 960,
        "controls": [{"control_id": "x", "onset_tick": 960, "payload": {}}],
    }
    with pytest.raises(branch.SelectedWorldBranchError, match="never a retry"):
        branch.prepare_selected_world_branch_v1(
            resolved,
            child_run_id=RunId("child-run-v1"),
            child_trajectory_id="trajectory-child-v1",
            effective_next_absolute_tick=960,
            control_suffix=valid,
            retry_of_run_id=RunId("parent-run-v1"),
        )


def test_current_science_front_door_carrier_has_authenticated_host_boundary() -> None:
    branch.assert_canonical_selected_world_branch_execution_supported_v1()


@pytest.mark.parametrize(
    "defect",
    [None, "NULL_TERMINAL_PARENT", "FABRICATED_TERMINAL_PARENT", "WRONG_GENESIS"],
)
def test_founder_genesis_chain_is_real_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    defect: str | None,
) -> None:
    run_root = tmp_path / "run"
    occurrence_parent = run_root / "workhorse-checkpoint-occurrences"
    bundle_parent = run_root / "workhorse-checkpoints"
    genesis_tick, terminal_tick = 1, 6
    genesis = replace(
        _occurrence(
            tick=genesis_tick,
            parent_digest=canonical_sha256({"unused": "genesis"}),
        ),
        parent_checkpoint_digest=(
            canonical_sha256({"wrong": "genesis"})
            if defect == "WRONG_GENESIS"
            else None
        ),
    )
    terminal_parent = genesis.occurrence_digest()
    if defect == "NULL_TERMINAL_PARENT":
        terminal_parent = None
    elif defect == "FABRICATED_TERMINAL_PARENT":
        terminal_parent = canonical_sha256({"fabricated": "parent"})
    terminal = replace(
        _occurrence(
            tick=terminal_tick,
            parent_digest=genesis.occurrence_digest(),
        ),
        parent_checkpoint_digest=terminal_parent,
    )

    def publish(root: Path, occurrence: AbsoluteTickCheckpointOccurrenceV1) -> None:
        root.mkdir(parents=True)
        descriptor = root / branch.ABSOLUTE_TICK_OCCURRENCE_DESCRIPTOR_NAME
        commit_path = root / branch.ABSOLUTE_TICK_OCCURRENCE_COMMIT_NAME
        descriptor.write_text(json.dumps(occurrence.to_dict(), sort_keys=True))
        commit = AbsoluteTickCheckpointOccurrenceCommitV1(
            occurrence_digest=occurrence.occurrence_digest(),
            checkpoint_bundle_digest=occurrence.checkpoint_bundle_digest,
            checkpoint_format_id=occurrence.checkpoint_format_id,
            committed_at_utc="2026-08-29T00:00:00Z",
        )
        commit_path.write_text(json.dumps(commit.to_dict(), sort_keys=True))

    genesis_root = occurrence_parent / f"checkpoint-tick-{genesis_tick:012d}"
    terminal_root = occurrence_parent / f"checkpoint-tick-{terminal_tick:012d}"
    genesis_bundle = bundle_parent / f"checkpoint-tick-{genesis_tick:012d}"
    terminal_bundle = bundle_parent / f"checkpoint-tick-{terminal_tick:012d}"
    publish(genesis_root, genesis)
    publish(terminal_root, terminal)
    genesis_bundle.mkdir(parents=True)
    terminal_bundle.mkdir(parents=True)
    monkeypatch.setattr(
        branch,
        "inspect_absolute_tick_checkpoint_occurrence_v1",
        lambda *_args, **_kwargs: SimpleNamespace(
            status=branch.CheckpointOccurrenceInspectionStatusV1.VISIBLE
        ),
    )
    descriptor_path = terminal_root / branch.ABSOLUTE_TICK_OCCURRENCE_DESCRIPTOR_NAME
    commit_path = terminal_root / branch.ABSOLUTE_TICK_OCCURRENCE_COMMIT_NAME
    core = {
        "schema_version": (
            "eiel_workhorse_row_checkpoint_occurrence_projection_v2"
        ),
        "claim_class": "TECHNICAL_ONLY",
        "science_authority_granted": False,
        "qualification_authority_granted": False,
        "time_authority": "absolute_tick_v1",
        "checkpoint_at_tick": terminal_tick,
        "checkpoint_manifest_path": str(terminal_bundle / "manifest.json"),
        "checkpoint_manifest_file_sha256": "1" * 64,
        "checkpoint_bundle_body_sha256": "2" * 64,
        "checkpoint_identity_sha256": "3" * 64,
        "canonical_state_sha256": "4" * 64,
        "canonical_parent_binding_sha256": "5" * 64,
        "operation_contract_sha256": "6" * 64,
        "absolute_tick_bundle_path": str(terminal_bundle),
        "absolute_tick_bundle_digest": terminal.checkpoint_bundle_digest.to_dict(),
        "occurrence_root": str(terminal_root),
        "occurrence_digest": terminal.occurrence_digest().to_dict(),
        "occurrence_descriptor_path": str(descriptor_path),
        "occurrence_descriptor_file_sha256": hashlib.sha256(
            descriptor_path.read_bytes()
        ).hexdigest(),
        "occurrence_commit_path": str(commit_path),
        "occurrence_commit_file_sha256": hashlib.sha256(
            commit_path.read_bytes()
        ).hexdigest(),
        "parent_checkpoint_digest": (
            None
            if terminal.parent_checkpoint_digest is None
            else terminal.parent_checkpoint_digest.to_dict()
        ),
        "canonical_visible": True,
        "restore_admitted_exact": False,
    }
    projection = {**core, "projection_content_sha256": canonical_sha256(core).hex}
    if defect is None:
        observed_genesis, observed_terminal = branch._founder_occurrence_chain_v1(
            projection,
            genesis_tick=genesis_tick,
            terminal_tick=terminal_tick,
            run_root=run_root,
        )
        assert observed_genesis.parent_checkpoint_digest is None
        assert (
            observed_terminal.parent_checkpoint_digest
            == observed_genesis.occurrence_digest()
        )
    else:
        with pytest.raises(
            branch.SelectedWorldBranchError,
            match="genesis/terminal occurrence chain differs",
        ):
            branch._founder_occurrence_chain_v1(
                projection,
                genesis_tick=genesis_tick,
                terminal_tick=terminal_tick,
                run_root=run_root,
            )


@pytest.mark.parametrize(
    "tamper",
    [None, "SOURCE", "CONTROL", "EVENT", "INDEX", "PARENT_CHAIN"],
)
def test_completed_exact_child_promotes_only_from_complete_authenticated_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str | None,
) -> None:
    parent_manifest = tmp_path / "parent-selected.json"
    parent_manifest.write_text("{}")
    parent_digest = ContentDigest("sha256", hashlib.sha256(b"{}").hexdigest())
    parent_artifact = RunArtifactRefV1(
        artifact_id=ArtifactId("parent-selected-world-v1"),
        artifact_kind=branch.WORKHORSE_SELECTED_WORLD_ARTIFACT_KIND,
        artifact_schema_version=SchemaVersion(
            indexed_checkpoint.WORKHORSE_SELECTED_WORLD_CHECKPOINT_SCHEMA_V1
        ),
        locator=str(parent_manifest),
        content_digest=parent_digest,
        size_bytes=2,
    )
    source_occurrence = canonical_sha256({"occurrence": "source"})
    source_parent = canonical_sha256({"occurrence": "source-parent"})
    parent = StateRefV1(
        expedition_id="expedition-child-promotion-v1",
        world_trajectory_id="trajectory-parent-v1",
        logical_world_id=77,
        next_absolute_tick=25,
        state_class=StateClassV1.BRANCHABLE_EXACT,
        semantic_world_sha256=ContentDigest("sha256", "1" * 64),
        control_history_sha256=ContentDigest("sha256", "2" * 64),
        source_object_sha256=parent_digest,
        source_artifact_ref=parent_artifact,
        exact_state_sha256=ContentDigest("sha256", "3" * 64),
        checkpoint_occurrence_sha256=source_occurrence,
        rng_state_sha256=ContentDigest("sha256", "4" * 64),
        event_prefix_sha256=ContentDigest("sha256", "5" * 64),
        parent_occurrence_sha256=source_parent,
        replay_recipe_sha256=None,
    )
    monkeypatch.setattr(
        branch,
        "workhorse_selected_world_authority_ref_v1",
        lambda _state: object(),
    )
    monkeypatch.setattr(
        branch,
        "resolve_workhorse_selected_world_continuation_v1",
        lambda **_kwargs: SimpleNamespace(state_ref=parent),
    )

    run_root = tmp_path / "child-provider"
    run_root.mkdir()
    host_path = run_root / "workhorse-selected-world-host-observation.json"
    host_path.write_text("{}")
    result_path = run_root / "workhorse-indexed-continuation-result.json"
    result_path.write_text("{}")
    indexed_path = run_root / "indexed-checkpoint.json"
    indexed_path.write_text("{}")
    terminal_occurrence = canonical_sha256({"occurrence": "terminal"})
    projection_core = {
        "schema_version": (
            "eiel_workhorse_indexed_continuation_checkpoint_occurrence_projection_v2"
        ),
        "claim_class": "TECHNICAL_ONLY",
        "science_authority_granted": False,
        "qualification_authority_granted": False,
        "time_authority": "absolute_tick_v1",
        "checkpoint_at_tick": 30,
        "checkpoint_manifest_path": str(run_root / "checkpoint.json"),
        "checkpoint_manifest_file_sha256": "6" * 64,
        "checkpoint_bundle_body_sha256": "7" * 64,
        "checkpoint_identity_sha256": "8" * 64,
        "canonical_state_sha256": "9" * 64,
        "canonical_parent_binding_sha256": "a" * 64,
        "operation_contract_sha256": "b" * 64,
        "absolute_tick_bundle_path": str(run_root / "checkpoint-tick-30"),
        "absolute_tick_bundle_digest": canonical_sha256(
            {"bundle": "terminal"}
        ).to_dict(),
        "occurrence_root": str(run_root / "occurrence-tick-30"),
        "occurrence_digest": terminal_occurrence.to_dict(),
        "occurrence_descriptor_path": str(run_root / "descriptor.json"),
        "occurrence_descriptor_file_sha256": "c" * 64,
        "occurrence_commit_path": str(run_root / "commit.json"),
        "occurrence_commit_file_sha256": "d" * 64,
        "parent_checkpoint_digest": (
            canonical_sha256({"wrong": "parent"}).to_dict()
            if tamper == "PARENT_CHAIN"
            else source_occurrence.to_dict()
        ),
        "canonical_visible": True,
        "restore_admitted_exact": False,
    }
    projection = {
        **projection_core,
        "projection_content_sha256": canonical_sha256(projection_core).hex,
    }
    control_contract = {
        "control_contract_sha256": "e" * 64,
        "controls": {
            "environment.resource.multiplier": 1.5,
            "mutation.rate": 0.25,
        },
    }
    application = {
        "control_contract_sha256": (
            "0" * 64 if tamper == "CONTROL" else "e" * 64
        ),
        "control_application_sha256": "f" * 64,
    }
    logical = {
        "source_checkpoint_occurrence_sha256": (
            "0" * 64 if tamper == "SOURCE" else source_occurrence.hex
        ),
        "source_parent_occurrence_sha256": source_parent.hex,
        "source_semantic_world_sha256": parent.semantic_world_sha256.hex,
        "source_control_history_sha256": parent.control_history_sha256.hex,
        "source_rng_state_sha256": parent.rng_state_sha256.hex,
        "source_event_prefix_sha256": parent.event_prefix_sha256.hex,
        "science_continuation_contract_sha256": "a" * 64,
        "start_next_tick": 25,
        "target_next_tick": 30,
        "world_count": 1,
    }
    ack = {
        "world_uids": [77],
        "start_next_absolute_tick": 24 if tamper == "EVENT" else 25,
        "end_next_absolute_tick": 30,
        "receipt_body_sha256": "1" * 64,
        "dispatch_identity_sha256": "2" * 64,
        "rng_identity_sha256": "3" * 64,
    }
    result = {
        "receipt_file_sha256": "6" * 64,
        "receipt_content_sha256": "7" * 64,
        "continuation_contract": {
            "science_continuation_identity": logical,
            "scientific_control_contract": control_contract,
            "checkpoint_ticks": [30],
        },
        "control_application": application,
        "restored_world_uids": [77],
        "dispatch_measurements": [{"event_sink_receipt": ack}],
    }
    observation = {
        "source_continuation": {
            "execution_result_path": str(result_path),
            "execution_result_file_sha256": "6" * 64,
            "execution_result_content_sha256": "7" * 64,
        },
        "terminal_checkpoint": {
            "indexed_manifest_path": str(indexed_path),
            "checkpoint_at_tick": 30,
            "checkpoint_occurrence_sha256": terminal_occurrence.hex,
            "checkpoint_occurrence_projection_sha256": projection[
                "projection_content_sha256"
            ],
        },
        "world_observations": [{"world_uid": 77}],
        "observation_file_sha256": hashlib.sha256(b"{}").hexdigest(),
        "observation_content_sha256": "4" * 64,
    }
    publication = {
        "checkpoint_at_tick": 30,
        "occurrence_digest": terminal_occurrence.hex,
        "checkpoint_occurrence_projection_content_sha256": projection[
            "projection_content_sha256"
        ],
        "indexed_manifest_file_sha256": (
            "0" * 64
            if tamper == "INDEX"
            else hashlib.sha256(indexed_path.read_bytes()).hexdigest()
        ),
    }
    provider_result = {
        "execution_result_path": str(result_path),
        "execution_result_file_sha256": "6" * 64,
        "execution_result_content_sha256": "7" * 64,
        "indexed_checkpoint_receipts": [
            {
                "checkpoint_occurrence_projection": projection,
                "indexed_checkpoint": {"manifest_path": str(indexed_path)},
            }
        ],
        "host_boundary_observation": {
            "path": str(host_path),
            "file_sha256": observation["observation_file_sha256"],
            "content_sha256": observation["observation_content_sha256"],
        },
    }
    (run_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "eiel_campaign_batch_manifest_v1",
                "status": "completed",
                "provider_id": (
                    "eiel-full-evolution-cuda-workhorse-v2-provider-v1"
                ),
                "result": provider_result,
            }
        )
    )
    monkeypatch.setattr(
        branch,
        "validate_workhorse_selected_world_observation_v1",
        lambda _path: observation,
    )
    monkeypatch.setattr(
        branch,
        "validate_workhorse_indexed_continuation_result_v2",
        lambda _path: result,
    )
    monkeypatch.setattr(
        branch,
        "validate_workhorse_terminalized_indexed_continuation_receipt_v2",
        lambda _path: {"indexed_checkpoint_publications": [publication]},
    )
    monkeypatch.setattr(
        branch,
        "inspect_workhorse_indexed_checkpoint_set_v1",
        lambda _path: (
            {
                "world_index_by_uid": [{"world_uid": 77, "rng_key": 991}],
                "world_state_set_identity": {"rng_epoch": 4},
            },
            indexed_path,
        ),
    )
    sentinel = (object(), object(), object())
    captured: dict[str, object] = {}

    def materialize(*_args: object, **kwargs: object):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        branch,
        "materialize_workhorse_selected_world_state_ref_v1",
        materialize,
    )
    if tamper is None:
        assert branch.materialize_workhorse_child_exact_state_ref_v1(
            host_path,
            tmp_path / "promoted",
            parent_state_ref=parent,
            expedition_id=parent.expedition_id,
            world_trajectory_id="trajectory-child-v1",
        ) == sentinel
        assert captured["checkpoint_occurrence_projection"] == projection
        assert captured["semantic_world_sha256"] == "a" * 64
        assert captured["control_history_sha256"] not in {
            parent.control_history_sha256.hex,
            "f" * 64,
        }
        assert captured["rng_state_sha256"] != parent.rng_state_sha256.hex
        assert captured["event_prefix_sha256"] != parent.event_prefix_sha256.hex
    else:
        with pytest.raises(branch.SelectedWorldBranchError):
            branch.materialize_workhorse_child_exact_state_ref_v1(
                host_path,
                tmp_path / "promoted",
                parent_state_ref=parent,
                expedition_id=parent.expedition_id,
                world_trajectory_id="trajectory-child-v1",
            )

    repo = Path(__file__).resolve().parents[1]
    compact_contract = (
        repo / "emergence_contracts" / "science_request_v1.py"
    ).read_text(encoding="utf-8")
    qrr_projection = (
        repo / "eiel_composition" / "science_qrr_projection_v1.py"
    ).read_text(encoding="utf-8")
    provider = (
        repo / "engine" / "campaigns" / "cuda_full_evolution_v2_provider.py"
    ).read_text(encoding="utf-8")
    assert '"continuation"' in compact_contract
    assert "BRANCHABLE_EXACT" in compact_contract
    assert "current flat QRR projection cannot represent a rich continuation" in (
        qrr_projection
    )
    assert '"status": "BASE_CHECKPOINT_ONLY_NOT_SCIENCE_CONTINUABLE"' in provider
    assert '"science_continuable": False' in provider
