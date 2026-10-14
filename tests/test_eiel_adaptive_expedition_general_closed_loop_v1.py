from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pytest

from eiel_composition.adaptive_expedition import continuation_adapter as branch
from eiel_composition.adaptive_expedition.coordinator import prepare_decision_v1
from eiel_composition.adaptive_expedition.general_closed_loop import (
    GeneralClosedLoopDependenciesV1,
    GeneralClosedLoopError,
    build_exact_state_child_continuation_v1,
    run_general_closed_loop_v1,
)
from eiel_composition.adaptive_expedition.ledger import (
    ACTION_DISPATCH_BOUND,
    ACTION_SETTLED,
    DECISION_PREPARED,
    read_ledger_store_v1,
)
from eiel_composition.adaptive_expedition.lifecycle import _inventory_binding
from eiel_composition.adaptive_expedition.policy_protocol import (
    canonical_plain_value_v1,
    freeze_canonical_mapping_v1,
    policy_binding_v1,
    seal_policy_proposal_v1,
    validate_policy_input_v1,
)
from eiel_composition.science_request_v1_compiler import compile_science_request_v1
from eiel_composition.science_world_bundle_v1 import admit_science_world_bundle_v1
from eiel_composition import science_cuda_v2_carrier_v1 as carrier
from emergence_contracts import canonical_json_bytes
from emergence_contracts.adaptive_science_expedition_v1 import (
    ChildRealizationKindV1,
    ExpeditionManifestV1,
    GeneratedChildContinuationV1,
    ObservationFrameV1,
    RetainedParentBankV1,
    StateRefV1,
)
from emergence_contracts.canonical import canonical_sha256
from emergence_contracts.checkpoint_occurrence import (
    AbsoluteTickCheckpointOccurrenceCommitV1,
)
from emergence_contracts.ids import ContentDigest, RunId, SchemaVersion
from emergence_contracts.payloads import CanonicalPayloadV1
from emergence_contracts.science_request_v1 import validate_eiel_science_request_v1
from engine.perf.cuda_full_evolution_v2.workhorse import indexed_checkpoint
from engine.perf.cuda_full_evolution_v2.workhorse.scientific_control import (
    workhorse_active_scientific_control_state_v1,
)
from tests.test_eiel_adaptive_expedition_coordinator_v1 import (
    _atlas,
    _bank,
    _frame,
    _manifest,
)
from tests.test_eiel_adaptive_expedition_founder_loop_v1 import _founder_request
from tests.test_eiel_adaptive_expedition_selected_world_branch_v1 import _occurrence
from tests.test_eiel_cuda_full_evolution_v2_workhorse_indexed_checkpoint import (
    _physical_checkpoint,
)


def test_exact_parent_candidate_starts_as_recovery_transient() -> None:
    binding = _inventory_binding(
        "scratch/retained-parent-candidates/exact-child-abc/selected-world/state.bin"
    )
    assert binding == {
        "artifact_family": "adaptive_expedition.exact_parent_candidate_active",
        "retention_class": "R1",
        "owner_writer": "eiel-adaptive-expedition-coordinator-v1",
        "release_predicate": (
            "next deterministic Parent Bank decision recorded and all active "
            "child recovery obligations closed"
        ),
    }


class _ScriptedPolicy:
    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        self._profile = freeze_canonical_mapping_v1(
            {
                "profile_id": f"GENERAL_LOOP_{scenario.upper()}_V1",
                "profile_version": "1",
                "policy_input_authority": {
                    "schema_version": "eiel_adaptive_policy_input_authority_v1",
                    "allowed_paths": [
                        "/available_state_refs/*/next_absolute_tick",
                        "/available_state_refs/*/state_ref_id",
                        "/decision_frontier/frontier_id",
                        "/decision_frontier/frontier_ordinal",
                        "/decision_frontier/observation_frames/*/absolute_tick",
                        "/decision_frontier/observation_frames/*/frame_id",
                        "/decision_frontier/observation_frames/*/state_ref_id",
                        "/decision_frontier/observation_frames/*/trajectory_id",
                        "/expedition_id",
                        "/logical_capacity",
                        "/manifest_sha256",
                        "/outer_rng/cursor",
                        "/outer_rng/seed",
                        "/policy_binding",
                        "/protocol_version",
                        "/retained_parent_bank/entries/*/state_ref_id",
                    ],
                },
                "probe_contracts": [],
            }
        )
        self._binding = policy_binding_v1(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            algorithm_sha256=canonical_sha256(
                {"algorithm": "general-loop-scripted-policy-v1"}
            ),
            profile=self._profile,
            profile_schema_version="eiel_general_loop_scripted_profile_v1",
        )

    @property
    def policy_id(self) -> str:
        return "EIEL_GENERAL_LOOP_SCRIPTED_POLICY_V1"

    @property
    def policy_version(self) -> str:
        return "1"

    @property
    def profile(self):
        return self._profile

    @property
    def binding(self):
        return self._binding

    def decide(self, decision_input):
        checked = validate_policy_input_v1(
            decision_input, expected_binding=self.binding, profile=self.profile
        )
        value = canonical_plain_value_v1(checked)
        frontier = value["decision_frontier"]
        ordinal = frontier["frontier_ordinal"]
        frames = frontier["observation_frames"]
        bank_entries = value["retained_parent_bank"]["entries"]
        atlas_delta = {"upsert_entries": [], "retire_entry_ids": []}
        bank_delta = {
            "retain_entries": [],
            "release_state_ref_ids": [],
            "not_retained": [],
        }
        actions = []

        if self.scenario == "all_actions":
            assert len(frames) == 3
            first, second, third = frames
            atlas_delta["upsert_entries"] = [
                {
                    "entry_id": f"atlas-{ordinal:04d}-one",
                    "frame_id": first["frame_id"],
                    "trajectory_id": first["trajectory_id"],
                    "frontier_ordinal": ordinal,
                    "observation_absolute_tick": first["absolute_tick"],
                    "descriptor_values": {"opaque": 1},
                    "niche_id": "same-observed-niche",
                    "representative_role": "CANONICAL_INTERPRETIVE",
                    "reason_codes": ["COMPACT_INTERPRETIVE_REPRESENTATIVE"],
                }
            ]
            bank_delta["retain_entries"] = [
                {
                    "state_ref_id": first["state_ref_id"],
                    "trajectory_id": first["trajectory_id"],
                    "frontier_ordinal": ordinal,
                    "observation_absolute_tick": first["absolute_tick"],
                    "parent_next_absolute_tick": first["absolute_tick"],
                    "niche_id": "same-observed-niche",
                    "survival_reason": "DIVERGENT_LINEAGE_STEPPING_STONE",
                    "divergence_signature": ["lineage-a"],
                    "source_frame_id": first["frame_id"],
                },
                {
                    "state_ref_id": second["state_ref_id"],
                    "trajectory_id": second["trajectory_id"],
                    "frontier_ordinal": ordinal,
                    "observation_absolute_tick": second["absolute_tick"],
                    "parent_next_absolute_tick": second["absolute_tick"],
                    "niche_id": "same-observed-niche",
                    "survival_reason": "DIVERGENT_STRUCTURE_STEPPING_STONE",
                    "divergence_signature": ["structure-b"],
                    "source_frame_id": second["frame_id"],
                },
            ]
            bank_delta["not_retained"] = [
                {
                    "state_ref_id": third["state_ref_id"],
                    "reason_code": "BOUNDED_BANK_BUDGET_LOWER_PRIORITY",
                }
            ]
            actions = [
                {
                    "action_id": f"action-{ordinal:04d}-global",
                    "action_kind": "EXPLORE_GLOBAL",
                    "bucket": "exploration",
                    "slot_cost": 1,
                    "child_start_next_tick": 0,
                    "frontier_ordinal": ordinal,
                    "control_bindings": [
                        {
                            "control_id": "environment.resource.multiplier",
                            "value": 1.5,
                        },
                        {"control_id": "mutation.rate", "value": 0.25},
                    ],
                    "founder_execution": {
                        "schema_version": (
                            "eiel_adaptive_founder_execution_profile_v1"
                        ),
                        "world_count": 1,
                        "seed_derivation": (
                            "sha256_manifest_action_replicate_ordinal_unique_u63_v1"
                        ),
                    },
                    "reason_codes": ["DECLARED_FOUNDER_EXPLORATION"],
                },
                {
                    "action_id": f"action-{ordinal:04d}-branch",
                    "action_kind": "BRANCH_LOCAL",
                    "bucket": "frontier",
                    "slot_cost": 1,
                    "frontier_ordinal": ordinal,
                    "parent_state_ref_id": first["state_ref_id"],
                    "parent_next_absolute_tick": first["absolute_tick"],
                    "control_bindings": [
                        {
                            "control_id": "environment.resource.multiplier",
                            "value": 1.5,
                        },
                        {"control_id": "mutation.rate", "value": 0.25},
                    ],
                    "reason_codes": ["POLICY_SELECTED_LOCAL_BRANCH"],
                },
                {
                    "action_id": f"action-{ordinal:04d}-replicate",
                    "action_kind": "REPLICATE",
                    "bucket": "replication",
                    "slot_cost": 1,
                    "frontier_ordinal": ordinal,
                    "parent_state_ref_id": second["state_ref_id"],
                    "parent_next_absolute_tick": second["absolute_tick"],
                    "reason_codes": ["POLICY_SELECTED_HISTORICAL_REPLICATE"],
                },
                {
                    "action_id": f"action-{ordinal:04d}-park",
                    "action_kind": "PARK",
                    "bucket": "observation",
                    "slot_cost": 0,
                    "frontier_ordinal": ordinal,
                    "parent_state_ref_id": third["state_ref_id"],
                    "parent_next_absolute_tick": third["absolute_tick"],
                    "reason_codes": ["POLICY_RECORDED_DEAD_END"],
                },
                {
                    "action_id": f"action-{ordinal:04d}-retire",
                    "action_kind": "RETIRE",
                    "bucket": "observation",
                    "slot_cost": 0,
                    "frontier_ordinal": ordinal,
                    "parent_state_ref_id": first["state_ref_id"],
                    "parent_next_absolute_tick": first["absolute_tick"],
                    "reason_codes": ["POLICY_RETIRED_INTERPRETIVE_REFERENCE"],
                },
            ]
        elif self.scenario == "branch_only":
            first = frames[0]
            actions = [
                {
                    "action_id": f"action-{ordinal:04d}-branch",
                    "action_kind": "BRANCH_LOCAL",
                    "bucket": "frontier",
                    "slot_cost": 1,
                    "frontier_ordinal": ordinal,
                    "parent_state_ref_id": first["state_ref_id"],
                    "parent_next_absolute_tick": first["absolute_tick"],
                    "control_bindings": [
                        {
                            "control_id": "environment.resource.multiplier",
                            "value": 1.5,
                        },
                        {"control_id": "mutation.rate", "value": 0.25},
                    ],
                    "reason_codes": ["POLICY_SELECTED_LOCAL_BRANCH"],
                }
            ]
        elif self.scenario == "historical_replicate":
            assert bank_entries
            state_id = bank_entries[0]["state_ref_id"]
            state = next(
                row
                for row in value["available_state_refs"]
                if row["state_ref_id"] == state_id
            )
            actions = [
                {
                    "action_id": f"action-{ordinal:04d}-historical-replicate",
                    "action_kind": "REPLICATE",
                    "bucket": "replication",
                    "slot_cost": 1,
                    "frontier_ordinal": ordinal,
                    "parent_state_ref_id": state_id,
                    "parent_next_absolute_tick": state["next_absolute_tick"],
                    "reason_codes": ["RETAINED_HISTORICAL_REPLICATION"],
                }
            ]
        else:
            raise AssertionError(self.scenario)

        outer = value["outer_rng"]
        return seal_policy_proposal_v1(
            policy_binding=self.binding,
            profile=self.profile,
            decision_input=checked,
            rng_cursor_after=outer["cursor"],
            atlas_delta=atlas_delta,
            retained_parent_bank_delta=bank_delta,
            semantic_actions=actions,
            rejected_candidate_summary={"scenario": self.scenario},
        )


def _founder_basis_request():
    raw = _founder_request()
    raw["world"] = {
        "width": 64,
        "height": 64,
        "population": 1024,
        "capacity": 2048,
    }
    initialization = raw["world_template"]["initialization"]
    initialization["initial_population"] = 1024
    initialization["initial_live_count"] = 1024
    initialization["exact_founder_composition"][0]["count"] = 1024
    return validate_eiel_science_request_v1(raw)


def _general_manifest(policy, founder_request) -> ExpeditionManifestV1:
    raw = _manifest(policy).to_dict()
    raw["expedition_id"] = "expedition-general-loop-v1"
    raw["founder_basis_refs"] = [
        {
            "schema_version": "run_artifact_ref_v1",
            "artifact_id": "general-loop-founder-request-v1",
            "artifact_kind": "eiel_science_request_v1",
            "artifact_schema_version": "eiel_science_request_v1",
            "locator": "opaque://general-loop/founder-request-v1",
            "content_digest": founder_request.request_sha256().to_dict(),
            "size_bytes": len(canonical_json_bytes(founder_request.to_dict())),
        }
    ]
    raw["search_space"] = {
        "schema_version": "adaptive_search_space_v1",
        "payload": {
            "controls": [
                {
                    "control_id": "environment.resource.multiplier",
                    "allowed_values": [1.0, 1.5],
                },
                {
                    "control_id": "mutation.rate",
                    "allowed_values": [0.125, 0.25],
                },
            ]
        },
    }
    raw["epoch_ticks"] = 5
    raw["maximum_absolute_tick"] = 60
    raw["maximum_frontiers"] = 3
    raw["maximum_actions_per_frontier"] = 16
    raw["maximum_retained_parent_entries"] = 4
    raw["maximum_retained_parent_bytes"] = 4 * 1024 * 1024
    return ExpeditionManifestV1.from_dict(raw)


def _synthetic_state(manifest: ExpeditionManifestV1, ordinal: int) -> StateRefV1:
    raw = _frame().source_state_ref.to_dict()
    digest = canonical_sha256({"synthetic-state": ordinal})
    raw.update(
        {
            "expedition_id": manifest.expedition_id,
            "world_trajectory_id": f"synthetic-trajectory-{ordinal}",
            "logical_world_id": 8000 + ordinal,
            "next_absolute_tick": 25,
            "semantic_world_sha256": canonical_sha256(
                {"synthetic-world": ordinal}
            ).to_dict(),
            "control_history_sha256": canonical_sha256(
                {"synthetic-controls": ordinal}
            ).to_dict(),
            "source_object_sha256": digest.to_dict(),
            "source_artifact_ref": {
                "schema_version": "run_artifact_ref_v1",
                "artifact_id": f"synthetic-parent-{ordinal}",
                "artifact_kind": "checkpoint.workhorse_selected_world_v1",
                "artifact_schema_version": (
                    "eiel_workhorse_selected_world_checkpoint_v1"
                ),
                "locator": f"opaque://synthetic-parent/{ordinal}",
                "content_digest": digest.to_dict(),
                "size_bytes": 128 + ordinal,
            },
            "exact_state_sha256": canonical_sha256({"exact-state": ordinal}).to_dict(),
            "checkpoint_occurrence_sha256": canonical_sha256(
                {"occurrence": ordinal}
            ).to_dict(),
            "rng_state_sha256": canonical_sha256({"rng": ordinal}).to_dict(),
            "event_prefix_sha256": canonical_sha256({"events": ordinal}).to_dict(),
            "parent_occurrence_sha256": canonical_sha256(
                {"parent-occurrence": ordinal}
            ).to_dict(),
        }
    )
    return StateRefV1.from_dict(raw)


def _frame_for_state(
    manifest: ExpeditionManifestV1,
    state: StateRefV1,
) -> ObservationFrameV1:
    raw = _frame().to_dict()
    raw.update(
        {
            "expedition_id": manifest.expedition_id,
            "world_trajectory_id": state.world_trajectory_id,
            "logical_world_id": state.logical_world_id,
            "next_absolute_tick": state.next_absolute_tick,
            "semantic_world_sha256": state.semantic_world_sha256.to_dict(),
            "control_history_sha256": state.control_history_sha256.to_dict(),
            "observation_profile_id": manifest.observation_profile_id,
            "observation_descriptor_set_sha256": (
                manifest.observation_descriptor_set_sha256.to_dict()
            ),
            "source_state_ref": state.to_dict(),
        }
    )
    return ObservationFrameV1.from_dict(raw)


@dataclass(frozen=True, slots=True)
class _Projection:
    source: str
    frames: tuple[ObservationFrameV1, ...]
    request_sha256: ContentDigest | None = None

    def projection_sha256(self):
        return canonical_sha256(
            {
                "source": self.source,
                "request_sha256": (
                    None
                    if self.request_sha256 is None
                    else self.request_sha256.to_dict()
                ),
                "frames": [frame.to_dict() for frame in self.frames],
            }
        )


class _Projector:
    def __init__(self, frames):
        self.frames = tuple(frames)
        self._terminal_source_frames = {}
        self._next_terminal_frame = 0

    def __call__(self, source, *, manifest):
        assert all(
            frame.expedition_id == manifest.expedition_id for frame in self.frames
        )
        marker = Path(source) / "authenticated-request-sha256"
        request_sha256 = (
            None
            if not marker.is_file()
            else ContentDigest("sha256", marker.read_text(encoding="ascii"))
        )
        # The initial synthetic frontier intentionally contains all three
        # observations.  Terminal settlement projects one observation per
        # child occurrence; select the corresponding deterministic frame so
        # the authenticated frontier remains a set rather than duplicating a
        # frame three times.
        source_path = Path(source)
        if source_path.name == "synthetic-canonical-occurrence":
            source_key = str(source_path.absolute())
            frame = self._terminal_source_frames.get(source_key)
            if frame is None:
                frame = _frame_for_state(
                    manifest,
                    _synthetic_state(manifest, 100 + self._next_terminal_frame),
                )
                self._next_terminal_frame += 1
                self._terminal_source_frames[source_key] = frame
            frames = (frame,)
        else:
            frames = self.frames
        return _Projection(str(source), frames, request_sha256)


class _Runner:
    def __init__(self):
        self.calls = []

    def __call__(self, continuation, **options):
        parent = Path(options["runtime_parent"])
        resume = options["resume_root"]
        if resume is None:
            parent.mkdir(parents=True, exist_ok=True)
            occurrence = parent / "synthetic-canonical-occurrence"
            occurrence.mkdir()
        else:
            occurrence = Path(resume)
        marker = occurrence / "authenticated-request-sha256"
        if marker.exists():
            assert (
                marker.read_text(encoding="ascii")
                == continuation.child_request_sha256.hex
            )
        else:
            marker.write_text(continuation.child_request_sha256.hex, encoding="ascii")
        self.calls.append(
            (continuation.action_id, continuation.to_dict(), dict(options))
        )
        return {"runtime_root": str(occurrence), "status": "COMPLETED", "ok": True}


class _WrongAuthenticatedRunner(_Runner):
    def __call__(self, continuation, **options):
        result = super().__call__(continuation, **options)
        marker = Path(result["runtime_root"]) / "authenticated-request-sha256"
        marker.write_text(canonical_sha256({"wrong": "child"}).hex, encoding="ascii")
        return result


class _CrashOnce:
    def __init__(self, stage):
        self.stage = stage
        self.triggered = False

    def __call__(self, stage, _frontier, _action):
        if stage == self.stage and not self.triggered:
            self.triggered = True
            raise RuntimeError(f"crash at {stage}")


def _synthetic_exact_builder(*, variant="stable", **kwargs):
    state = kwargs["parent_state_ref"]
    decision = kwargs["decision"]
    action_id = kwargs["action_id"]
    row = next(row for row in decision.action_rows if row.action_id == action_id)
    identity = canonical_sha256(
        {
            "variant": variant,
            "action_id": action_id,
            "parent": state.state_ref_sha256().to_dict(),
        }
    ).hex
    payload = CanonicalPayloadV1.from_value(
        SchemaVersion("eiel_synthetic_exact_child_request_v1"),
        {"action_id": action_id, "variant": variant},
    )
    return GeneratedChildContinuationV1(
        expedition_id=kwargs["manifest"].expedition_id,
        manifest_sha256=kwargs["manifest"].manifest_sha256(),
        decision_sha256=decision.decision_sha256(),
        action_id=action_id,
        action_kind=row.action_kind,
        realization_kind=ChildRealizationKindV1.EXACT_STATE_REQUEST,
        parent_state_ref=state,
        founder_basis_sha256=kwargs["manifest"].founder_basis_sha256(),
        child_trajectory_id=f"synthetic-child-trajectory-{identity[:24]}",
        child_run_id=RunId(f"synthetic-child-run-{identity[:24]}"),
        retry_of_run_id=None,
        effective_next_absolute_tick=state.next_absolute_tick,
        control_schedule_sha256=canonical_sha256(
            {"branch_tick": state.next_absolute_tick, "variant": variant}
        ),
        child_request=payload,
        child_request_sha256=payload.content_digest(),
    )


def _loop_inputs(tmp_path: Path, *, scenario="all_actions", runner=None):
    policy = _ScriptedPolicy(scenario)
    founder = _founder_basis_request()
    manifest = _general_manifest(policy, founder)
    frames = tuple(
        _frame_for_state(manifest, _synthetic_state(manifest, ordinal))
        for ordinal in range(3)
    )
    selected_runner = runner or _Runner()
    return {
        "manifest": manifest,
        "policy": policy,
        "founder_request": founder,
        "ledger_root": tmp_path / "ledger",
        "atlas": _atlas(manifest),
        "retained_parent_bank": _bank(manifest),
        "logical_capacity": 5,
        "maximum_frontiers": 1,
        "initial_observation_sources": [tmp_path / "authenticated-observation"],
        "runtime_parent": tmp_path / "runtime",
        "dependencies": GeneralClosedLoopDependenciesV1(
            observation_projector=_Projector(frames),
            exact_child_builder=_synthetic_exact_builder,
            child_runner=selected_runner,
        ),
    }


def _workhorse_parent(
    tmp_path,
    monkeypatch,
    *,
    expedition_id,
    world_trajectory_id="trajectory-workhorse-parent-v1",
    destination=None,
):
    metadata, checkpoint_manifest, state_path = _physical_checkpoint(
        tmp_path / "physical-parent",
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
        tmp_path / "indexed-parent",
    )
    parent_digest = canonical_sha256({"occurrence": "prior-parent"})
    occurrence = _occurrence(tick=25, parent_digest=parent_digest)
    occurrence_root = tmp_path / "occurrence"
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
    core = {
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
    projection = {**core, "projection_content_sha256": canonical_sha256(core).hex}
    return branch.materialize_workhorse_selected_world_state_ref_v1(
        indexed["manifest_path"],
        destination or tmp_path / "retained-parent",
        world_uid=9102,
        checkpoint_occurrence_projection=projection,
        expedition_id=expedition_id,
        world_trajectory_id=world_trajectory_id,
        semantic_world_sha256="6" * 64,
        control_history_sha256="7" * 64,
        rng_state_sha256="8" * 64,
        event_prefix_sha256="9" * 64,
        active_scientific_control_state=workhorse_active_scientific_control_state_v1(
            captured_next_absolute_tick=25,
            resource_multiplier=1.0,
            mutation_rate=0.125,
        ),
    )


def test_exact_child_binds_authority_controls_absolute_tick_and_repack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = _ScriptedPolicy("branch_only")
    founder = _founder_basis_request()
    manifest = _general_manifest(policy, founder)
    state, authority, resolved = _workhorse_parent(
        tmp_path, monkeypatch, expedition_id=manifest.expedition_id
    )
    frame = _frame_for_state(manifest, state)
    prepared = prepare_decision_v1(
        manifest=manifest,
        policy=policy,
        frontier_id="frontier-exact-0000",
        frontier_ordinal=0,
        observation_frames=[frame],
        bounded_history=[],
        atlas=_atlas(manifest),
        retained_parent_bank=_bank(manifest),
        logical_capacity=1,
        outer_rng_cursor=0,
    )
    action_id = prepared.decision.action_rows[0].action_id
    first = build_exact_state_child_continuation_v1(
        manifest=manifest,
        decision=prepared.decision,
        action_id=action_id,
        founder_request=founder,
        parent_state_ref=state,
    )
    second = build_exact_state_child_continuation_v1(
        manifest=manifest,
        decision=prepared.decision,
        action_id=action_id,
        founder_request=founder,
        parent_state_ref=state,
    )
    assert first.to_dict() == second.to_dict()
    assert first.parent_state_ref == state
    assert first.child_run_id != resolved.parent_run_id
    assert first.child_trajectory_id != state.world_trajectory_id
    assert first.retry_of_run_id is None
    assert first.effective_next_absolute_tick == 25
    request = validate_eiel_science_request_v1(first.child_request.to_value())
    raw = request.to_dict()
    assert raw["run"] == {
        "ticks": 30,
        "batch": {"policy": "exact", "worlds": 1},
        "checkpoint_ticks": [30],
    }
    assert raw["observe"]["snapshot_ticks"] == [25, 30]
    assert raw["observe"]["events"] == ["total_event_count"]
    assert raw["observe"]["state"] == ["genome_identity", "population_count"]
    assert raw["observe"]["runtime_counters"] == [
        "birth_count",
        "death_count",
        "mutation_count",
    ]
    assert raw["continuation"]["state_ref"] == state.to_dict()
    assert raw["continuation"]["authority_receipt_ref"] == authority.to_dict()
    assert {
        value["schedule"]["onset_tick"]
        for value in raw["world_template"]["controls"].values()
    } == {25}

    compilation = compile_science_request_v1(request)
    requested = set()
    for world in compilation.world_bundle.worlds:
        acquisition = world.acquisition.to_value()
        requested.update(acquisition["event_observations"])
        requested.update(acquisition["state_observations"])
        requested.update(acquisition["runtime_counters"])
        for reducer in world.reducers.to_value():
            requested.update(reducer["required_observations"])
    admission = admit_science_world_bundle_v1(
        compilation.world_bundle,
        supported_observations=tuple(sorted(requested)),
        supported_profiles=("SCIENCE_STANDARD_V1",),
        maximum_total_ticks=30,
    )
    plan = carrier.plan_science_cuda_v2_execution_v1(
        compilation.world_bundle,
        admission,
        physical_batch_capacity=1,
        temporal_k=2,
    )
    assert plan.batches[0].world_slots[0][0] == 0
    assert plan.batches[0].world_slots[0][1].world_uid == 9102
    assert resolved.physical_cell_index == 1


def test_exact_child_fails_closed_when_neighbor_authority_is_tampered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = _ScriptedPolicy("branch_only")
    founder = _founder_basis_request()
    manifest = _general_manifest(policy, founder)
    state, authority, _resolved = _workhorse_parent(
        tmp_path, monkeypatch, expedition_id=manifest.expedition_id
    )
    frame = _frame_for_state(manifest, state)
    prepared = prepare_decision_v1(
        manifest=manifest,
        policy=policy,
        frontier_id="frontier-tamper-0000",
        frontier_ordinal=0,
        observation_frames=[frame],
        bounded_history=[],
        atlas=_atlas(manifest),
        retained_parent_bank=_bank(manifest),
        logical_capacity=1,
        outer_rng_cursor=0,
    )
    authority_path = Path(authority.locator)
    authority_path.chmod(0o600)
    authority_path.write_bytes(authority_path.read_bytes() + b"\n")
    with pytest.raises(GeneralClosedLoopError, match="cannot reopen exactly"):
        build_exact_state_child_continuation_v1(
            manifest=manifest,
            decision=prepared.decision,
            action_id=prepared.decision.action_rows[0].action_id,
            founder_request=founder,
            parent_state_ref=state,
        )


def test_general_loop_executes_three_paths_and_books_park_retire_without_dispatch(
    tmp_path: Path,
) -> None:
    runner = _Runner()
    result = run_general_closed_loop_v1(**_loop_inputs(tmp_path, runner=runner))
    assert result.terminal is True
    assert len(result.children) == 3
    assert len(runner.calls) == 3
    assert {child.action_kind.value for child in result.children} == {
        "EXPLORE_GLOBAL",
        "BRANCH_LOCAL",
        "REPLICATE",
    }
    assert sum(child.parent_state_ref is None for child in result.children) == 1
    assert all(child.retry_of_run_id is None for child in result.children)

    decision = next(
        event["payload"]
        for event in result.ledger["events"]
        if event["event_type"] == DECISION_PREPARED
    )
    assert {row["action_kind"] for row in decision["bookkeeping_actions"]} == {
        "PARK",
        "RETIRE",
    }
    assert all(
        row["outcome"] == "BOOKKEEPING_ONLY_NO_EXECUTION"
        for row in decision["bookkeeping_actions"]
    )
    assert decision["no_hidden_reward_authority"] is True
    bound_ids = {
        event["payload"]["action_id"]
        for event in result.ledger["events"]
        if event["event_type"] == ACTION_DISPATCH_BOUND
    }
    assert not any(
        row["action_id"] in bound_ids for row in decision["bookkeeping_actions"]
    )
    assert result.atlas.revision == 2
    assert len(result.atlas.entries) == 2
    assert result.retained_parent_bank.revision == 2
    assert result.retained_parent_bank.current_entry_count == 4
    assert {entry.reason_codes[0] for entry in result.retained_parent_bank.entries} == {
        "DIVERGENT_LINEAGE_STEPPING_STONE",
        "DIVERGENT_STRUCTURE_STEPPING_STONE",
    }
    assert (
        decision["retention_outcomes"]["policy_not_retained"][0]["reason_code"]
        == "BOUNDED_BANK_BUDGET_LOWER_PRIORITY"
    )
    assert result.ledger["events"][-1]["payload"]["programme_stage"] == (
        "G4_BOUNDED_TRUE_BRANCHING"
    )
    assert (
        result.ledger["events"][-1]["payload"]["historical_continuation_authority"]
        is True
    )


def test_general_loop_rejects_an_authenticated_exact_result_for_another_child(
    tmp_path: Path,
) -> None:
    inputs = _loop_inputs(
        tmp_path,
        scenario="branch_only",
        runner=_WrongAuthenticatedRunner(),
    )
    with pytest.raises(
        GeneralClosedLoopError,
        match="names a different Science Request",
    ):
        run_general_closed_loop_v1(**inputs)
    ledger = read_ledger_store_v1(inputs["ledger_root"])
    assert not any(event["event_type"] == ACTION_SETTLED for event in ledger["events"])


def test_crash_replay_never_redispatches_settled_actions(tmp_path: Path) -> None:
    runner = _Runner()
    inputs = _loop_inputs(tmp_path, runner=runner)
    crash = _CrashOnce("AFTER_ACTION_SETTLED")
    with pytest.raises(RuntimeError, match="crash at AFTER_ACTION_SETTLED"):
        run_general_closed_loop_v1(**inputs, interruption_hook=crash)
    partial = read_ledger_store_v1(inputs["ledger_root"])
    settled = next(
        event["payload"]["action_id"]
        for event in partial["events"]
        if event["event_type"] == ACTION_SETTLED
    )
    result = run_general_closed_loop_v1(**inputs)
    assert result.terminal is True
    assert [action_id for action_id, *_ in runner.calls].count(settled) == 1
    assert len(runner.calls) == 3
    calls_before_terminal_replay = deepcopy(runner.calls)
    replay = run_general_closed_loop_v1(**inputs)
    assert replay.ledger == result.ledger
    assert runner.calls == calls_before_terminal_replay


def test_recovery_rejects_changed_child_realization_before_dispatch(
    tmp_path: Path,
) -> None:
    inputs = _loop_inputs(tmp_path)
    crash = _CrashOnce("AFTER_DECISION_PREPARED")
    with pytest.raises(RuntimeError, match="crash at AFTER_DECISION_PREPARED"):
        run_general_closed_loop_v1(**inputs, interruption_hook=crash)
    changed_dependencies = GeneralClosedLoopDependenciesV1(
        observation_projector=inputs["dependencies"].observation_projector,
        exact_child_builder=lambda **kwargs: _synthetic_exact_builder(
            variant="changed-after-crash", **kwargs
        ),
        child_runner=inputs["dependencies"].child_runner,
    )
    with pytest.raises(ValueError, match="replayed decision differs"):
        run_general_closed_loop_v1(**{**inputs, "dependencies": changed_dependencies})


def test_historical_replicate_resolves_parent_bank_state_absent_from_frontier(
    tmp_path: Path,
) -> None:
    policy = _ScriptedPolicy("historical_replicate")
    founder = _founder_basis_request()
    manifest = _general_manifest(policy, founder)
    historical = _synthetic_state(manifest, 10)
    current = _frame_for_state(manifest, _synthetic_state(manifest, 11))
    bank = RetainedParentBankV1.from_dict(
        {
            "schema_version": "eiel_adaptive_science_retained_parent_bank_v1",
            "expedition_id": manifest.expedition_id,
            "manifest_sha256": manifest.manifest_sha256().to_dict(),
            "revision": 0,
            "maximum_entries": manifest.maximum_retained_parent_entries,
            "maximum_total_bytes": manifest.maximum_retained_parent_bytes,
            "current_entry_count": 1,
            "current_total_bytes": historical.source_artifact_ref.size_bytes,
            "entries": [
                {
                    "state_ref": historical.to_dict(),
                    "selected_at_frontier_ordinal": 0,
                    "reason_codes": ["HISTORICAL_STEPPING_STONE"],
                    "retained_bytes": historical.source_artifact_ref.size_bytes,
                }
            ],
            "previous_bank_sha256": None,
        }
    )
    runner = _Runner()
    result = run_general_closed_loop_v1(
        manifest=manifest,
        policy=policy,
        founder_request=founder,
        ledger_root=tmp_path / "ledger",
        atlas=_atlas(manifest),
        retained_parent_bank=bank,
        logical_capacity=1,
        maximum_frontiers=1,
        initial_observation_sources=[tmp_path / "current-observation"],
        runtime_parent=tmp_path / "runtime",
        dependencies=GeneralClosedLoopDependenciesV1(
            observation_projector=_Projector((current,)),
            exact_child_builder=_synthetic_exact_builder,
            child_runner=runner,
        ),
    )
    assert len(result.children) == 1
    assert result.children[0].action_kind.value == "REPLICATE"
    assert result.children[0].parent_state_ref == historical
    assert result.children[0].parent_state_ref != current.source_state_ref
