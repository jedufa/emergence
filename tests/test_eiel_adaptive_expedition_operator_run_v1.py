from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from emergence_contracts import canonical_json_bytes
from emergence_contracts.adaptive_science_expedition_v1 import (
    ExpeditionManifestV1,
    ObservationFrameV1,
)
from eiel_composition.adaptive_expedition import cli as expedition_cli_module
from eiel_composition.adaptive_expedition.cli import main as expedition_main
from eiel_composition.adaptive_expedition.founder_closed_loop import (
    FounderClosedLoopDependenciesV1,
    run_founder_closed_loop_v1,
)
from eiel_composition.adaptive_expedition.operator_run import (
    ExpeditionOperatorDependenciesV1,
    ExpeditionOperatorError,
    EXPEDITION_OPERATOR_EXECUTION_MODE_V1,
    EXPEDITION_OPERATOR_PROGRAMME_STAGE_V1,
    _materialize_initial_exact_observation_sources_v1,
    _read_frontier_timing_records_v1,
    resolve_founder_request_v1,
    run_expedition_operator_v1,
)
from eiel_composition.adaptive_expedition.lifecycle import (
    _preflight_expedition_outer_object_budget_v1,
)
from eiel_composition.adaptive_expedition.policies.adaptive_illumination_v1 import (
    AdaptiveIlluminationPolicyV1,
)
from eiel_composition.adaptive_expedition.policy_protocol import (
    canonical_plain_value_v1,
)
from tests.test_eiel_adaptive_expedition_general_closed_loop_v1 import (
    _frame_for_state,
    _workhorse_parent,
)
from tests.test_eiel_adaptive_expedition_founder_closed_loop_v1 import (
    _CrashOnce,
    _Projector,
    _Runner,
)
from tests.test_eiel_adaptive_expedition_founder_loop_v1 import (
    _founder_policy,
    _prepared,
)


def _write(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value))


def _operator_manifest(tmp_path: Path) -> tuple[ExpeditionManifestV1, Path]:
    manifest, founder_request, _prepared_decision = _prepared()
    founder_path = tmp_path / "founder-request.json"
    _write(founder_path, founder_request.to_dict())
    raw = manifest.to_dict()
    raw["founder_basis_refs"] = [
        {
            "schema_version": "run_artifact_ref_v1",
            "artifact_id": "founder-science-request-v1",
            "artifact_kind": "eiel_science_request_v1",
            "artifact_schema_version": "eiel_science_request_v1",
            "locator": founder_path.resolve().as_uri(),
            "content_digest": founder_request.request_sha256().to_dict(),
            "size_bytes": len(canonical_json_bytes(founder_request.to_dict())),
        }
    ]
    raw["wall_time_budget_seconds"] = 10
    raw["maximum_frontiers"] = 4
    raw["maximum_actions_per_frontier"] = 1
    return ExpeditionManifestV1.from_dict(raw), founder_path


def test_founder_reference_is_content_bound_and_tamper_fails_closed(
    tmp_path: Path,
) -> None:
    manifest, founder_path = _operator_manifest(tmp_path)
    request, resolved = resolve_founder_request_v1(manifest)
    assert request.request_sha256() == manifest.founder_basis_refs[0].content_digest
    assert resolved == founder_path.resolve()

    changed = json.loads(founder_path.read_bytes())
    changed["question"]["description"] = "changed after manifest freeze"
    _write(founder_path, changed)
    with pytest.raises(ExpeditionOperatorError, match="digest differs"):
        resolve_founder_request_v1(manifest)


def test_manifest_bound_exact_initial_frame_is_retained_and_reopens_after_source_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _founder_path = _operator_manifest(tmp_path)
    exact_parent_root = tmp_path / "exact-parent"
    state, _authority, _resolved = _workhorse_parent(
        exact_parent_root,
        monkeypatch,
        expedition_id=manifest.expedition_id,
    )
    frame = _frame_for_state(manifest, state)
    external = tmp_path / "initial-exact-observation-frame.json"
    _write(external, frame.to_dict())
    raw = manifest.to_dict()
    raw["founder_basis_refs"].append(
        {
            "schema_version": "run_artifact_ref_v1",
            "artifact_id": "initial-exact-parent-v1",
            "artifact_kind": "eiel_adaptive_science_observation_frame_v1",
            "artifact_schema_version": ("eiel_adaptive_science_observation_frame_v1"),
            "locator": external.resolve().as_uri(),
            "content_digest": frame.observation_frame_sha256().to_dict(),
            "size_bytes": len(canonical_json_bytes(frame.to_dict())),
        }
    )
    bound = ExpeditionManifestV1.from_dict(raw)
    control = tmp_path / "operator-control"
    first = _materialize_initial_exact_observation_sources_v1(
        manifest=bound,
        control_root=control,
    )
    assert len(first) == 1
    retained = ObservationFrameV1.from_dict(json.loads(Path(first[0]).read_bytes()))
    assert retained.source_state_ref.state_ref_sha256() != state.state_ref_sha256()
    assert retained.source_state_ref.exact_state_sha256 == state.exact_state_sha256
    assert retained.source_state_ref.checkpoint_occurrence_sha256 == (
        state.checkpoint_occurrence_sha256
    )
    package_root = Path(retained.source_state_ref.source_artifact_ref.locator).parent
    assert package_root.is_relative_to(control.parent / "durable" / "founder-basis")
    assert (package_root / "occurrence" / "descriptor.json").is_file()
    assert (package_root / "occurrence" / "commit.json").is_file()

    external.unlink()
    for directory in (
        exact_parent_root,
        *(path for path in exact_parent_root.rglob("*") if path.is_dir()),
    ):
        directory.chmod(0o755)
    shutil.rmtree(exact_parent_root)
    assert (
        _materialize_initial_exact_observation_sources_v1(
            manifest=bound,
            control_root=control,
        )
        == first
    )

    Path(first[0]).chmod(0o644)
    Path(first[0]).write_text(json.dumps(frame.to_dict(), indent=2), encoding="utf-8")
    with pytest.raises(ExpeditionOperatorError, match="manifest binding"):
        _materialize_initial_exact_observation_sources_v1(
            manifest=bound,
            control_root=control,
        )


def test_single_manifest_run_stops_at_safe_frontier_and_resumes_identically(
    tmp_path: Path,
) -> None:
    manifest, founder_path = _operator_manifest(tmp_path)
    policy = _founder_policy()
    runner = _Runner()
    started = datetime(2026, 8, 29, 0, 0, tzinfo=timezone.utc)
    observations = iter(
        [
            started,
            started,
            started + timedelta(seconds=5),
            started + timedelta(seconds=11),
        ]
    )
    dependencies = ExpeditionOperatorDependenciesV1(
        clock=lambda: next(observations),
        closed_loop_runner=run_founder_closed_loop_v1,
        closed_loop_dependencies=FounderClosedLoopDependenciesV1(
            observation_projector=_Projector(),
            child_runner=runner,
        ),
        execution_mode="FOUNDER_ONLY_G2_5",
        programme_stage="G2.5_FOUNDER_ONLY",
        historical_continuation_authority=False,
    )

    first = run_expedition_operator_v1(
        manifest=manifest,
        policy=policy,
        runtime_parent=tmp_path / "runtime",
        repo_root=tmp_path,
        dependencies=dependencies,
    )
    receipt = first.receipt()
    assert receipt["terminal_status"] == "STOPPED"
    # The settled frontier is followed by a terminal, non-dispatching
    # finalization frontier before safe-stop closeout.
    assert receipt["frontier_count"] == 2
    assert receipt["child_count"] == 1
    assert receipt["qualification_status"] == "NOT_REQUESTED"
    assert len(runner.calls) == 1
    assert first.start_record["deadline_at_utc"] == "2026-08-29T00:00:10.000000Z"
    assert first.atlas_path.is_file()
    assert first.retained_parent_bank_path.is_file()
    assert first.journal_path.is_file()
    assert first.human_review_path.is_file()
    assert first.closeout_path.is_file()
    assert Path(
        first.lifecycle_closeout["artifact_disposition_manifest_path"]
    ).is_file()
    assert Path(first.lifecycle_closeout["storage_disposition_receipt_path"]).is_file()
    review = json.loads(first.human_review_path.read_bytes())
    assert len(review["review"]) == 13
    assert (
        review["review"]["retained_parents"]
        != review["review"]["response_or_frontier_regions"]
    )
    assert review["review"]["unresolved_scientific_questions"]["rows"]
    assert review["review"]["discarded_parent_candidates"]["status_code"] == (
        "NO_PARENT_DISCARD_REASON_RECORDED"
    )
    control_rows = [
        row
        for row in review["review"]["exploration_coverage"]["rows"]
        if row["row_kind"] == "OPAQUE_CONTROL_COORDINATE_COVERAGE_INDEX"
    ]
    assert {row["control_coordinate"] for row in control_rows} == {
        "environment.resource.multiplier",
        "mutation.rate",
    }
    assert all(row["declared_in_authorized_search_space"] for row in control_rows)
    assert all(
        len(value) == 64
        for row in control_rows
        for value in row["selected_value_sha256_samples"]
    )
    assert review["qualification_status"] == "NOT_REQUESTED"
    timing_records = sorted(
        (first.runtime_root / "control" / "frontier-dispatch-timing").glob("*.json")
    )
    assert [path.name for path in timing_records] == ["00000000.json"]
    stop = first.loop_result.ledger["events"][-2]["payload"]
    assert stop["reason"] == "WALL_TIME_SAFE_FRONTIER_RESERVE_REACHED"
    assert stop["next_frontier_ordinal"] == 1
    assert stop["reserve_seconds_before_dispatch"] == 10

    # A terminal resume reads the original start/deadline, replay-validates the
    # ledger, and accepts only byte-identical closeout files. It does not call
    # the clock, policy runner, or child runner again.
    founder_path.unlink()
    frozen_review = first.human_review_path.read_bytes()
    second = run_expedition_operator_v1(
        manifest=manifest,
        policy=policy,
        runtime_parent=tmp_path / "runtime",
        repo_root=tmp_path,
        dependencies=dependencies,
    )
    assert second.receipt() == receipt
    assert second.loop_result.ledger == first.loop_result.ledger
    assert second.human_review_path.read_bytes() == frozen_review
    assert len(runner.calls) == 1


def test_expired_recovery_budget_stops_before_first_frontier_dispatch(
    tmp_path: Path,
) -> None:
    manifest, _founder_path = _operator_manifest(tmp_path)
    policy = _founder_policy()
    runner = _Runner()
    started = datetime(2026, 8, 29, 0, 0, tzinfo=timezone.utc)
    observations = iter([started, started + timedelta(seconds=11)])
    result = run_expedition_operator_v1(
        manifest=manifest,
        policy=policy,
        runtime_parent=tmp_path / "runtime-expired",
        repo_root=tmp_path,
        dependencies=ExpeditionOperatorDependenciesV1(
            clock=lambda: next(observations),
            closed_loop_runner=run_founder_closed_loop_v1,
            closed_loop_dependencies=FounderClosedLoopDependenciesV1(
                observation_projector=_Projector(),
                child_runner=runner,
            ),
            execution_mode="FOUNDER_ONLY_G2_5",
            programme_stage="G2.5_FOUNDER_ONLY",
            historical_continuation_authority=False,
        ),
    )

    assert result.receipt()["terminal_status"] == "STOPPED"
    assert result.receipt()["frontier_count"] == 0
    assert result.receipt()["child_count"] == 0
    assert runner.calls == []
    events = result.loop_result.ledger["events"]
    assert [event["event_type"] for event in events] == [
        "STOP_REQUESTED",
        "EXPEDITION_TERMINAL",
    ]
    assert events[0]["payload"]["before_dispatch"] is True


def test_timing_crash_debris_is_strictly_recovered_before_resume(
    tmp_path: Path,
) -> None:
    manifest, _founder_path = _operator_manifest(tmp_path)
    policy = _founder_policy()
    runner = _Runner()
    crash = _CrashOnce("AFTER_FRONTIER_OBSERVED")
    started = datetime(2026, 8, 29, 0, 0, tzinfo=timezone.utc)
    first_clock = iter([started, started])
    runtime_parent = tmp_path / "runtime-crash-debris"

    def crash_after_timing(**options):
        return run_founder_closed_loop_v1(
            **options,
            interruption_hook=crash,
        )

    with pytest.raises(RuntimeError, match="AFTER_FRONTIER_OBSERVED"):
        run_expedition_operator_v1(
            manifest=manifest,
            policy=policy,
            runtime_parent=runtime_parent,
            repo_root=tmp_path,
            dependencies=ExpeditionOperatorDependenciesV1(
                clock=lambda: next(first_clock),
                closed_loop_runner=crash_after_timing,
                closed_loop_dependencies=FounderClosedLoopDependenciesV1(
                    observation_projector=_Projector(),
                    child_runner=runner,
                ),
                execution_mode="FOUNDER_ONLY_G2_5",
                programme_stage="G2.5_FOUNDER_ONLY",
                historical_continuation_authority=False,
            ),
        )

    runtime_root = next(
        path
        for path in runtime_parent.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    timing_root = runtime_root / "control" / "frontier-dispatch-timing"
    canonical = timing_root / "00000000.json"
    linked_debris = timing_root / ".00000000.json.deadbeef"
    partial_debris = timing_root / ".00000001.json.cafebabe"
    os.link(canonical, linked_debris)
    partial_debris.write_bytes(b'{"partial":')
    partial_debris.chmod(0o600)

    recovery_clock = iter(
        [started + timedelta(seconds=1), started + timedelta(seconds=6)]
    )
    recovered = run_expedition_operator_v1(
        manifest=manifest,
        policy=policy,
        runtime_parent=runtime_parent,
        repo_root=tmp_path,
        dependencies=ExpeditionOperatorDependenciesV1(
            clock=lambda: next(recovery_clock),
            closed_loop_runner=run_founder_closed_loop_v1,
            closed_loop_dependencies=FounderClosedLoopDependenciesV1(
                observation_projector=_Projector(),
                child_runner=runner,
            ),
            execution_mode="FOUNDER_ONLY_G2_5",
            programme_stage="G2.5_FOUNDER_ONLY",
            historical_continuation_authority=False,
        ),
    )
    assert recovered.receipt()["terminal_status"] == "STOPPED"
    assert not linked_debris.exists()
    assert not partial_debris.exists()
    assert [path.name for path in timing_root.iterdir()] == ["00000000.json"]

    disposition = json.loads(
        Path(
            recovered.lifecycle_closeout["artifact_disposition_manifest_path"]
        ).read_bytes()
    )
    inventory_paths = {row["path"] for row in disposition["artifacts"]}
    assert {"closeout", "control", "durable/ledger"} <= inventory_paths
    assert not any(path.startswith("control/") for path in inventory_paths)
    assert not any(path.startswith("durable/ledger/") for path in inventory_paths)


def test_recovery_rejects_operator_clock_rollback(
    tmp_path: Path,
) -> None:
    manifest, _founder_path = _operator_manifest(tmp_path)
    policy = _founder_policy()
    runner = _Runner()
    crash = _CrashOnce("AFTER_FRONTIER_OBSERVED")
    started = datetime(2026, 8, 29, 0, 0, tzinfo=timezone.utc)
    first_clock = iter([started, started])
    runtime_parent = tmp_path / "runtime-clock-rollback"

    def crash_after_timing(**options):
        return run_founder_closed_loop_v1(
            **options,
            interruption_hook=crash,
        )

    with pytest.raises(RuntimeError, match="AFTER_FRONTIER_OBSERVED"):
        run_expedition_operator_v1(
            manifest=manifest,
            policy=policy,
            runtime_parent=runtime_parent,
            repo_root=tmp_path,
            dependencies=ExpeditionOperatorDependenciesV1(
                clock=lambda: next(first_clock),
                closed_loop_runner=crash_after_timing,
                closed_loop_dependencies=FounderClosedLoopDependenciesV1(
                    observation_projector=_Projector(),
                    child_runner=runner,
                ),
                execution_mode="FOUNDER_ONLY_G2_5",
                programme_stage="G2.5_FOUNDER_ONLY",
                historical_continuation_authority=False,
            ),
        )

    with pytest.raises(ExpeditionOperatorError, match="clock moved backwards"):
        run_expedition_operator_v1(
            manifest=manifest,
            policy=policy,
            runtime_parent=runtime_parent,
            repo_root=tmp_path,
            dependencies=ExpeditionOperatorDependenciesV1(
                clock=lambda: started - timedelta(microseconds=1),
                closed_loop_runner=run_founder_closed_loop_v1,
                closed_loop_dependencies=FounderClosedLoopDependenciesV1(
                    observation_projector=_Projector(),
                    child_runner=runner,
                ),
                execution_mode="FOUNDER_ONLY_G2_5",
                programme_stage="G2.5_FOUNDER_ONLY",
                historical_continuation_authority=False,
            ),
        )
    assert runner.calls == []


def test_timing_recovery_preserves_non_private_debris_fail_closed(
    tmp_path: Path,
) -> None:
    timing_root = tmp_path / "unsafe-timing-debris"
    timing_root.mkdir()
    debris = timing_root / ".00000000.json.deadbeef"
    debris.write_bytes(b"partial")
    debris.chmod(0o644)
    start_record = {
        "started_at_utc": "2026-08-29T00:00:00.000000Z",
        "deadline_at_utc": "2026-08-29T12:00:00.000000Z",
        "manifest_sha256": "1" * 64,
    }

    with pytest.raises(ExpeditionOperatorError, match="strict private"):
        _read_frontier_timing_records_v1(
            timing_root,
            start_record=start_record,
        )
    assert debris.is_file()


def test_outer_object_preflight_accepts_bounded_12h_shape_and_fails_before_runtime(
    tmp_path: Path,
) -> None:
    manifest, _founder_path = _operator_manifest(tmp_path)
    bounded_raw = manifest.to_dict()
    bounded_raw["maximum_frontiers"] = 128
    bounded_raw["maximum_actions_per_frontier"] = 16
    bounded_raw["wall_time_budget_seconds"] = 12 * 60 * 60
    bounded = ExpeditionManifestV1.from_dict(bounded_raw)
    estimate = _preflight_expedition_outer_object_budget_v1(bounded)
    assert estimate["maximum_outer_objects"] < estimate["outer_object_cap"]

    oversized_raw = bounded.to_dict()
    oversized_raw["maximum_actions_per_frontier"] = 32
    oversized = ExpeditionManifestV1.from_dict(oversized_raw)
    runtime_parent = tmp_path / "runtime-object-overflow"
    with pytest.raises(
        ExpeditionOperatorError, match="fixed outer lifecycle object cap"
    ):
        run_expedition_operator_v1(
            manifest=oversized,
            policy=_founder_policy(),
            runtime_parent=runtime_parent,
            repo_root=tmp_path,
        )
    assert not runtime_parent.exists()


def test_operator_rejects_incomplete_policy_presentation_before_opening_runtime(
    tmp_path: Path,
) -> None:
    manifest, _founder_path = _operator_manifest(tmp_path)
    profile = canonical_plain_value_v1(_founder_policy().profile)
    profile["profile_version"] = "invalid-human-review-presentation"
    del profile["human_review_projection"]["empty_status_by_section"][
        "failed_replication_or_causal_followups"
    ]
    policy = AdaptiveIlluminationPolicyV1(profile)
    raw = manifest.to_dict()
    raw["policy_binding"] = canonical_plain_value_v1(policy.binding)
    bound = ExpeditionManifestV1.from_dict(raw)
    runtime_parent = tmp_path / "runtime-invalid-presentation"

    with pytest.raises(
        ExpeditionOperatorError,
        match="human-review presentation is invalid",
    ):
        run_expedition_operator_v1(
            manifest=bound,
            policy=policy,
            runtime_parent=runtime_parent,
            repo_root=tmp_path,
        )

    assert not runtime_parent.exists()


def test_operator_recursively_rejects_policy_presentation_escalation_before_runtime(
    tmp_path: Path,
) -> None:
    manifest, _founder_path = _operator_manifest(tmp_path)
    profile = canonical_plain_value_v1(_founder_policy().profile)
    profile["profile_version"] = "invalid-nested-human-review-presentation"
    profile["human_review_projection"]["human_scientific_choices"][0]["metadata"] = {
        "qualification_authority": "FORBIDDEN"
    }
    policy = AdaptiveIlluminationPolicyV1(profile)
    raw = manifest.to_dict()
    raw["policy_binding"] = canonical_plain_value_v1(policy.binding)
    bound = ExpeditionManifestV1.from_dict(raw)
    runtime_parent = tmp_path / "runtime-policy-presentation-escalation"

    with pytest.raises(
        ExpeditionOperatorError,
        match="human-review presentation is invalid",
    ):
        run_expedition_operator_v1(
            manifest=bound,
            policy=policy,
            runtime_parent=runtime_parent,
            repo_root=tmp_path,
        )
    assert not runtime_parent.exists()


def test_cli_normal_run_surface_consumes_only_the_manifest(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    manifest, _founder_path = _operator_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    _write(manifest_path, manifest.to_dict())
    captured = []

    def fake_run(**options):
        captured.append(options)
        return SimpleNamespace(
            receipt=lambda: {
                "status": "EXPEDITION_TERMINAL",
                "expedition_id": options["manifest"].expedition_id,
                "manifest_sha256": options["manifest"].manifest_sha256().hex,
                "terminal_status": "COMPLETED",
                "qualification_status": "NOT_REQUESTED",
                "no_claims": True,
            }
        )

    monkeypatch.setattr(
        expedition_cli_module,
        "run_expedition_operator_v1",
        fake_run,
    )
    assert expedition_main(["run", str(manifest_path), "--human"]) == 0
    assert len(captured) == 1
    assert captured[0]["manifest"].manifest_sha256() == manifest.manifest_sha256()
    assert captured[0]["policy"].binding == _founder_policy().binding
    output = capsys.readouterr().out
    assert "status: EXPEDITION_TERMINAL" in output
    assert "terminal_status: COMPLETED" in output


def test_plain_operator_defaults_remain_generic_and_non_historical() -> None:
    dependencies = ExpeditionOperatorDependenciesV1()
    assert dependencies.execution_mode == EXPEDITION_OPERATOR_EXECUTION_MODE_V1
    assert dependencies.programme_stage == EXPEDITION_OPERATOR_PROGRAMME_STAGE_V1
    assert dependencies.historical_continuation_authority is False
