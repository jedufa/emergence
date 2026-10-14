from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pytest

from emergence_contracts.adaptive_science_expedition_v1 import ObservationFrameV1
from emergence_contracts.canonical import canonical_sha256
from emergence_contracts.ids import ContentDigest
from eiel_composition.adaptive_expedition.founder_closed_loop import (
    FounderClosedLoopDependenciesV1,
    FounderClosedLoopError,
    _resume_root_v1,
    run_founder_closed_loop_v1,
)
from eiel_composition.adaptive_expedition.ledger import (
    ACTION_DISPATCH_BOUND,
    ACTION_SETTLED,
    DECISION_PREPARED,
    EXPEDITION_TERMINAL,
    FRONTIER_OBSERVED,
    STOP_REQUESTED,
    read_ledger_store_v1,
)
from tests.test_eiel_adaptive_expedition_coordinator_v1 import _atlas, _bank, _frame
from tests.test_eiel_adaptive_expedition_founder_loop_v1 import (
    _founder_policy,
    _prepared,
)


@dataclass(frozen=True, slots=True)
class _Projection:
    source: str
    frames: tuple[ObservationFrameV1, ...]
    request_sha256: ContentDigest | None = None

    def projection_sha256(self):
        return canonical_sha256(
            {
                "adapter": "authenticated-test-double-v1",
                "source": self.source,
                "request_sha256": (
                    None
                    if self.request_sha256 is None
                    else self.request_sha256.to_dict()
                ),
                "frame_sha256s": [
                    frame.observation_frame_sha256().to_dict() for frame in self.frames
                ],
            }
        )


class _Projector:
    def __call__(self, source, *, manifest):
        assert manifest.no_claims is True
        marker = Path(source) / "authenticated-request-sha256"
        request_sha256 = (
            None
            if not marker.is_file()
            else ContentDigest("sha256", marker.read_text(encoding="ascii"))
        )
        return _Projection(str(source), (_frame(),), request_sha256)


class _ObservationOnlyProjector:
    def __call__(self, source, *, manifest):
        value = _frame().to_dict()
        state = value["source_state_ref"]
        state["state_class"] = "OBSERVATION_ONLY"
        for field in (
            "exact_state_sha256",
            "checkpoint_occurrence_sha256",
            "rng_state_sha256",
            "event_prefix_sha256",
            "parent_occurrence_sha256",
            "replay_recipe_sha256",
        ):
            state[field] = None
        frame = ObservationFrameV1.from_dict(value)
        marker = Path(source) / "authenticated-request-sha256"
        request_sha256 = (
            None
            if not marker.is_file()
            else ContentDigest("sha256", marker.read_text(encoding="ascii"))
        )
        return _Projection(str(source), (frame,), request_sha256)


class _Runner:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, continuation, **options):
        parent = Path(options["runtime_parent"])
        resume = options["resume_root"]
        if resume is None:
            parent.mkdir(parents=True, exist_ok=True)
            occurrence = parent / "canonical-science-request-occurrence"
            occurrence.mkdir()
        else:
            occurrence = Path(resume)
            assert occurrence.is_dir()
        request_value = continuation.child_request.to_value()
        request_sha = canonical_sha256(request_value).to_dict()
        campaign_id = request_value["campaign_id"]
        passport_core = {
            "run_id": occurrence.name,
            "campaign_id": campaign_id,
            "terminal_status": "active",
            "source_identity": {"request_sha256": request_sha},
        }
        (occurrence / "run_passport.json").write_text(
            json.dumps(
                {
                    **passport_core,
                    "passport_sha256": canonical_sha256(passport_core).hex,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        journal_core = {
            "schema_version": "eiel_science_request_v1_run_journal_v1",
            "identity": {
                "campaign_id": campaign_id,
                "request_sha256": request_sha,
            },
            "status": "RUNNING",
            "batches": [],
            "aggregate": None,
            "last_event": "created",
        }
        (occurrence / "run_journal.json").write_text(
            json.dumps(
                {
                    **journal_core,
                    "journal_sha256": canonical_sha256(journal_core).hex,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        marker = occurrence / "authenticated-request-sha256"
        if marker.exists():
            assert (
                marker.read_text(encoding="ascii")
                == canonical_sha256(request_value).hex
            )
        else:
            marker.write_text(canonical_sha256(request_value).hex, encoding="ascii")
        self.calls.append((continuation.to_dict(), dict(options)))
        return {
            "ok": True,
            "status": "COMPLETED",
            "runtime_root": str(occurrence),
        }


class _WrongAuthenticatedRunner(_Runner):
    def __call__(self, continuation, **options):
        result = super().__call__(continuation, **options)
        marker = Path(result["runtime_root"]) / "authenticated-request-sha256"
        marker.write_text(canonical_sha256({"wrong": "child"}).hex, encoding="ascii")
        return result


class _CrashOnce:
    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.triggered = False

    def __call__(self, stage, frontier_ordinal, action_id):
        if stage == self.stage and not self.triggered:
            self.triggered = True
            raise RuntimeError(f"injected crash at {stage}")


def _inputs(tmp_path: Path, *, runner: _Runner | None = None):
    manifest, founder_request, _prepared_decision = _prepared()
    policy = _founder_policy()
    selected_runner = runner or _Runner()
    return {
        "manifest": manifest,
        "policy": policy,
        "founder_request": founder_request,
        "ledger_root": tmp_path / "ledger",
        "atlas": _atlas(manifest),
        "retained_parent_bank": _bank(manifest),
        "logical_capacity": 1,
        "maximum_frontiers": 2,
        "runtime_parent": tmp_path / "runtime",
        "dependencies": FounderClosedLoopDependenciesV1(
            observation_projector=_Projector(),
            child_runner=selected_runner,
        ),
    }


def test_live_loop_seals_founder_children_before_canonical_dispatch(
    tmp_path: Path,
) -> None:
    runner = _Runner()
    result = run_founder_closed_loop_v1(**_inputs(tmp_path, runner=runner))

    assert result.terminal is True
    assert result.shadow_mode is False
    # The final settled child is folded through a non-dispatching terminal
    # finalization decision before the terminal receipt is sealed.
    assert len(result.decisions) == 3
    assert len(result.children) == 2
    assert len(runner.calls) == 2
    assert all(child.parent_state_ref is None for child in result.children)
    assert all(child.effective_next_absolute_tick == 0 for child in result.children)
    assert all(child.retry_of_run_id is None for child in result.children)
    assert {
        child.child_request.to_value()["expect_worlds"] for child in result.children
    } == {3}
    assert result.atlas.revision == 3
    assert len(result.atlas.entries) == 2
    assert result.retained_parent_bank.revision == 3
    assert result.retained_parent_bank.current_entry_count == 1

    events = result.ledger["events"]
    assert [event["event_type"] for event in events] == [
        FRONTIER_OBSERVED,
        DECISION_PREPARED,
        ACTION_DISPATCH_BOUND,
        ACTION_SETTLED,
        FRONTIER_OBSERVED,
        DECISION_PREPARED,
        ACTION_DISPATCH_BOUND,
        ACTION_SETTLED,
        FRONTIER_OBSERVED,
        DECISION_PREPARED,
        EXPEDITION_TERMINAL,
    ]
    assert events[-2]["payload"]["terminal_frontier_finalization"] is True
    assert events[-2]["payload"]["action_ids"] == []
    assert events[-2]["payload"]["action_dispatch_suppressed"] is True
    for binding in (
        event["payload"]
        for event in events
        if event["event_type"] == ACTION_DISPATCH_BOUND
    ):
        assert binding["continuation"]["parent_state_ref"] is None
        assert binding["continuation"]["effective_next_absolute_tick"] == 0
        assert binding["historical_continuation_authority"] is False


def test_founder_loop_rejects_an_authenticated_result_for_another_child(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path, runner=_WrongAuthenticatedRunner())
    with pytest.raises(
        FounderClosedLoopError,
        match="names a different Science Request",
    ):
        run_founder_closed_loop_v1(**inputs)
    ledger = read_ledger_store_v1(inputs["ledger_root"])
    assert not any(event["event_type"] == ACTION_SETTLED for event in ledger["events"])


@pytest.mark.parametrize(
    "stage",
    [
        "AFTER_FRONTIER_OBSERVED",
        "AFTER_DECISION_PREPARED",
        "AFTER_ACTION_DISPATCH_BOUND",
    ],
)
def test_recovery_regenerates_the_same_decision_and_child_before_execution(
    tmp_path: Path, stage: str
) -> None:
    runner = _Runner()
    inputs = _inputs(tmp_path, runner=runner)
    crash = _CrashOnce(stage)
    with pytest.raises(RuntimeError, match="injected crash"):
        run_founder_closed_loop_v1(**inputs, interruption_hook=crash)

    partial = read_ledger_store_v1(inputs["ledger_root"])
    prepared = [
        event["payload"]
        for event in partial["events"]
        if event["event_type"] == DECISION_PREPARED
    ]
    bound = [
        event["payload"]
        for event in partial["events"]
        if event["event_type"] == ACTION_DISPATCH_BOUND
    ]
    result = run_founder_closed_loop_v1(**inputs)

    if prepared:
        assert result.decisions[0].to_dict() == prepared[0]["decision"]
    if bound:
        assert result.children[0].to_dict() == bound[0]["continuation"]
    assert result.terminal is True
    assert (
        len(
            [
                event
                for event in result.ledger["events"]
                if event["event_type"] == DECISION_PREPARED
            ]
        )
        == 3
    )


def test_recovery_completes_terminal_finalization_without_dispatch(
    tmp_path: Path,
) -> None:
    runner = _Runner()
    inputs = _inputs(tmp_path, runner=runner)
    crash = _CrashOnce("AFTER_TERMINAL_FINALIZATION_DECISION")
    with pytest.raises(RuntimeError, match="injected crash"):
        run_founder_closed_loop_v1(**inputs, interruption_hook=crash)

    partial = read_ledger_store_v1(inputs["ledger_root"])
    assert partial["events"][-1]["event_type"] == DECISION_PREPARED
    assert partial["events"][-1]["payload"]["terminal_frontier_finalization"] is True
    assert not any(
        event["event_type"] == EXPEDITION_TERMINAL for event in partial["events"]
    )
    calls_before_resume = len(runner.calls)

    result = run_founder_closed_loop_v1(**inputs)
    assert result.terminal is True
    assert len(runner.calls) == calls_before_resume
    assert result.ledger["events"][-1]["event_type"] == EXPEDITION_TERMINAL


def test_crash_after_execution_resumes_the_exact_action_occurrence(
    tmp_path: Path,
) -> None:
    runner = _Runner()
    inputs = _inputs(tmp_path, runner=runner)
    crash = _CrashOnce("AFTER_ACTION_EXECUTION")
    with pytest.raises(RuntimeError, match="injected crash"):
        run_founder_closed_loop_v1(**inputs, interruption_hook=crash)

    partial = read_ledger_store_v1(inputs["ledger_root"])
    binding = next(
        event["payload"]
        for event in partial["events"]
        if event["event_type"] == ACTION_DISPATCH_BOUND
    )
    assert not any(event["event_type"] == ACTION_SETTLED for event in partial["events"])

    result = run_founder_closed_loop_v1(**inputs)
    assert result.terminal is True
    assert runner.calls[0][0] == binding["continuation"]
    assert runner.calls[1][0] == binding["continuation"]
    assert runner.calls[0][1]["resume_root"] is None
    assert runner.calls[1][1]["resume_root"] is not None
    assert Path(runner.calls[1][1]["resume_root"]).parent == Path(
        binding["action_runtime_parent"]
    )


def test_terminal_failed_occurrence_is_preserved_and_gets_fresh_retry_root(
    tmp_path: Path,
) -> None:
    def write_sealed(path: Path, core: dict[str, object], field: str) -> None:
        path.write_text(
            json.dumps({**core, field: canonical_sha256(core).hex}, sort_keys=True),
            encoding="utf-8",
        )

    parent = tmp_path / "action"
    failed = parent / "failed-occurrence"
    failed.mkdir(parents=True)
    request_sha = {"algorithm": "sha256", "hex": "a" * 64}
    write_sealed(
        failed / "run_passport.json",
        {
            "run_id": "failed-occurrence",
            "campaign_id": "campaign-v1",
            "terminal_status": "failed",
            "source_identity": {"request_sha256": request_sha},
        },
        "passport_sha256",
    )
    write_sealed(
        failed / "run_artifact_failure_receipt.json",
        {
            "schema_version": "eiel_run_artifact_failure_receipt_v1",
            "run_id": "failed-occurrence",
            "campaign_id": "campaign-v1",
            "runtime_root": str(failed),
            "status": "failed",
        },
        "receipt_sha256",
    )
    write_sealed(
        failed / "science_campaign_failure_receipt.json",
        {
            "schema_version": "eiel_science_request_v1_run_failure_receipt_v1",
            "runtime_root": str(failed),
            "status": "SCIENCE_BATCH_EXECUTION_FAILED",
            "request_sha256": request_sha,
        },
        "receipt_sha256",
    )

    assert (
        _resume_root_v1(
            parent,
            expected_request_sha256="a" * 64,
            expected_campaign_id="campaign-v1",
        )
        is None
    )

    active = parent / "fresh-occurrence"
    active.mkdir()
    write_sealed(
        active / "run_passport.json",
        {
            "run_id": "fresh-occurrence",
            "campaign_id": "campaign-v1",
            "terminal_status": "active",
            "source_identity": {"request_sha256": request_sha},
        },
        "passport_sha256",
    )
    write_sealed(
        active / "run_journal.json",
        {
            "schema_version": "eiel_science_request_v1_run_journal_v1",
            "identity": {
                "campaign_id": "campaign-v1",
                "request_sha256": request_sha,
            },
            "status": "RUNNING",
            "batches": [],
            "aggregate": None,
            "last_event": "created",
        },
        "journal_sha256",
    )
    assert (
        _resume_root_v1(
            parent,
            expected_request_sha256="a" * 64,
            expected_campaign_id="campaign-v1",
        )
        == str(active.resolve())
    )


def test_retryable_scheduler_admission_occurrence_resumes_in_place(
    tmp_path: Path,
) -> None:
    def write_sealed(path: Path, core: dict[str, object], field: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({**core, field: canonical_sha256(core).hex}, sort_keys=True),
            encoding="utf-8",
        )

    parent = tmp_path / "action"
    failed = parent / "admission-occurrence"
    request_sha = {"algorithm": "sha256", "hex": "a" * 64}
    write_sealed(
        failed / "run_passport.json",
        {
            "run_id": "admission-occurrence",
            "campaign_id": "campaign-v1",
            "terminal_status": "failed",
            "source_identity": {"request_sha256": request_sha},
        },
        "passport_sha256",
    )
    write_sealed(
        failed / "run_artifact_failure_receipt.json",
        {
            "schema_version": "eiel_run_artifact_failure_receipt_v1",
            "run_id": "admission-occurrence",
            "campaign_id": "campaign-v1",
            "runtime_root": str(failed),
            "status": "failed",
        },
        "receipt_sha256",
    )
    write_sealed(
        failed / "science_campaign_failure_receipt.json",
        {
            "schema_version": "eiel_science_request_v1_run_failure_receipt_v1",
            "runtime_root": str(failed),
            "status": "SCIENCE_BATCH_EXECUTION_FAILED",
            "request_sha256": request_sha,
        },
        "receipt_sha256",
    )
    attempt = failed / "batches" / "batch-0001" / "attempt-0001"
    result = {
        "schema_version": "eiel_execution_result_v2",
        "ok": False,
        "status": "precheck_failed",
        "result_class": "ATTEMPT_ADMISSION_UNAVAILABLE",
        "warnings": [
            "CRCA_SCHEDULER_ATTEMPT_REJECTED:single-worker execution lock"
        ],
        "crca_attempt_authority": {
            "execution_authority_granted": False,
            "reason_codes": [
                "CRCA_SCHEDULER_ATTEMPT_REJECTED:single-worker execution lock"
            ],
        },
    }
    attempt.mkdir(parents=True, exist_ok=True)
    (attempt / "execution_result.json").write_text(
        json.dumps(result, sort_keys=True), encoding="utf-8"
    )
    journal = {
        "schema_version": "eiel_science_request_v1_run_journal_v1",
        "identity": {
            "campaign_id": "campaign-v1",
            "request_sha256": request_sha,
        },
        "status": "FAILED",
        "current_batch_index": 1,
        "batches": [
            {"batch_index": 0, "state": "COMPLETED"},
            {
                "batch_index": 1,
                "state": "FAILED",
                "attempt": 1,
                "attempt_root": "batches/batch-0001/attempt-0001",
                "failure": {
                    "type": "SCIENCE_BATCH_EXECUTION_FAILED",
                    "message": "ATTEMPT_ADMISSION_UNAVAILABLE",
                },
            },
        ],
        "aggregate": None,
        "last_event": "batch-0001-failed",
    }
    write_sealed(failed / "run_journal.json", journal, "journal_sha256")

    assert _resume_root_v1(
        parent,
        expected_request_sha256="a" * 64,
        expected_campaign_id="campaign-v1",
    ) == str(failed.resolve())


def test_shadow_mode_records_replayed_decisions_without_dispatch(
    tmp_path: Path,
) -> None:
    runner = _Runner()
    inputs = _inputs(tmp_path, runner=runner)
    inputs.pop("runtime_parent")
    result = run_founder_closed_loop_v1(
        **inputs,
        shadow_mode=True,
        shadow_observation_frontiers=[
            [tmp_path / "retained-science-a"],
            [tmp_path / "retained-science-b"],
        ],
    )

    assert result.terminal is True
    assert result.shadow_mode is True
    assert len(result.decisions) == 2
    assert result.children == ()
    assert runner.calls == []
    assert not any(
        event["event_type"] in {ACTION_DISPATCH_BOUND, ACTION_SETTLED}
        for event in result.ledger["events"]
    )
    decision_events = [
        event
        for event in result.ledger["events"]
        if event["event_type"] == DECISION_PREPARED
    ]
    assert all(event["payload"]["action_ids"] == [] for event in decision_events)
    assert all(
        event["payload"]["proposed_founder_action_ids"] for event in decision_events
    )


def test_shadow_founder_selection_excludes_historical_policy_actions(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    inputs.pop("runtime_parent")
    result = run_founder_closed_loop_v1(
        **{
            **inputs,
            "logical_capacity": 4,
            "maximum_frontiers": 1,
        },
        shadow_mode=True,
        shadow_observation_frontiers=[[tmp_path / "retained-science"]],
    )

    decision = result.decisions[0]
    assert any(row.action_kind.value == "BRANCH_LOCAL" for row in decision.action_rows)
    event = next(
        event
        for event in result.ledger["events"]
        if event["event_type"] == DECISION_PREPARED
    )
    founder_ids = set(event["payload"]["proposed_founder_action_ids"])
    assert len(founder_ids) == 3
    assert all(
        row.action_kind.value == "EXPLORE_GLOBAL"
        for row in decision.action_rows
        if row.action_id in founder_ids
    )


def test_observation_only_states_update_atlas_but_never_parent_bank(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    inputs["dependencies"] = FounderClosedLoopDependenciesV1(
        observation_projector=_ObservationOnlyProjector(),
        child_runner=_Runner(),
    )
    result = run_founder_closed_loop_v1(
        **{
            **inputs,
            "maximum_frontiers": 1,
            "initial_observation_sources": [tmp_path / "initial-observation"],
        },
    )

    # The initial observation and the terminal child observation are both
    # scientific observations even though neither is exactly continuable.
    assert result.atlas.revision == 2
    assert result.atlas.entries
    assert result.retained_parent_bank.revision == 2
    assert result.retained_parent_bank.current_entry_count == 0
    decision_event = next(
        event
        for event in result.ledger["events"]
        if event["event_type"] == DECISION_PREPARED
    )
    assert {
        "state_ref_id": (
            "SR-"
            + _ObservationOnlyProjector()(
                tmp_path / "probe", manifest=inputs["manifest"]
            )
            .frames[0]
            .source_state_ref.state_ref_sha256()
            .hex
        ),
        "reason_code": "STATE_NOT_EXACTLY_CONTINUABLE",
    } in decision_event["payload"]["retention_outcomes"]["policy_not_retained"]


def test_loop_refuses_to_exceed_manifest_frontier_bound(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    with pytest.raises(FounderClosedLoopError, match="within the frozen manifest"):
        run_founder_closed_loop_v1(
            **{**inputs, "maximum_frontiers": inputs["manifest"].maximum_frontiers + 1}
        )


def test_safe_frontier_stop_is_durable_and_recoverable(tmp_path: Path) -> None:
    runner = _Runner()
    calls: list[int] = []

    def stop_after_first(frontier_ordinal, ledger):
        if not ledger["events"]:
            return None
        assert ledger["events"][-1]["event_type"] == ACTION_SETTLED
        calls.append(frontier_ordinal)
        return {
            "reason": "WALL_TIME_BUDGET_REACHED_AT_SAFE_FRONTIER",
            "deadline_at_utc": "2026-08-29T09:00:00Z",
            "observed_at_utc": "2026-08-29T09:00:01Z",
        }

    result = run_founder_closed_loop_v1(
        **_inputs(tmp_path, runner=runner),
        safe_frontier_stop=stop_after_first,
    )

    assert calls == [0]
    assert len(result.decisions) == 2
    assert len(result.children) == 1
    assert [event["event_type"] for event in result.ledger["events"]][-2:] == [
        STOP_REQUESTED,
        EXPEDITION_TERMINAL,
    ]
    finalization = next(
        event["payload"]
        for event in result.ledger["events"]
        if event["event_type"] == DECISION_PREPARED
        and event["payload"].get("terminal_frontier_finalization") is True
    )
    assert finalization["action_ids"] == []
    assert result.ledger["events"][-1]["payload"]["terminal_status"] == "STOPPED"

    replay = run_founder_closed_loop_v1(
        **_inputs(tmp_path, runner=runner),
        safe_frontier_stop=lambda *_: pytest.fail(
            "terminal replay called stop observer"
        ),
    )
    assert replay.ledger == result.ledger
    assert len(runner.calls) == 1
