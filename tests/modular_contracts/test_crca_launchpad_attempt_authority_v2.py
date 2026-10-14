from __future__ import annotations

import copy
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Callable

import pytest

from eiel_composition.launchpad_attempt_authority import (
    AUTHORITATIVE_BUNDLE_SCHEMA_VERSION,
    AUTHORITATIVE_BUNDLE_SCHEMA_VERSION_V2,
    AuthoritativeAttemptBundleV1,
    AuthoritativeAttemptBundleV2,
    LaunchpadAttemptAuthorityError,
    authoritative_attempt_bundle_from_dict,
)
from emergence_contracts import (
    AttemptAdmissionRequestV1,
    AttemptFrontiersV1,
    AttemptRetryBasisCodeV1,
    AttemptRetryBasisV1,
    AttemptTerminalStatusV1,
    AttemptTerminalV1,
    RestoreCheckpointAuthorizationV1,
    RunOccurrenceV1,
    canonical_sha256 as execution_sha256,
    derive_attempt_id,
)
from tests.modular_contracts.launchpad_fixtures import (
    TEST_PROCESS_ID,
    TEST_PROCESS_START_IDENTITY,
    T1,
    T2,
    _ed,
    _fixture,
    _prepare_start,
    _sd,
    _terminal_for_started,
)
from tools.actors.common import ActorError
from tools.scheduler.launchpad_attempts import (
    PREPARATION_SCHEMA_VERSION,
    PREPARATION_SCHEMA_VERSION_V2,
)
from tools.scheduler.programme_contracts import (
    LaunchpadAttemptClaimV1,
    canonical_sha256,
)


def _child_fixture(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _fixture(tmp_path / "source")
    source_started = _prepare_start(source)
    source_bundle = AuthoritativeAttemptBundleV1.from_dict(source_started["bundle"])
    source_terminal = AttemptTerminalV1(
        attempt_id=source_bundle.attempt_start.attempt_id,
        attempt_start_digest=source_bundle.attempt_start.start_digest(),
        status=AttemptTerminalStatusV1.COMPLETED,
        final_frontiers=AttemptFrontiersV1(4, 4, 2, 2),
        result_digest=_ed("source-terminal-result"),
        reason="source checkpoint is durably sealed",
    )
    source_record = {
        "start": source_bundle.attempt_start.to_dict(),
        "concretized_execution": source_bundle.concretized_execution.to_dict(),
        "terminal": source_terminal.to_dict(),
    }

    child = _fixture(tmp_path / "child")
    semantic = child["semantic"]
    checkpoint_digest = _ed("workhorse-occurrence-descriptor")
    run = RunOccurrenceV1.create(
        experiment_id=semantic.experiment_id,
        cell_or_repeat_ordinal=1,
        semantic_spec_digest=semantic.semantic_spec_digest(),
        parent_checkpoint_digest=checkpoint_digest,
        branch_spec_digest=_ed("closed-workhorse-child-branch"),
    )
    attempt_id = derive_attempt_id(
        run_id=run.run_id,
        monotonic_ordinal=1,
        nonce="fresh-workhorse-child-restore",
    )
    original_admission = child["prepare"]["attempt_admission_request"]
    admission = AttemptAdmissionRequestV1(
        resource_requirements_digest=_ed("resource-requirements"),
        executor_constraints_digest=_ed("executor-constraints"),
        requested_restore_checkpoint_digest=checkpoint_digest,
        observational_persistence_cadence_digest=None,
        proposed_reason=original_admission["proposed_reason"],
    )
    concretized = replace(
        child["concretized"],
        attempt_admission_request_digest=admission.attempt_admission_request_digest(),
    )
    retry = AttemptRetryBasisV1(
        run_id=run.run_id,
        candidate_attempt_id=attempt_id,
        prior_attempt_id=None,
        code=AttemptRetryBasisCodeV1.INITIAL_EXECUTION,
        reason="Fresh child Run restored from one external source occurrence.",
        semantic_spec_digest=semantic.semantic_spec_digest(),
        prior_concretized_execution_digest=None,
        candidate_concretized_execution_digest=(
            concretized.concretized_execution_digest()
        ),
        basis_evidence_digests=(),
        implementation_conformance_digest=None,
    )
    frontiers = AttemptFrontiersV1(2, 2, 2, 2)
    restore = RestoreCheckpointAuthorizationV1(
        checkpoint_digest=checkpoint_digest,
        run_id=source_bundle.run_occurrence.run_id,
        source_attempt_id=source_bundle.attempt_start.attempt_id,
        source_attempt_terminal_digest=source_terminal.terminal_digest(),
        absolute_next_tick=2,
        sealed_bundle_digest=_ed("workhorse-source-materials"),
        completion_marker_digest=_ed("workhorse-occurrence-commit"),
    )
    original_claim = child["claim"]
    claim_values = {
        field.name: getattr(original_claim, field.name)
        for field in fields(LaunchpadAttemptClaimV1)
        if field.name not in {"claim_id", "schema_version"}
    }
    claim_values.update(
        {
            "run_id": str(run.run_id),
            "attempt_id": str(attempt_id),
            "run_occurrence_digest": canonical_sha256(run.to_dict()),
            "concretized_execution_digest": _sd(
                concretized.concretized_execution_digest()
            ),
            "attempt_admission_request_digest": _sd(
                admission.attempt_admission_request_digest()
            ),
            "retry_basis_digest": _sd(retry.basis_digest()),
            "initial_frontiers_digest": canonical_sha256(frontiers.to_dict()),
        }
    )
    prepare = dict(child["prepare"])
    prepare.update(
        {
            "claim": LaunchpadAttemptClaimV1.create(**claim_values),
            "run_occurrence": run.to_dict(),
            "concretized_execution": concretized.to_dict(),
            "attempt_admission_request": admission.to_dict(),
            "retry_basis": retry.to_dict(),
            "initial_frontiers": frontiers.to_dict(),
            "monotonic_ordinal": 1,
            "nonce": "fresh-workhorse-child-restore",
            "parent_attempt_id": None,
            "restore_authorization": restore.to_dict(),
            "restore_source_attempt_record": source_record,
            "prior_attempt_records": (),
            "resource_scope_digest": canonical_sha256(
                {
                    "schema_version": "eiel_exclusive_gpu_scope_v1",
                    "resource_ids": list(prepare["resource_ids"]),
                    "run_id": str(run.run_id),
                }
            ),
        }
    )
    child.update(
        {
            "claim": prepare["claim"],
            "run": run,
            "concretized": concretized,
            "attempt_id": str(attempt_id),
            "prepare": prepare,
        }
    )
    return source_record, child


def _start_child(child: dict[str, Any]) -> dict[str, Any]:
    prepared = child["issuer"].prepare(**child["prepare"])
    assert prepared["preparation"]["schema_version"] == PREPARATION_SCHEMA_VERSION_V2
    return child["issuer"].start(
        attempt_id=child["attempt_id"],
        process_owner_id="provider-process-owner",
        process_owner_digest=child["owner_digest"],
        process_id=TEST_PROCESS_ID,
        process_start_identity=TEST_PROCESS_START_IDENTITY,
        at_utc=T1,
    )


def test_v1_emission_is_unchanged_and_rejects_v2_field(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    prepared = fixture["issuer"].prepare(**fixture["prepare"])
    started = fixture["issuer"].start(
        attempt_id=fixture["attempt_id"],
        process_owner_id="provider-process-owner",
        process_owner_digest=fixture["owner_digest"],
        process_id=TEST_PROCESS_ID,
        process_start_identity=TEST_PROCESS_START_IDENTITY,
        at_utc=T1,
    )

    assert prepared["preparation"]["schema_version"] == PREPARATION_SCHEMA_VERSION
    assert "restore_source_attempt_record" not in prepared["preparation"]
    assert started["bundle"]["schema_version"] == AUTHORITATIVE_BUNDLE_SCHEMA_VERSION
    assert "restore_source_attempt_record" not in started["bundle"]
    assert (
        AuthoritativeAttemptBundleV1.from_dict(started["bundle"]).raw
        == (started["bundle"])
    )
    extra = {**started["bundle"], "restore_source_attempt_record": None}
    with pytest.raises(LaunchpadAttemptAuthorityError, match="fields differ"):
        AuthoritativeAttemptBundleV1.from_dict(extra)


def test_v2_bundle_binds_external_source_and_fresh_child_run(tmp_path: Path) -> None:
    source_record, child = _child_fixture(tmp_path)
    started = _start_child(child)
    bundle = started["bundle"]

    assert bundle["schema_version"] == AUTHORITATIVE_BUNDLE_SCHEMA_VERSION_V2
    assert bundle["restore_source_attempt_record"] == source_record
    parsed = AuthoritativeAttemptBundleV2.from_dict(bundle)
    assert authoritative_attempt_bundle_from_dict(bundle) == parsed
    assert parsed.validate()["schema_version"] == (
        "crca_authoritative_launchpad_attempt_validation_v2"
    )
    assert parsed.attempt_start.monotonic_ordinal == 1
    assert parsed.attempt_start.parent_attempt_id is None
    assert parsed.prior_records == ()
    assert parsed.run_occurrence.parent_checkpoint_digest == (
        parsed.restore_authorization.checkpoint_digest
    )
    assert parsed.attempt_start.restore_checkpoint_digest == (
        parsed.run_occurrence.parent_checkpoint_digest
    )
    assert parsed.restore_source_record.start.run_id != parsed.run_occurrence.run_id
    with pytest.raises(LaunchpadAttemptAuthorityError, match="fields differ"):
        AuthoritativeAttemptBundleV1.from_dict(bundle)


def test_scheduler_reopens_exact_terminal_restore_source_read_only(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    started = _prepare_start(fixture)
    terminal = _terminal_for_started(
        started,
        status=AttemptTerminalStatusV1.COMPLETED,
        frontiers=AttemptFrontiersV1(4, 4, 2, 2),
    )
    fixture["issuer"].terminalize(
        attempt_id=fixture["attempt_id"],
        process_owner_id="provider-process-owner",
        process_owner_digest=fixture["owner_digest"],
        terminal=terminal.to_dict(),
        at_utc=T2,
    )
    before = {
        str(path): path.read_bytes()
        for path in fixture["issuer"].scheduler.rglob("*.json")
    }

    terminal_record = fixture["issuer"].terminal_attempt_record(fixture["attempt_id"])
    source = fixture["issuer"].restore_source_attempt_record(fixture["attempt_id"])

    assert terminal_record["terminal_bundle"]["attempt_terminal"] == (
        terminal.to_dict()
    )
    assert terminal_record["terminal_bundle_path"].endswith(".json")
    assert (
        terminal_record["terminal_bundle_digest"]
        == canonical_sha256(terminal_record["terminal_bundle"]).to_dict()
    )
    assert terminal_record["restore_source_attempt_record"] == source
    assert source == {
        "start": started["bundle"]["attempt_start"],
        "concretized_execution": started["bundle"]["concretized_execution"],
        "terminal": terminal.to_dict(),
    }
    assert {
        str(path): path.read_bytes()
        for path in fixture["issuer"].scheduler.rglob("*.json")
    } == before


def _wrong_source_run(prepare: dict[str, Any]) -> None:
    prepare["restore_source_attempt_record"]["start"]["run_id"] = "RUN-FOREIGN"


def _wrong_source_terminal(prepare: dict[str, Any]) -> None:
    prepare["restore_source_attempt_record"]["terminal"]["result_digest"] = _ed(
        "forged-result"
    ).to_dict()


def _source_frontier_before_checkpoint(prepare: dict[str, Any]) -> None:
    terminal = prepare["restore_source_attempt_record"]["terminal"]
    terminal["final_frontiers"]["durable_checkpoint_next_tick"] = 1
    prepare["restore_authorization"]["source_attempt_terminal_digest"] = (
        execution_sha256(terminal).to_dict()
    )


def _child_is_same_run(prepare: dict[str, Any]) -> None:
    prepare["run_occurrence"]["run_id"] = prepare["restore_authorization"]["run_id"]


def _child_has_parent_attempt(prepare: dict[str, Any]) -> None:
    prepare["parent_attempt_id"] = prepare["restore_authorization"]["source_attempt_id"]


@pytest.mark.parametrize(
    "mutate",
    [
        _wrong_source_run,
        _wrong_source_terminal,
        _source_frontier_before_checkpoint,
        _child_is_same_run,
        _child_has_parent_attempt,
    ],
)
def test_v2_hostile_source_or_child_mutation_fails_before_attempt(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    _source_record, child = _child_fixture(tmp_path)
    prepare = copy.deepcopy(child["prepare"])
    mutate(prepare)

    with pytest.raises(ActorError, match="restore|source|child|Run occurrence"):
        child["issuer"].prepare(**prepare)
    assert not child["issuer"].starts_dir.exists()
    assert not child["issuer"].bundle_heads_dir.exists()


def test_v2_parser_rejects_mutated_source_record(tmp_path: Path) -> None:
    _source_record, child = _child_fixture(tmp_path)
    bundle = copy.deepcopy(_start_child(child)["bundle"])
    bundle["restore_source_attempt_record"]["terminal"]["result_digest"] = _ed(
        "mutated-after-issue"
    ).to_dict()

    with pytest.raises(
        (LaunchpadAttemptAuthorityError, ValueError),
        match="terminal|source|digest",
    ):
        AuthoritativeAttemptBundleV2.from_dict(bundle)
