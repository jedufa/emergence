from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from eiel_composition.adaptive_expedition.selected_world_exactness_evidence import (
    RECEIPT_CLAIM_CLASS,
    RECEIPT_SCHEMA,
    SelectedWorldExactnessEvidenceError,
    _ExecutionIdentity,
    _ParentEvidence,
    _TickEvidence,
    _TrajectoryEvidence,
    _assemble_receipt,
    _canonical_bytes,
    _seal_receipt,
    validate_selected_world_exactness_receipt_v1,
)


def _digest(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def _execution(root: Path, label: str) -> _ExecutionIdentity:
    run_root = root / label
    run_root.mkdir()
    return _ExecutionIdentity(
        run_root=run_root,
        request_sha256=_digest(f"request:{label}"),
        request_receipt_file_sha256=_digest(f"request-file:{label}"),
        campaign_spec_sha256=_digest(f"campaign:{label}"),
        plan_sha256=_digest(f"plan:{label}"),
        execution_result_sha256=_digest(f"result:{label}"),
        execution_result_file_sha256=_digest(f"result-file:{label}"),
        lifecycle_manifest_sha256=_digest(f"lifecycle:{label}"),
        storage_receipt_sha256=_digest(f"storage:{label}"),
        terminal_lifecycle_commit_sha256=_digest(f"commit:{label}"),
        terminal_storage_receipt_sha256=_digest(f"terminal-storage:{label}"),
        source_commit="1" * 40,
        result={},
    )


def _tick(
    root: Path,
    *,
    label: str,
    tick: int,
    payload: bytes,
    previous: str | None,
) -> _TickEvidence:
    state_path = root / f"{label}-{tick}.bin"
    state_path.write_bytes(payload)
    occurrence = _digest(f"occurrence:{label}:{tick}")
    return _TickEvidence(
        tick=tick,
        occurrence_sha256=occurrence,
        occurrence_parent_sha256=previous,
        index_manifest_path=root / f"{label}-{tick}-index.json",
        index_manifest_file_sha256=_digest(f"index-file:{label}:{tick}"),
        index_manifest_body_sha256=_digest(f"index-body:{label}:{tick}"),
        world_state_set_sha256=_digest(f"world-set:{label}:{tick}"),
        world_count=2 if label == "reference" else 1,
        rng_epoch=4,
        rng_key=83,
        state_sha256=_digest(payload),
        state_bytes=len(payload),
        physical_state_path=state_path,
        state_offset=0,
        event_trajectory_sha256=_digest(payload),
    )


def _fixtures(
    root: Path,
) -> tuple[
    _ParentEvidence,
    _TrajectoryEvidence,
    _TrajectoryEvidence,
    _TrajectoryEvidence,
]:
    parent_state = root / "parent-state.bin"
    parent_state.write_bytes(b"exact-parent")
    parent = _ParentEvidence(
        execution=_execution(root, "founder"),
        world_uid=210000,
        branch_tick=6,
        parent_run_id="founder-run",
        parent_occurrence_sha256=_digest("parent-occurrence"),
        checkpoint_occurrence_sha256=_digest("checkpoint-occurrence"),
        semantic_world_sha256=_digest("semantic-world"),
        control_history_sha256=_digest("control-history"),
        selected_source_object_sha256=_digest("source-object"),
        exact_state_sha256=_digest(b"exact-parent"),
        rng_state_sha256=_digest("rng-state"),
        event_prefix_sha256=_digest("event-prefix"),
        rng_epoch=4,
        rng_key=83,
        active_control_state_sha256=_digest("active-controls"),
        source_index_manifest_path=root / "parent-index.json",
        source_index_manifest_file_sha256=_digest("parent-index-file"),
        source_index_manifest_body_sha256=_digest("parent-index-body"),
        source_world_state_set_sha256=_digest("parent-world-set"),
        source_checkpoint_identity_sha256=_digest("source-checkpoint"),
        source_physical_state_path=parent_state,
        source_physical_state_sha256=_digest(b"exact-parent"),
        source_physical_state_bytes=len(b"exact-parent"),
        source_physical_slot=1,
        source_state_offset=0,
        source_state_bytes=len(b"exact-parent"),
    )

    def trajectory(
        label: str,
        *,
        physical_B: int,
        payloads: tuple[bytes, bytes],
        changed: bool,
    ) -> _TrajectoryEvidence:
        ticks: list[_TickEvidence] = []
        previous = None
        for tick, payload in zip((7, 8), payloads, strict=True):
            row = _tick(
                root,
                label=label,
                tick=tick,
                payload=payload,
                previous=previous,
            )
            ticks.append(row)
            previous = row.occurrence_sha256
        return _TrajectoryEvidence(
            label=label,
            execution=_execution(root, label),
            branch_tick=6,
            target_tick=8,
            physical_B=physical_B,
            continuation_scope="BRANCH_LOCAL",
            science_continuation_contract_sha256=_digest("science-contract"),
            science_continuation_identity_sha256=_digest(f"science-identity:{label}"),
            continuation_contract_sha256=_digest(f"continuation:{label}"),
            repack_binding_sha256=_digest(f"repack:{label}"),
            attempt_id=f"attempt-{label}",
            source_world_state_set_sha256=parent.source_world_state_set_sha256,
            source_selected_state_sha256=parent.exact_state_sha256,
            source_rng_key=parent.rng_key,
            source_checkpoint_identity_sha256=(
                parent.source_checkpoint_identity_sha256
            ),
            source_index_manifest_path=parent.source_index_manifest_path,
            source_index_manifest_file_sha256=(
                parent.source_index_manifest_file_sha256
            ),
            source_physical_state_path=parent.source_physical_state_path,
            source_physical_slot=parent.source_physical_slot,
            source_state_offset=parent.source_state_offset,
            control_contract_sha256=(
                _digest("changed-control-contract") if changed else None
            ),
            control_application_sha256=(
                _digest("changed-control-application") if changed else None
            ),
            applied_control_next_absolute_tick=6 if changed else None,
            first_permitted_divergence_tick=6 if changed else None,
            controls={"selection_strength": 0.25} if changed else None,
            ticks=tuple(ticks),
        )

    reference = trajectory(
        "reference", physical_B=2, payloads=(b"same-7", b"same-8"), changed=False
    )
    same = trajectory(
        "same", physical_B=1, payloads=(b"same-7", b"same-8"), changed=False
    )
    changed = trajectory(
        "changed",
        physical_B=1,
        payloads=(b"changed-7", b"changed-8"),
        changed=True,
    )
    return parent, reference, same, changed


def test_seals_deterministic_engineering_only_receipt(tmp_path: Path) -> None:
    parent, reference, same, changed = _fixtures(tmp_path)

    first = _assemble_receipt(
        parent=parent,
        reference=reference,
        same_control=same,
        changed_control=changed,
    )
    second = _assemble_receipt(
        parent=parent,
        reference=reference,
        same_control=same,
        changed_control=changed,
    )
    assert first == second
    assert first["schema_version"] == RECEIPT_SCHEMA
    assert first["claim_class"] == RECEIPT_CLAIM_CLASS
    assert (
        first["same_control_repack_exactness"][
            "all_compared_post_branch_ticks_byte_equal"
        ]
        is True
    )
    assert first["changed_control_divergence"]["first_observed_divergence_tick"] == 7
    assert first["qualification_authority_granted"] is False
    assert first["science_authority_granted"] is False

    output = tmp_path / "receipt.json"
    sealed = _seal_receipt(output, first)
    replayed = _seal_receipt(output, second)
    verified = validate_selected_world_exactness_receipt_v1(
        output, reauthenticate_sources=False
    )
    assert sealed == replayed == verified
    assert output.stat().st_mode & 0o222 == 0


def test_rejects_same_control_repack_byte_difference(tmp_path: Path) -> None:
    parent, reference, same, changed = _fixtures(tmp_path)
    altered = _tick(
        tmp_path,
        label="same-altered",
        tick=7,
        payload=b"not-the-reference",
        previous=None,
    )
    same = replace(same, ticks=(altered, same.ticks[1]))

    with pytest.raises(
        SelectedWorldExactnessEvidenceError,
        match="same-control B2/B1 selected state differs at tick 7",
    ):
        _assemble_receipt(
            parent=parent,
            reference=reference,
            same_control=same,
            changed_control=changed,
        )


def test_rejects_child_identity_alias_and_wrong_branch_control_tick(
    tmp_path: Path,
) -> None:
    parent, reference, same, changed = _fixtures(tmp_path)
    aliased = replace(changed, execution=same.execution)
    with pytest.raises(
        SelectedWorldExactnessEvidenceError,
        match="aliases another child request",
    ):
        _assemble_receipt(
            parent=parent,
            reference=reference,
            same_control=same,
            changed_control=aliased,
        )

    wrong_tick = replace(changed, applied_control_next_absolute_tick=7)
    with pytest.raises(
        SelectedWorldExactnessEvidenceError,
        match="changed-control branch-tick contract differs",
    ):
        _assemble_receipt(
            parent=parent,
            reference=reference,
            same_control=same,
            changed_control=wrong_tick,
        )


def test_rejects_prebranch_rows_and_source_byte_tampering(tmp_path: Path) -> None:
    parent, reference, same, changed = _fixtures(tmp_path)
    prebranch = replace(changed.ticks[0], tick=5)
    with pytest.raises(
        SelectedWorldExactnessEvidenceError,
        match="changed-control branch-tick contract differs",
    ):
        _assemble_receipt(
            parent=parent,
            reference=reference,
            same_control=same,
            changed_control=replace(changed, ticks=(prebranch, changed.ticks[1])),
        )

    same.ticks[0].physical_state_path.write_bytes(b"tampered")
    with pytest.raises(
        SelectedWorldExactnessEvidenceError,
        match="bytes changed after receipt authentication",
    ):
        _assemble_receipt(
            parent=parent,
            reference=reference,
            same_control=same,
            changed_control=changed,
        )


def test_rejects_writable_or_content_tampered_sealed_receipt(tmp_path: Path) -> None:
    parent, reference, same, changed = _fixtures(tmp_path)
    value = _assemble_receipt(
        parent=parent,
        reference=reference,
        same_control=same,
        changed_control=changed,
    )
    output = tmp_path / "receipt.json"
    _seal_receipt(output, value)
    output.chmod(0o644)

    with pytest.raises(
        SelectedWorldExactnessEvidenceError,
        match="schema, seal, or authority differs",
    ):
        validate_selected_world_exactness_receipt_v1(output)


def _rewrite_with_recomputed_self_digest(path: Path, value: dict[str, object]) -> None:
    body = {key: item for key, item in value.items() if key != "receipt_content_sha256"}
    value["receipt_content_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    path.chmod(0o644)
    path.write_bytes(_canonical_bytes(value))
    path.chmod(0o444)


def test_recomputes_divergence_and_request_relations_after_self_digest_tamper(
    tmp_path: Path,
) -> None:
    parent, reference, same, changed = _fixtures(tmp_path)
    value = _assemble_receipt(
        parent=parent,
        reference=reference,
        same_control=same,
        changed_control=changed,
    )
    output = tmp_path / "receipt.json"
    _seal_receipt(output, value)

    tampered = dict(value)
    tampered["changed_control_divergence"] = dict(
        value["changed_control_divergence"]
    )
    tampered["changed_control_divergence"]["first_observed_divergence_tick"] = 0
    _rewrite_with_recomputed_self_digest(output, tampered)
    with pytest.raises(
        SelectedWorldExactnessEvidenceError,
        match="divergence summary does not recompute",
    ):
        validate_selected_world_exactness_receipt_v1(output)

    tampered = dict(value)
    tampered["request_identity_binding"] = dict(value["request_identity_binding"])
    tampered["request_identity_binding"][
        "changed_control_execution_request_sha256"
    ] = tampered["request_identity_binding"][
        "same_control_execution_request_sha256"
    ]
    _rewrite_with_recomputed_self_digest(output, tampered)
    with pytest.raises(
        SelectedWorldExactnessEvidenceError,
        match="request identity summary does not recompute",
    ):
        validate_selected_world_exactness_receipt_v1(output)


def test_rejects_embedded_science_or_qualification_claim_after_reseal(
    tmp_path: Path,
) -> None:
    parent, reference, same, changed = _fixtures(tmp_path)
    value = _assemble_receipt(
        parent=parent,
        reference=reference,
        same_control=same,
        changed_control=changed,
    )
    output = tmp_path / "receipt.json"
    _seal_receipt(output, value)
    tampered = dict(value)
    tampered["trajectories"] = dict(value["trajectories"])
    changed_projection = dict(
        tampered["trajectories"]["repacked_b1_changed_control"]
    )
    changed_projection["control_binding"] = dict(
        changed_projection["control_binding"]
    )
    changed_projection["control_binding"]["controls"] = {
        "scientific_claim": "SUPPORTED"
    }
    tampered["trajectories"]["repacked_b1_changed_control"] = changed_projection
    _rewrite_with_recomputed_self_digest(output, tampered)
    with pytest.raises(
        SelectedWorldExactnessEvidenceError,
        match="contains a forbidden claim",
    ):
        validate_selected_world_exactness_receipt_v1(output)


def test_receipt_excludes_mutable_current_storage_revision_and_scopes_carrier(
    tmp_path: Path,
) -> None:
    parent, reference, same, changed = _fixtures(tmp_path)
    value = _assemble_receipt(
        parent=parent,
        reference=reference,
        same_control=same,
        changed_control=changed,
    )
    for execution in [
        value["parent"]["execution"],
        *[row["execution"] for row in value["trajectories"].values()],
    ]:
        assert "storage_receipt_sha256" not in execution
        assert "terminal_storage_receipt_sha256" in execution
    assert value["carrier_identity_scope"] == "LOW_LEVEL_INDEXED_SELECTED_STATE_ONLY"
    assert "parent_control_history_bound" not in value["checks"]


def test_default_verify_reauthenticates_roots_and_rejects_coherent_forgery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, reference, same, changed = _fixtures(tmp_path)
    value = _assemble_receipt(
        parent=parent,
        reference=reference,
        same_control=same,
        changed_control=changed,
    )
    output = tmp_path / "receipt.json"
    _seal_receipt(output, value)

    import eiel_composition.adaptive_expedition.selected_world_exactness_evidence as module

    monkeypatch.setattr(module, "_reopen_parent", lambda *args, **kwargs: parent)
    reopened = {
        "reference_b2_same_control": reference,
        "repacked_b1_same_control": same,
        "repacked_b1_changed_control": changed,
    }
    monkeypatch.setattr(
        module,
        "_reopen_trajectory",
        lambda label, *args, **kwargs: reopened[label],
    )
    assert validate_selected_world_exactness_receipt_v1(output)[
        "receipt_content_sha256"
    ] == value["receipt_content_sha256"]

    forged = json.loads(json.dumps(value))
    fake_state = _digest("coherent-forged-parent-and-child-state")
    forged["parent"]["exact_state_sha256"] = fake_state
    for trajectory in forged["trajectories"].values():
        trajectory["source_binding"]["selected_state_sha256"] = fake_state
    _rewrite_with_recomputed_self_digest(output, forged)
    with pytest.raises(
        SelectedWorldExactnessEvidenceError,
        match="differs from reauthenticated canonical source roots",
    ):
        validate_selected_world_exactness_receipt_v1(output)


def test_default_verify_rejects_forged_validator_identity(tmp_path: Path) -> None:
    parent, reference, same, changed = _fixtures(tmp_path)
    value = _assemble_receipt(
        parent=parent,
        reference=reference,
        same_control=same,
        changed_control=changed,
    )
    output = tmp_path / "receipt.json"
    _seal_receipt(output, value)
    forged = json.loads(json.dumps(value))
    forged["validator"]["implementation_sha256"] = _digest("forged-validator")
    _rewrite_with_recomputed_self_digest(output, forged)
    with pytest.raises(
        SelectedWorldExactnessEvidenceError,
        match="schema, seal, or authority differs",
    ):
        validate_selected_world_exactness_receipt_v1(output)
