from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.campaigns.decision_evidence_retention import (
    AUDIT_HOLD_ACTIVE,
    CLASS_A,
    DecisionEvidenceRetentionError,
    build_audit_hold_manifest,
    guard_prune_plan,
    load_manifest,
    protected_entries_for_selection,
    write_audit_hold_manifest,
)
from tools.scripts.eiel_runtime_storage_prune import build_prune_plan


def _projection(path: Path) -> None:
    path.parent.mkdir(parents=True)
    path.write_text('{"schema_version":"native_runtime_checkpoint_projection_v1"}\n', encoding="utf-8")


def test_manifest_binds_required_projection_generations(tmp_path: Path) -> None:
    manifest = build_audit_hold_manifest(
        campaign_id="gv2-b48-r1-b24g30",
        runtime_root=tmp_path,
        decision_predicate="neutral_package_identity_terminal_descendant_v1",
        required_generations=[0, 5, 10, 15, 20, 25, 30],
        plan_sha256="a" * 64,
    )

    assert manifest["schema_version"] == "campaign_audit_hold_manifest_v1"
    assert manifest["audit_hold_status"] == AUDIT_HOLD_ACTIVE
    assert manifest["required_generations"] == [0, 5, 10, 15, 20, 25, 30]
    assert any(
        rule["retention_class"] == CLASS_A
        and "bundle_gen000030" in rule["relative_glob"]
        for rule in manifest["protected_path_rules"]
    )
    assert len(manifest["manifest_sha256"]) == 64


def test_load_manifest_rejects_digest_drift(tmp_path: Path) -> None:
    manifest = build_audit_hold_manifest(
        campaign_id="campaign",
        runtime_root=tmp_path,
        decision_predicate="predicate",
        required_generations=[0, 30],
    )
    path = write_audit_hold_manifest(tmp_path / "manifest.json", manifest)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    loaded["campaign_id"] = "changed"
    path.write_text(json.dumps(loaded), encoding="utf-8")

    with pytest.raises(DecisionEvidenceRetentionError, match="digest mismatch"):
        load_manifest(path)


def test_protected_projection_inside_selected_bundle_blocks_prune(tmp_path: Path) -> None:
    protected = (
        tmp_path
        / "batches"
        / "batch-0000"
        / "checkpoint_bundles_v2"
        / "cell-a"
        / "bundle_gen000005"
        / "native_runtime_checkpoint_projection.json.gz"
    )
    _projection(protected)
    (protected.parent / "runner_state.json").write_text("{}\n", encoding="utf-8")
    final = (
        tmp_path
        / "batches"
        / "batch-0000"
        / "checkpoint_bundles_v2"
        / "cell-a"
        / "bundle_gen000030"
        / "native_runtime_checkpoint_projection.json.gz"
    )
    _projection(final)
    manifest = build_audit_hold_manifest(
        campaign_id="campaign",
        runtime_root=tmp_path,
        decision_predicate="predicate",
        required_generations=[0, 5, 10, 15, 20, 25, 30],
    )
    plan = build_prune_plan(tmp_path, {"intermediate_checkpoints"})

    with pytest.raises(DecisionEvidenceRetentionError, match="Class A") as excinfo:
        guard_prune_plan(root=tmp_path, plan=plan, manifest=manifest)

    assert excinfo.value.receipt["decision"] == "blocked"
    assert excinfo.value.receipt["protected_attempt_count"] >= 1
    protected_rows = protected_entries_for_selection(
        root=tmp_path,
        selected_path=protected.parent,
        manifest=manifest,
    )
    assert protected_rows[0]["path"].endswith("native_runtime_checkpoint_projection.json.gz")


def test_non_checkpoint_debug_trace_is_not_class_a_blocked(tmp_path: Path) -> None:
    trace = tmp_path / "cells" / "cell-a" / "debug_trace.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text("{}\n", encoding="utf-8")
    manifest = build_audit_hold_manifest(
        campaign_id="campaign",
        runtime_root=tmp_path,
        decision_predicate="predicate",
        required_generations=[0, 30],
    )
    plan = build_prune_plan(tmp_path, {"debug_traces"})

    receipt = guard_prune_plan(root=tmp_path, plan=plan, manifest=manifest)

    assert receipt["decision"] == "allowed"
    assert receipt["protected_attempt_count"] == 0


def test_class_b_debug_log_blocks_until_reducer_closeout(tmp_path: Path) -> None:
    debug_log = tmp_path / "cells" / "cell-a" / "debug.log"
    debug_log.parent.mkdir(parents=True)
    debug_log.write_text("exclusion detail\n", encoding="utf-8")
    plan = {"selected": [{"path": "cells/cell-a/debug.log"}]}

    pending = build_audit_hold_manifest(
        campaign_id="campaign",
        runtime_root=tmp_path,
        decision_predicate="predicate",
        required_generations=[0, 30],
        reducer_closeout_status="pending",
    )
    with pytest.raises(DecisionEvidenceRetentionError, match="Class B") as excinfo:
        guard_prune_plan(root=tmp_path, plan=plan, manifest=pending)
    assert excinfo.value.receipt["reducer_closeout_status"] == "pending"

    released = build_audit_hold_manifest(
        campaign_id="campaign",
        runtime_root=tmp_path,
        decision_predicate="predicate",
        required_generations=[0, 30],
        reducer_closeout_status="released",
        reducer_release_receipt_sha256="b" * 64,
    )
    receipt = guard_prune_plan(root=tmp_path, plan=plan, manifest=released)
    assert receipt["decision"] == "allowed"
