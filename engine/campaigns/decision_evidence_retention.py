"""Decision-bearing evidence retention and audit-hold guards."""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from engine.campaigns.canonical import canonical_sha256, pretty_json


SCHEMA_VERSION = "decision_bearing_evidence_retention_schema_v1"
MANIFEST_SCHEMA_VERSION = "campaign_audit_hold_manifest_v1"
RELEASE_SCHEMA_VERSION = "audit_hold_release_receipt_v1"
PROTECTED_ATTEMPT_SCHEMA_VERSION = "protected_prune_attempt_receipt_v1"

AUDIT_HOLD_ACTIVE = "AUDIT_HOLD_ACTIVE"
AUDIT_HOLD_RELEASED = "AUDIT_HOLD_RELEASED"

CLASS_A = "A_IMMUTABLE_UNTIL_INDEPENDENT_AUDIT_CLOSEOUT"
CLASS_B = "B_RETAIN_UNTIL_REDUCER_CLOSEOUT"
CLASS_C = "C_PRUNABLE_AFTER_COMPACT_REPLACEMENT_VERIFICATION"


class DecisionEvidenceRetentionError(RuntimeError):
    """Decision-bearing evidence retention rules were violated."""

    def __init__(self, message: str, *, receipt: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.receipt = dict(receipt or {})


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(pretty_json(value), encoding="utf-8")
    temporary.replace(path)


def retention_schema() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "classes": {
            CLASS_A: [
                "G0 native checkpoint projections",
                "decision-bearing segment projections",
                "milestone and terminal projections",
                "paired off-control projections",
                "event and causal transaction receipts",
                "segment-chain receipts",
                "route/package/artifact/source authority receipts",
                "logical-cell mapping and checkpoint catalogs",
                "mutation event identities",
                "terminal carrier state evidence",
                "reducer inputs",
                "independent audit inputs",
            ],
            CLASS_B: [
                "expanded diagnostics",
                "intermediate metrics",
                "preparation projections",
                "debug logs required to explain exclusions",
            ],
            CLASS_C: [
                "caches",
                "duplicate build payloads",
                "temporary source projections not used by decision predicates",
                "redundant logs",
                "transient compiler/build outputs",
            ],
        },
        "audit_hold_release_requirements": [
            "reducer_complete",
            "sample_manifest_frozen",
            "independent_audit_reconstructs_required_rows",
            "predicate_equivalence_established",
            "audit_decision_committed",
            "compact_audit_archive_verified",
            "release_receipt_written",
        ],
    }


def _generation_globs(generations: Sequence[int]) -> list[str]:
    result: list[str] = []
    for generation in sorted({int(value) for value in generations}):
        bundle = f"bundle_gen{generation:06d}"
        result.extend(
            [
                f"batches/*/checkpoint_bundles_v2/*/{bundle}/native_runtime_checkpoint_projection.json",
                f"batches/*/checkpoint_bundles_v2/*/{bundle}/native_runtime_checkpoint_projection.json.gz",
                f"checkpoint_bundles_v2/*/{bundle}/native_runtime_checkpoint_projection.json",
                f"checkpoint_bundles_v2/*/{bundle}/native_runtime_checkpoint_projection.json.gz",
                f"**/{bundle}/native_runtime_checkpoint_projection.json",
                f"**/{bundle}/native_runtime_checkpoint_projection.json.gz",
            ]
        )
    return result


def build_audit_hold_manifest(
    *,
    campaign_id: str,
    runtime_root: Path,
    decision_predicate: str,
    required_generations: Sequence[int],
    plan_sha256: str | None = None,
    treatment_control_pairs: Sequence[Mapping[str, Any]] = (),
    route_bindings: Mapping[str, Any] | None = None,
    reducer_closeout_status: str = "pending",
    reducer_release_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    if not campaign_id:
        raise DecisionEvidenceRetentionError("campaign_id is required")
    root = runtime_root.expanduser().resolve()
    generations = sorted({int(value) for value in required_generations})
    if not generations or generations[0] < 0:
        raise DecisionEvidenceRetentionError("required generations are invalid")
    if reducer_closeout_status not in {"pending", "released"}:
        raise DecisionEvidenceRetentionError(
            "reducer_closeout_status must be pending or released"
        )
    if reducer_closeout_status == "released" and (
        not isinstance(reducer_release_receipt_sha256, str)
        or len(reducer_release_receipt_sha256) != 64
        or any(char not in "0123456789abcdef" for char in reducer_release_receipt_sha256)
    ):
        raise DecisionEvidenceRetentionError(
            "released reducer closeout requires a lowercase SHA-256 receipt"
        )
    class_a_rules = [
        *(
            {
                "rule_id": f"native_projection_g{generation:06d}_{index}",
                "retention_class": CLASS_A,
                "relative_glob": pattern,
                "reason": "decision-bearing native checkpoint projection",
            }
            for index, pattern in enumerate(_generation_globs(generations))
            for generation in [int(pattern.split("bundle_gen", 1)[1][:6])]
        ),
        {
            "rule_id": "execution_plan",
            "retention_class": CLASS_A,
            "relative_glob": "execution_plan.json",
            "reason": "logical-cell mapping and plan identity",
        },
        {
            "rule_id": "batch_manifest",
            "retention_class": CLASS_A,
            "relative_glob": "batches/*/manifest.json",
            "reason": "batch and checkpoint catalog",
        },
        {
            "rule_id": "authority_receipts",
            "retention_class": CLASS_A,
            "relative_glob": "**/*authority*.json",
            "reason": "route/package/artifact/source authority",
        },
        {
            "rule_id": "decision_receipts",
            "retention_class": CLASS_A,
            "relative_glob": "**/*receipt*.json",
            "reason": "event, segment, route, package, or audit receipt",
        },
        {
            "rule_id": "reducer_inputs",
            "retention_class": CLASS_A,
            "relative_glob": "**/*classified_rows*.json*",
            "reason": "reducer or audit input rows",
        },
        {
            "rule_id": "audit_inputs",
            "retention_class": CLASS_A,
            "relative_glob": "**/*audit*sample*manifest*.json",
            "reason": "independent audit sample freeze",
        },
    ]
    class_b_rules = [
        {
            "rule_id": "exclusion_debug_logs",
            "retention_class": CLASS_B,
            "relative_glob": "**/*debug*.log",
            "reason": "debug log may explain exclusions until reducer closeout",
        },
        {
            "rule_id": "intermediate_metrics",
            "retention_class": CLASS_B,
            "relative_glob": "**/*metrics*.csv",
            "reason": "intermediate metrics retained until reducer closeout",
        },
    ]
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "plan_sha256": plan_sha256,
        "runtime_root": str(root),
        "decision_predicate": decision_predicate,
        "required_generations": generations,
        "treatment_control_pairs": list(treatment_control_pairs),
        "route_bindings": dict(route_bindings or {}),
        "retention_schema": retention_schema(),
        "audit_hold_status": AUDIT_HOLD_ACTIVE,
        "reducer_closeout_status": reducer_closeout_status,
        "reducer_release_receipt_sha256": reducer_release_receipt_sha256,
        "retention_status": "protected_until_audit_release",
        "prune_eligibility": "fail_closed_for_class_a_and_b_until_release",
        "protected_path_rules": class_a_rules + class_b_rules,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def write_audit_hold_manifest(path: Path, manifest: Mapping[str, Any]) -> Path:
    _atomic_json(path.expanduser().resolve(), dict(manifest))
    return path.expanduser().resolve()


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DecisionEvidenceRetentionError(f"cannot read retention manifest: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise DecisionEvidenceRetentionError("retention manifest schema mismatch")
    expected = value.get("manifest_sha256")
    if isinstance(expected, str):
        payload = dict(value)
        payload.pop("manifest_sha256", None)
        actual = canonical_sha256(payload)
        if actual != expected:
            raise DecisionEvidenceRetentionError("retention manifest digest mismatch")
    return value


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def classify_relative_path(relative_path: str, manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    rules = manifest.get("protected_path_rules")
    if not isinstance(rules, list):
        return None
    rel = relative_path.replace("\\", "/")
    for rule in rules:
        if not isinstance(rule, Mapping):
            continue
        pattern = str(rule.get("relative_glob") or "")
        if pattern and fnmatch.fnmatch(rel, pattern):
            return dict(rule)
    return None


def protected_entries_for_selection(
    *,
    root: Path,
    selected_path: Path,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    root = root.resolve()
    selected_path = selected_path.resolve()
    try:
        selected_path.relative_to(root)
    except ValueError as exc:
        raise DecisionEvidenceRetentionError(f"selected path escapes root: {selected_path}") from exc
    candidates = (
        [path for path in selected_path.rglob("*") if path.is_file()]
        if selected_path.is_dir()
        else [selected_path]
    )
    protected: list[dict[str, Any]] = []
    for path in candidates:
        if not path.exists():
            continue
        rel = _relative(path, root)
        rule = classify_relative_path(rel, manifest)
        if rule and rule.get("retention_class") in {
            CLASS_A,
            CLASS_B,
        }:
            protected.append(
                {
                    "path": rel,
                    "retention_class": rule.get("retention_class"),
                    "rule_id": rule.get("rule_id"),
                    "reason": rule.get("reason"),
                }
            )
    return protected


def guard_prune_plan(
    *,
    root: Path,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    status = manifest.get("audit_hold_status")
    reducer_status = manifest.get("reducer_closeout_status", "pending")
    selected = plan.get("selected")
    if not isinstance(selected, list):
        raise DecisionEvidenceRetentionError("prune plan lacks selected rows")
    protected: list[dict[str, Any]] = []
    for row in selected:
        if not isinstance(row, Mapping):
            continue
        rel = str(row.get("path") or "").strip()
        if not rel:
            continue
        protected.extend(
            protected_entries_for_selection(
                root=root,
                selected_path=root / rel,
                manifest=manifest,
            )
        )
    receipt = {
        "schema_version": PROTECTED_ATTEMPT_SCHEMA_VERSION,
        "campaign_id": manifest.get("campaign_id"),
        "runtime_root": str(root.expanduser().resolve()),
        "audit_hold_status": status,
        "reducer_closeout_status": reducer_status,
        "selected_count": len(selected),
        "protected_attempt_count": len(protected),
        "protected_attempts": protected,
        "decision": (
            "blocked"
            if (
                any(
                    row.get("retention_class") == CLASS_A
                    and status == AUDIT_HOLD_ACTIVE
                    for row in protected
                )
                or any(
                    row.get("retention_class") == CLASS_B
                    and reducer_status != "released"
                    for row in protected
                )
            )
            else "allowed"
        ),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    class_a_protected = [
        row for row in protected if row.get("retention_class") == CLASS_A
    ]
    if class_a_protected and status == AUDIT_HOLD_ACTIVE:
        raise DecisionEvidenceRetentionError(
            "prune plan selects Class A decision-bearing evidence under active audit hold",
            receipt=receipt,
        )
    class_b_protected = [
        row for row in protected if row.get("retention_class") == CLASS_B
    ]
    if class_b_protected and reducer_status != "released":
        raise DecisionEvidenceRetentionError(
            "prune plan selects Class B evidence before reducer closeout",
            receipt=receipt,
        )
    return receipt


def release_audit_hold(
    *,
    manifest: Mapping[str, Any],
    audit_decision: Mapping[str, Any],
    compact_archive_sha256: str,
) -> dict[str, Any]:
    required_flags = {
        "positive_rows_reproduced": audit_decision.get("positive_rows_reproduced") is True,
        "sampled_negatives_agree": audit_decision.get("sampled_negatives_agree") is True,
        "compact_archive_verified": bool(compact_archive_sha256),
    }
    ok = all(required_flags.values()) and audit_decision.get("decision") not in {
        "B48-AUDIT-E",
        "GV2-B48-R1-E",
    }
    receipt = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "campaign_id": manifest.get("campaign_id"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "audit_decision": audit_decision.get("decision"),
        "audit_decision_sha256": canonical_sha256(audit_decision),
        "compact_archive_sha256": compact_archive_sha256,
        "release_requirements": required_flags,
        "audit_hold_status_after": AUDIT_HOLD_RELEASED if ok else AUDIT_HOLD_ACTIVE,
        "release_allowed": ok,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    if not ok:
        raise DecisionEvidenceRetentionError("audit hold release requirements are not met")
    return receipt
