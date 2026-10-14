"""Small fixtures for the evidence-entailing review acceptance contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.actors.common import file_sha256
from tools.actors.portfolio import session_store
from tools.actors.work_package import load_work_package


def write_review_proof_for_session(
    state: Path,
    session_id: str,
    *,
    limitations: list[str] | None = None,
) -> Path:
    """Create a complete v2 proof for the package bound to a review Session."""

    session = session_store(state).get(session_id)
    payload = dict(session.get("payload") or {})
    target = dict(payload.get("review_target") or {})
    binding = dict(payload.get("work_package_binding") or {})
    repo_root = Path(str(dict(payload.get("repository") or {}).get("repo_root") or ""))
    package = load_work_package(repo_root / str(binding["path"]))
    proof_dir = state / "review-proof-inputs"
    proof_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = proof_dir / f"{session_id}-evidence.txt"
    evidence_path.write_text("focused review evidence\n", encoding="utf-8")
    evidence_sha = file_sha256(evidence_path)

    def ref() -> dict[str, str]:
        return {"uri": str(evidence_path), "sha256": evidence_sha}

    execution = {
        "schema_version": "eiel_review_execution_receipt_v1",
        "execution_mode": "REVIEWER_EXECUTED",
        "reviewed_commit": str(target["reviewed_commit"]),
        "command_or_source": "focused review fixture",
        "status": "PASS",
        "output_sha256": evidence_sha,
        "executor_id": str(payload.get("current_codex_id") or "fixture-reviewer"),
        "executed_at_utc": "2026-08-08T00:00:00Z",
        "environment": {"fixture": True},
        "review_session_id": session_id,
        "evidence_relation": "EXECUTED_IN_REVIEW",
        "source_execution_id": None,
    }
    results: list[dict[str, Any]] = []
    for obligation in package.get("acceptance_criteria") or []:
        results.append({
            "kind": "acceptance_criterion",
            "obligation": str(obligation),
            "status": "PASS",
            "evidence_refs": [ref()],
            "execution_receipt": None,
        })
    for kind, field in (("test", "tests"), ("evidence_item", "evidence")):
        for obligation in package.get(field) or []:
            results.append({
                "kind": kind,
                "obligation": str(obligation),
                "status": "PASS",
                "evidence_refs": [ref()],
                "execution_receipt": dict(execution),
            })
    proof = {
        "schema_version": "eiel_review_obligation_proof_v2",
        "work_id": str(payload["work_id"]),
        "reviewed_commit": str(target["reviewed_commit"]),
        "reviewed_package_sha256": str(target.get("reviewed_package_sha256") or binding["sha256"]),
        "implementation_sufficiency": "SUFFICIENT",
        "obligation_results": results,
        "limitations": list(limitations or []),
    }
    path = proof_dir / f"{session_id}.json"
    path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
