from __future__ import annotations

import json
from pathlib import Path

from tools.scripts import p12b_compact_terminal_closure_v1 as closure


def _write(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _plan() -> dict[str, object]:
    return {
        "comparisons": {
            "branches": [
                {"branch_id": "repeat-1", "kind": "fresh_repeat"},
                {
                    "branch_id": "resume-10",
                    "kind": "checkpoint_continuation",
                    "split_generation": 10,
                },
                {
                    "branch_id": "resume-20",
                    "kind": "checkpoint_continuation",
                    "split_generation": 20,
                },
                {"branch_id": "reporting-1", "kind": "reporting_parity"},
            ]
        }
    }


def test_comparison_integrity_accepts_completed_manifest_and_reporting_receipt(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "comparisons_manifest.json",
        {
            "status": "completed",
            "counts": {
                "completed": 4,
                "failed": 0,
                "interrupted": 0,
                "pending": 0,
                "running": 0,
            },
            "branches": [
                {"branch_id": "repeat-1", "kind": "fresh_repeat", "status": "completed"},
                {
                    "branch_id": "resume-10",
                    "kind": "checkpoint_continuation",
                    "status": "completed",
                },
                {
                    "branch_id": "resume-20",
                    "kind": "checkpoint_continuation",
                    "status": "completed",
                },
                {
                    "branch_id": "reporting-1",
                    "kind": "reporting_parity",
                    "status": "completed",
                },
            ],
        },
    )
    _write(
        tmp_path / "reporting_parity_receipt.json",
        {
            "qualification_evidence_status": "reporting_parity_proven",
            "status": "supported",
            "evidence_mode": "enabled_disabled_exact_terminal_state",
        },
    )

    result = closure._comparison_integrity(run_root=tmp_path, plan=_plan())

    assert result["ok"] is True
    assert result["failures"] == []


def test_comparison_integrity_fails_closed_on_missing_branch_and_reporting_gap(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "comparisons_manifest.json",
        {
            "status": "failed",
            "counts": {
                "completed": 3,
                "failed": 1,
                "interrupted": 0,
                "pending": 0,
                "running": 0,
            },
            "branches": [
                {"branch_id": "repeat-1", "kind": "fresh_repeat", "status": "completed"}
            ],
        },
    )
    _write(
        tmp_path / "reporting_parity_receipt.json",
        {
            "qualification_evidence_status": "reporting_parity_unsupported",
            "status": "unsupported",
        },
    )

    result = closure._comparison_integrity(run_root=tmp_path, plan=_plan())

    assert result["ok"] is False
    assert "comparisons_manifest_status:failed" in result["failures"]
    assert "comparisons_manifest_failed:1" in result["failures"]
    assert "comparison_branch_identity_mismatch" in result["failures"]
    assert (
        "reporting_parity_not_proven:reporting_parity_unsupported"
        in result["failures"]
    )
