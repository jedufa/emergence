from __future__ import annotations

import json
from pathlib import Path

from tools.scripts import p12_dense_id_surface_audit_v1 as audit


ROOT = Path(__file__).resolve().parents[1]


def _surfaces(payload: dict) -> list[dict]:
    return list(payload["surfaces"])


def test_dense_id_audit_detects_curated_surface_classes(tmp_path: Path) -> None:
    source = tmp_path / "src"
    (source / "engine/perf/native").mkdir(parents=True)
    (source / "engine/simulation").mkdir(parents=True)
    (source / "tools/scripts").mkdir(parents=True)
    (source / "tests").mkdir(parents=True)
    (source / "engine/perf/native/example.cpp").write_text(
        "\n".join(
            [
                "agent_id_to_slot[cell_id_map_offset + child_id] = slot;",
                "TORCH_CHECK(agent_grid.scalar_type() == torch::kInt32);",
                "slot_to_agent_id.scalar_type() == torch::kInt64;",
            ]
        ),
        encoding="utf-8",
    )
    (source / "engine/simulation/checkpoint_bundle_v2.py").write_text(
        "destination[slot] = source[agent_id]\ncache[agent_id] = ['x']\n",
        encoding="utf-8",
    )
    (source / "tools/scripts/wdl_report.py").write_text(
        "wdl_exposure_by_lineage[agent_id] = cursor\n",
        encoding="utf-8",
    )

    payload = audit.build_dense_id_surface_audit_v1(
        source_root=source,
        source_commit="test",
    )
    statuses = {row["status"] for row in _surfaces(payload)}
    authority_categories = {row["authority_category"] for row in _surfaces(payload)}

    assert payload["decision"] == audit.DECISION
    assert payload["authority_decision"] == audit.AUTHORITY_DECISION
    assert "unsafe_or_unqualified_dense_index" in statuses
    assert "needs_public_identity_decision" in statuses
    assert "accepted_sparse_identity_evidence" in statuses
    assert "needs_checkpoint_migration_or_rejection" in statuses
    assert "needs_compact_reporting_capacity_contract" in statuses
    assert "production mutable-state authority" in authority_categories
    assert "production identity authority" in authority_categories
    assert "checkpoint authority" in authority_categories
    assert "reporting/Chronicle authority" in authority_categories
    assert payload["authoritative_defect_file_count"] >= 3


def test_live_dense_id_audit_contains_current_p12_blockers() -> None:
    payload = audit.build_dense_id_surface_audit_v1(
        source_root=ROOT,
        source_commit="live",
    )
    rows = _surfaces(payload)

    assert payload["surface_count"] >= 20
    assert payload["raw_identity_file_count"] >= 300
    assert payload["authoritative_defect_file_count"] >= 1
    assert payload["compatibility_debt_file_count"] >= 1
    assert payload["audit_sha256"]
    assert any(
        row["surface_id"] == "dense_agent_id_to_slot_index"
        and row["domain"] == "cuda"
        for row in rows
    )
    assert any(
        row["surface_id"] == "accepted_sparse_identity_contract"
        and row["path"] == "engine/native_runtime/patch_public_reporting.py"
        for row in rows
    )
    assert any(
        row["surface_id"] == "checkpoint_dense_id_reconstruction"
        and row["path"] == "engine/simulation/checkpoint_bundle_v2.py"
        for row in rows
    )


def test_dense_id_audit_cli_writes_canonical_outputs(tmp_path: Path) -> None:
    output = tmp_path / "out"
    result = audit.build_outputs(
        audit.argparse.Namespace(
            source_root=str(ROOT),
            source_commit="cli-test",
            output_dir=str(output),
            json_name="audit.json",
            report_name="report.md",
        )
    )

    payload = json.loads((output / "audit.json").read_text(encoding="ascii"))
    report = (output / "report.md").read_text(encoding="ascii")
    assert result["ok"] is True
    assert payload["audit_sha256"] == result["audit_sha256"]
    assert "High-Risk Surface Table" in report
    assert "Authority Triage" in report
    assert str(output) not in report
    assert result["json_sha256"]
    assert result["report_sha256"]
