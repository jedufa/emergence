#!/usr/bin/env python3
"""Build the authoritative RECOMB-1 inheritance-surface audit.

The detailed scalar/policy and structural audits are deliberately retained as
separate reviewable shards.  This helper joins them, verifies that their
coverage gates are closed, and emits one canonical machine-readable artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"audit shard must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def build(*, scalar_path: Path, structural_path: Path, plan_path: Path) -> dict[str, Any]:
    scalar = _load(scalar_path)
    structural = _load(structural_path)
    plan = _load(plan_path)

    scalar_surfaces = list(scalar.get("hereditary_surfaces", []))
    structural_surfaces = list(structural.get("surfaces", []))
    coverage = dict(scalar.get("coverage_assertion", {}))
    unknown_count = int(coverage.get("unknown_current_surface_count", -1))
    if unknown_count != 0:
        raise ValueError(f"unknown scalar/policy production surfaces remain: {unknown_count}")
    if not scalar_surfaces or not structural_surfaces:
        raise ValueError("both scalar/policy and structural surfaces are required")
    if str(plan.get("audit_decision", "")) != "fixture_implementation_required_before_scientific_decision":
        raise ValueError("fixture-plan audit gate is missing or changed")
    if not bool(structural.get("audit_outcome", {}).get("status") == "complete_for_current_surface_classification"):
        raise ValueError("structural surface classification is incomplete")

    return {
        "schema_version": "recomb1_inheritance_surface_audit_v1",
        "audit_id": "RECOMB-1-inheritance-surface-audit-20260718",
        "scope": {
            "cpu_reference_only": True,
            "cuda_or_accelerated_work_started": False,
            "production_source_edited_during_audit": False,
            "question": (
                "Whether current inheritance materially removes marginal variation "
                "or breaks coordinated hereditary packages."
            ),
        },
        "source_shards": [
            {"path": str(scalar_path), "sha256": _sha256(scalar_path)},
            {"path": str(structural_path), "sha256": _sha256(structural_path)},
            {"path": str(plan_path), "sha256": _sha256(plan_path)},
        ],
        "coverage": {
            **coverage,
            "scalar_policy_surface_count": len(scalar_surfaces),
            "structural_surface_count": len(structural_surfaces),
            "total_surface_family_count": len(scalar_surfaces) + len(structural_surfaces),
            "unknown_production_surface_count": 0,
            "classification_gate_passed": True,
        },
        "pipeline_order": scalar.get("pipeline_order", []),
        "scalar_and_policy_surfaces": scalar_surfaces,
        "structural_surfaces": structural_surfaces,
        "nonhereditary_or_regenerated_genome_fields": scalar.get(
            "nonhereditary_or_regenerated_genome_fields", []
        ),
        "runtime_state_exclusions": scalar.get("runtime_state_exclusions", []),
        "rng_contract": scalar.get("rng_contract", {}),
        "checkpoint_contract": scalar.get("checkpoint_contract", {}),
        "scientific_ambiguities": scalar.get("precise_ambiguities", []),
        "variance_risk_findings": scalar.get("variance_risk_findings", []),
        "structural_cross_surface_findings": structural.get("cross_surface_findings", {}),
        "frozen_existing_draw_schedule": structural.get("frozen_draw_schedule", {}),
        "existing_evidence_coverage": plan.get("existing_test_and_evidence_coverage", []),
        "fixture_matrix_preregistration": plan.get("smallest_deterministic_fixture_matrix", {}),
        "decision_gate": plan.get("decision_gate", {}),
        "audit_conclusion": {
            "status": "complete_before_measurement",
            "production_decision": "not_yet_frozen",
            "reason": (
                "No production field remains unknown, but the deterministic fixture matrix "
                "must measure variance and package loss before mechanics change."
            ),
            "broad_scientific_claim_supported": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scalar-audit", required=True, type=Path)
    parser.add_argument("--structural-audit", required=True, type=Path)
    parser.add_argument("--fixture-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = build(
        scalar_path=args.scalar_audit,
        structural_path=args.structural_audit,
        plan_path=args.fixture_plan,
    )
    _write_json(args.output, payload)


if __name__ == "__main__":
    main()
