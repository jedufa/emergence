"""Curated historical benchmark for the read-only development preparation loop."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from .core import (
    DevAutomationError,
    atomic_write_json,
    build_context,
    canonical_sha256,
    find_project_root,
    resolve_state_root,
)


MANIFEST_VERSION = "emergence_dev_benchmark_manifest_v2"
RECEIPT_VERSION = "emergence_dev_benchmark_receipt_v2"


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DevAutomationError(f"cannot load benchmark manifest: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != MANIFEST_VERSION:
        raise DevAutomationError(f"benchmark schema_version must be {MANIFEST_VERSION!r}")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 10:
        raise DevAutomationError("benchmark manifest must contain exactly ten cases")
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise DevAutomationError(f"benchmark case {index} must be an object")
        if not isinstance(case.get("id"), str) or not isinstance(case.get("objective"), str):
            raise DevAutomationError(f"benchmark case {index} requires id and objective")
        required = case.get("required_paths")
        if not isinstance(required, list) or not required or not all(isinstance(item, str) for item in required):
            raise DevAutomationError(f"benchmark case {index} requires non-empty required_paths")
        baseline = case.get("baseline_discovery_operations")
        if isinstance(baseline, bool) or not isinstance(baseline, int) or baseline < 1:
            raise DevAutomationError(f"benchmark case {index} has invalid baseline operations")
        for field in ("historical_reference", "baseline_derivation", "relationship_class", "overrides"):
            if field not in case:
                raise DevAutomationError(f"benchmark case {index} requires {field}")
        derivation = case["baseline_derivation"]
        if (
            not isinstance(derivation, dict)
            or not isinstance(derivation.get("components"), list)
            or sum(
                item.get("operations", 0)
                for item in derivation["components"]
                if isinstance(item, dict)
            )
            != baseline
        ):
            raise DevAutomationError(
                f"benchmark case {index} baseline components must sum to baseline operations"
            )
        if not isinstance(case["overrides"], list):
            raise DevAutomationError(f"benchmark case {index} overrides must be an array")
    return payload


def run_benchmark(
    manifest_path: str | Path,
    *,
    project_root: str | Path | None = None,
    state_root: str | Path | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    root = find_project_root(project_root)
    state = resolve_state_root(state_root)
    manifest = _load_manifest(Path(manifest_path))
    rows: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        context = build_context(
            case["objective"],
            project_root=root,
            state_root=state,
        )
        observed = {row["path"] for row in context["candidates"]}
        observed.update(row["path"] for row in context["suggested_tests"])
        for relationship in context["relationships"]:
            source = relationship.get("source")
            if isinstance(source, str):
                observed.add(source)
            observed.update(
                item for item in relationship.get("counterparts", []) if isinstance(item, str)
            )
        missing = sorted(set(case["required_paths"]) - observed)
        observed_critical = {row["id"] for row in context["critical_obligations"]}
        missing_critical = sorted(set(case.get("required_critical_rules", [])) - observed_critical)
        baseline = case["baseline_discovery_operations"]
        automated = 1
        reduction = 100.0 * (baseline - automated) / baseline
        rows.append(
            {
                "id": case["id"],
                "objective": case["objective"],
                "required_paths": case["required_paths"],
                "observed_paths": sorted(observed),
                "missing_required_paths": missing,
                "missing_critical_rules": missing_critical,
                "integrity_passed": not missing and not missing_critical,
                "context_tokens": context["estimated_context_tokens"],
                "context_budget": context["context_budget"],
                "context_budget_passed": (
                    context["estimated_context_tokens"]
                    <= context["context_budget"]["internal_tokens"]
                ),
                "historical_reference": case["historical_reference"],
                "baseline_derivation": case["baseline_derivation"],
                "relationship_class": case["relationship_class"],
                "overrides": case["overrides"],
                "baseline_discovery_operations": baseline,
                "automated_discovery_operations": automated,
                "modeled_discovery_reduction_percent": round(reduction, 2),
            }
        )
    median_reduction = statistics.median(
        row["modeled_discovery_reduction_percent"] for row in rows
    )
    integrity = all(row["integrity_passed"] for row in rows)
    context_budget = all(row["context_budget_passed"] for row in rows)
    efficiency = median_reduction >= 30.0
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_VERSION,
        "manifest_sha256": canonical_sha256(manifest),
        "project_root": str(root),
        "cases": rows,
        "integrity_passed": integrity,
        "context_budget_passed": context_budget,
        "median_modeled_discovery_reduction_percent": median_reduction,
        "metric_definition": (
            "Modeled reduction from manually derived historical discovery operations "
            "to one deterministic prepare invocation; not measured wall time or token use."
        ),
        "efficiency_passed": efficiency,
        "go_no_go": "go" if integrity and context_budget and efficiency else "no-go",
        "codemod_stage_authorized": bool(integrity and context_budget and efficiency),
    }
    receipt["receipt_hash_scope"] = "canonical receipt fields excluding project_root"
    receipt["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "project_root"}
    )
    if output is not None:
        atomic_write_json(Path(output).resolve(), receipt)
    return receipt
