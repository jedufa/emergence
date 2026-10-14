#!/usr/bin/env python3
"""Read-only, hash-anchored comparison for the bounded paired CPU run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Mapping

from .analysis import analyze_campaign
from .artifact_lifecycle import (
    ArtifactLifecycleError,
    validate_artifact_disposition_manifest_v1,
)
from .canonical import canonical_sha256
from .execution_front_door_v1 import (
    ExecutionFrontDoorError,
    ExecutionRequestV2,
    verify_mechanical_execution_v1,
)


class PairedUsageError(ValueError):
    """The supplied run does not match its trusted anchors or protocol."""


_PAIR_PARAM = "env_cfg.reproduce_try_prob"
_CORE_METRICS = (
    "population.alive",
    "population.births",
    "population.deaths",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PairedUsageError(message)


def _check_trusted_anchor(actual: str, expected: str, label: str) -> None:
    _require(actual == expected, f"{label} digest does not match trusted anchor")


def _paired_metric_summary(
    rows: list[dict[str, Any]], left_id: str, right_id: str
) -> dict[str, Any]:
    by_cell = {str(row["cell_id"]): row for row in rows}
    _require(len(by_cell) == len(rows) and left_id in by_cell and right_id in by_cell, "producer observations do not contain exactly the paired cells")
    measured: dict[str, Any] = {}
    for metric in _CORE_METRICS:
        left_row, right_row = by_cell[left_id], by_cell[right_id]
        left_present, right_present = metric in left_row, metric in right_row
        lvalue, rvalue = left_row.get(metric), right_row.get(metric)
        delta = None
        if (
            left_present
            and right_present
            and isinstance(lvalue, (int, float))
            and not isinstance(lvalue, bool)
            and isinstance(rvalue, (int, float))
            and not isinstance(rvalue, bool)
        ):
            delta = rvalue - lvalue
        measured[metric] = {
            "control": {"present": left_present, "value": lvalue},
            "treatment": {"present": right_present, "value": rvalue},
            "treatment_minus_control": delta,
        }
    return measured


def _check_branch_identity(
    branch: Mapping[str, Any],
    branch_manifest: Mapping[str, Any],
    *,
    campaign_id: str,
    plan_sha256: str,
) -> str:
    branch_id = branch.get("branch_id")
    _require(isinstance(branch_id, str) and branch_id, "comparison branch ID is missing")
    _require(branch_manifest.get("branch_id") == branch_id, "comparison branch manifest ID mismatch")
    _require(branch_manifest.get("campaign_id") == campaign_id and branch_manifest.get("plan_sha256") == plan_sha256, "comparison branch campaign/plan mismatch")
    _require(branch_manifest.get("kind") == branch.get("kind") and branch_manifest.get("status") == "completed", "comparison branch kind/status mismatch")
    return branch_id


def _check_comparison_match(
    comparison: Any,
    *,
    surfaces: set[str],
    label: str,
) -> Mapping[str, Any]:
    _require(isinstance(comparison, Mapping) and comparison.get("match") is True, f"{label} comparison did not match")
    _require(not comparison.get("mismatches") and not comparison.get("missing_surfaces"), f"{label} comparison has mismatch or missing surfaces")
    _require(surfaces <= set(comparison.get("compared_surfaces", [])), f"{label} omits required comparison surfaces")
    if "metrics_rows_sha256" in surfaces or "terminal_checkpoint_digest" in surfaces:
        _require(comparison.get("baseline") == comparison.get("candidate"), f"{label} baseline and candidate observations differ")
    return comparison


def _read_json(root: Path, relative: str) -> dict[str, Any]:
    path = PurePosixPath(relative)
    _require(not path.is_absolute() and ".." not in path.parts, "unsafe run-relative path")
    candidate = root.joinpath(*path.parts)
    _require(not candidate.is_symlink(), f"symlink is not accepted: {relative}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise PairedUsageError(f"cannot read JSON artifact {relative}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON artifact is not an object: {relative}")
    return value


def _file_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _sealed_hash(value: Mapping[str, Any], field: str, label: str) -> str:
    supplied = value.get(field)
    core = dict(value)
    core.pop(field, None)
    _require(
        isinstance(supplied, str) and supplied == canonical_sha256(core),
        f"{label} canonical seal mismatch",
    )
    return supplied


def _check_pair_plan(plan: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    core = dict(plan)
    declared = core.pop("plan_sha256", None)
    _require(
        isinstance(declared, str) and declared == canonical_sha256(core),
        "compiled plan canonical hash mismatch",
    )
    execution = plan.get("execution")
    _require(isinstance(execution, Mapping), "plan execution settings are missing")
    _require(
        execution.get("generations") == 6
        and execution.get("ticks_per_generation") == 160
        and execution.get("checkpoints") == [3, 6]
        and isinstance(execution.get("tick_schedule"), Mapping)
        and execution["tick_schedule"].get("total_ticks") == 960,
        "plan does not match the declared 6x160, checkpoints 3/6 horizon",
    )
    cells = plan.get("cells")
    _require(isinstance(cells, list) and len(cells) == 2, "plan must have two paired cells")
    left, right = cells
    _require(isinstance(left, Mapping) and isinstance(right, Mapping), "paired cells are malformed")
    left_params, right_params = left.get("parameters"), right.get("parameters")
    left_seeds, right_seeds = left.get("seeds"), right.get("seeds")
    _require(isinstance(left_params, Mapping) and isinstance(right_params, Mapping), "cell controls are missing")
    _require(isinstance(left_seeds, Mapping) and isinstance(right_seeds, Mapping), "paired seeds are missing")
    lp, rp = dict(left_params), dict(right_params)
    left_try, right_try = lp.pop(_PAIR_PARAM, None), rp.pop(_PAIR_PARAM, None)
    _require(lp == rp and left_seeds == right_seeds, "paired cells differ beyond the declared reproduction control")
    _require(left_try == 0.0 and right_try == 0.15, "paired reproduction probabilities must be 0.0 and 0.15")
    left_dimensions, right_dimensions = left.get("dimensions"), right.get("dimensions")
    _require(isinstance(left_dimensions, Mapping) and isinstance(right_dimensions, Mapping), "cell dimensions are missing")
    ld, rd = dict(left_dimensions), dict(right_dimensions)
    _require(ld.pop("try_probability", None) == left_try, "control dimension disagrees with its parameter")
    _require(rd.pop("try_probability", None) == right_try and ld == rd, "paired dimensions differ beyond the reproduction control")
    _require(isinstance(left.get("cell_id"), str) and isinstance(right.get("cell_id"), str), "paired cell IDs are missing")
    _require(left["cell_id"] != right["cell_id"], "paired cell IDs must be distinct")
    return dict(left), dict(right)


def _verify_inventory_file(root: Path, manifest: Mapping[str, Any], relative: str) -> dict[str, Any]:
    rows = manifest.get("artifacts")
    _require(isinstance(rows, list), "disposition inventory is malformed")
    matches = [row for row in rows if isinstance(row, Mapping) and row.get("path") == relative]
    _require(len(matches) == 1, f"disposition inventory must bind {relative} exactly once")
    row = dict(matches[0])
    path = root / relative
    _require(not path.is_symlink() and path.is_file(), f"selected artifact is not a regular file: {relative}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise PairedUsageError(f"selected artifact escapes run root: {relative}") from exc
    observed_sha, observed_bytes = _file_sha256(resolved)
    _require(row.get("sha256") == observed_sha, f"artifact SHA-256 differs from disposition inventory: {relative}")
    _require(row.get("bytes") == observed_bytes, f"artifact byte count differs from disposition inventory: {relative}")
    return {"path": relative, "sha256": observed_sha, "bytes": observed_bytes}


def _check_terminal_bindings(
    root: Path,
    request_sha: str,
    result_sha: str,
    disposition_sha: str,
    plan_sha: str,
    campaign_id: str,
    source_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    marker = _read_json(root, "terminal_lifecycle_commit.json")
    _sealed_hash(marker, "commit_sha256", "terminal lifecycle commit")
    _require(marker.get("schema_version") == "eiel_execution_terminal_lifecycle_commit_v1", "terminal lifecycle commit schema mismatch")
    _require(marker.get("request_sha256") == request_sha and marker.get("plan_sha256") == plan_sha, "terminal lifecycle commit request/plan mismatch")
    _require(marker.get("source_identity_sha256") == canonical_sha256(source_identity), "terminal lifecycle commit source identity mismatch")
    result_ref, manifest_ref, storage_ref = marker.get("result"), marker.get("manifest"), marker.get("storage_receipt")
    _require(isinstance(result_ref, Mapping) and isinstance(manifest_ref, Mapping) and isinstance(storage_ref, Mapping), "terminal lifecycle references are incomplete")
    result_file_sha, _ = _file_sha256(root / "execution_result.json")
    manifest_file_sha, _ = _file_sha256(root / "artifact_disposition_manifest.json")
    _require(result_ref.get("path") == "execution_result.json" and result_ref.get("file_sha256") == result_file_sha and result_ref.get("result_sha256") == result_sha, "terminal commit does not bind the exact execution result")
    _require(manifest_ref.get("path") == "artifact_disposition_manifest.json" and manifest_ref.get("file_sha256") == manifest_file_sha and manifest_ref.get("manifest_sha256") == disposition_sha, "terminal commit does not bind the exact disposition manifest")
    storage = _read_json(root, "storage_disposition_receipt.json")
    _sealed_hash(storage, "receipt_sha256", "current storage disposition receipt")
    # This receipt is an actuator state that the exact-release worker advances
    # after terminal commit. Keep the immutable commit intact and validate the
    # current self-seal plus its anchored disposition/plan below.
    _require(storage_ref.get("path") == "storage_disposition_receipt.json", "terminal commit storage receipt path mismatch")
    _require(storage.get("schema_version") == "eiel_storage_disposition_receipt_v1", "current storage receipt schema mismatch")
    _require(storage.get("manifest_sha256") == disposition_sha and storage.get("plan_sha256") == plan_sha, "current storage receipt disposition/plan binding mismatch")
    _require(storage.get("campaign_id") == campaign_id, "current storage receipt campaign mismatch")
    return marker, storage


def _check_controls_and_rows(
    root: Path,
    plan: Mapping[str, Any],
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    route_sha256: str,
    provider_id: str,
    profile_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    batches = plan.get("physical_batches")
    _require(isinstance(batches, list) and len(batches) == 2, "plan must contain exactly two physical batches")
    cell_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    inventory_files: list[dict[str, Any]] = []
    for ordinal, cell in enumerate((left, right)):
        relative = f"batches/batch-{ordinal:04d}/manifest.json"
        batch = _read_json(root, relative)
        inventory_files.append(_verify_inventory_file(root, _read_json(root, "artifact_disposition_manifest.json"), relative))
        cell_id = str(cell["cell_id"])
        _require(batch.get("schema_version") == "eiel_campaign_batch_manifest_v1" and batch.get("status") == "completed", f"batch {ordinal} did not complete")
        _require(batch.get("campaign_id") == plan.get("campaign_id") and batch.get("plan_sha256") == plan.get("plan_sha256"), f"batch {ordinal} campaign/plan mismatch")
        _require(batch.get("execution_route_sha256") == route_sha256 and batch.get("provider_id") == provider_id and batch.get("profile_id") == profile_id, f"batch {ordinal} provider/route/profile binding mismatch")
        _require(batch.get("cell_ids") == [cell_id], f"batch {ordinal} cell binding mismatch")
        rows = batch.get("metrics_rows")
        _require(isinstance(rows, list) and len(rows) == 1 and rows[0].get("cell_id") == cell_id, f"batch {ordinal} must contain exactly its producer metrics row")
        row = rows[0]
        metrics = row.get("metrics")
        _require(isinstance(metrics, Mapping), f"batch {ordinal} producer metrics are missing")
        # Preserve absent and null measurements as such; no default zero is introduced.
        cell_rows.append({"cell_id": cell_id, **dict(metrics)})
        generations = row.get("generation_metrics")
        _require(isinstance(generations, list) and len(generations) == 6, f"batch {ordinal} must retain six generation observations")
        generation_rows.extend(dict(item) for item in generations if isinstance(item, Mapping))
        result = batch.get("result")
        _require(isinstance(result, Mapping), f"batch {ordinal} provider result is missing")
        receipts = result.get("runner_receipts")
        _require(isinstance(receipts, list) and receipts, f"batch {ordinal} lacks primitive control receipts")
        expected_try = cell.get("parameters", {}).get(_PAIR_PARAM)
        for receipt in receipts:
            control = receipt.get("primitive_controls") if isinstance(receipt, Mapping) else None
            wanted = {_PAIR_PARAM: expected_try}
            _require(
                isinstance(control, Mapping)
                and control.get("schema_version") == "eiel_production_primitive_override_receipt_v1"
                and control.get("requested") == wanted
                and control.get("applied") == wanted,
                f"batch {ordinal} primitive control readback mismatch",
            )
    return cell_rows, generation_rows, inventory_files


def _check_continuations(
    root: Path,
    plan: Mapping[str, Any],
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    route_sha256: str,
    provider_id: str,
    profile_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rel = "comparisons_manifest.json"
    comparisons = _read_json(root, rel)
    manifest = _read_json(root, "artifact_disposition_manifest.json")
    files = [_verify_inventory_file(root, manifest, rel)]
    branches = comparisons.get("branches")
    _require(comparisons.get("campaign_id") == plan.get("campaign_id") and comparisons.get("plan_sha256") == plan.get("plan_sha256"), "comparison manifest campaign/plan mismatch")
    _require(comparisons.get("execution_route_sha256") == route_sha256 and comparisons.get("provider_id") == provider_id and comparisons.get("profile_id") == profile_id, "comparison manifest provider/route/profile binding mismatch")
    _require(isinstance(branches, list) and len(branches) == 3, "comparison manifest must declare two continuations and one reporting parity branch")
    expected = {
        str(left["cell_id"]): "checkpoint_continuation",
        str(right["cell_id"]): "checkpoint_continuation",
    }
    seen_cells: set[str] = set()
    continuation_evidence: list[dict[str, Any]] = []
    reporting_branch: Mapping[str, Any] | None = None
    for branch in branches:
        _require(isinstance(branch, Mapping), "comparison branch is malformed")
        branch_id = branch.get("branch_id")
        relative = f"comparisons/{branch_id}/manifest.json"
        branch_manifest = _read_json(root, relative)
        files.append(_verify_inventory_file(root, manifest, relative))
        branch_id = _check_branch_identity(branch, branch_manifest, campaign_id=str(plan.get("campaign_id")), plan_sha256=str(plan.get("plan_sha256")))
        _require(branch_manifest.get("execution_route_sha256") == route_sha256 and branch_manifest.get("provider_id") == provider_id and branch_manifest.get("profile_id") == profile_id, "comparison branch provider/route/profile binding mismatch")
        if branch.get("kind") == "checkpoint_continuation":
            cell_id = str(branch.get("source_cell_id"))
            _require(cell_id in expected and expected[cell_id] == branch_manifest.get("kind"), "continuation branch maps to an unexpected cell")
            _require(cell_id not in seen_cells, "duplicate continuation branch for a paired cell")
            seen_cells.add(cell_id)
            _require(branch_manifest.get("source_cell_id") == cell_id and branch_manifest.get("source_batch_id") == f"batch-{expected_cells_ordinal(cell_id, left, right):04d}", "continuation source cell/batch binding mismatch")
            _require(branch_manifest.get("split_generation") == 3, "continuation did not restore from the declared midpoint checkpoint")
            comparison = _check_comparison_match(branch_manifest.get("comparison"), surfaces={"metrics_rows_sha256", "terminal_checkpoint_digest"}, label="continuation")
            continuation_evidence.append({"branch_id": branch_id, "cell_id": cell_id, "split_generation": 3, "compared_surfaces": list(comparison["compared_surfaces"]), "match": True})
        elif branch.get("kind") == "reporting_parity":
            _require(reporting_branch is None, "duplicate reporting parity branch")
            reporting_branch = branch_manifest
            comparison = _check_comparison_match(branch_manifest.get("comparison"), surfaces={"terminal_checkpoint_digest"}, label="reporting parity")
        else:
            raise PairedUsageError("unexpected comparison branch kind")
    _require(seen_cells == set(expected), "both paired cells need a verified continuation")
    _require(isinstance(reporting_branch, Mapping), "reporting parity branch is missing")
    parity = _read_json(root, "reporting_parity_receipt.json")
    files.append(_verify_inventory_file(root, manifest, "reporting_parity_receipt.json"))
    _require(parity.get("schema_version") == "eiel_campaign_reporting_parity_receipt_v1" and parity.get("status") == "supported", "reporting parity receipt is not supported")
    _require(parity.get("campaign_id") == plan.get("campaign_id") and parity.get("plan_sha256") == plan.get("plan_sha256"), "reporting parity receipt campaign/plan mismatch")
    _require(parity.get("execution_route_sha256") == route_sha256 and parity.get("provider_id") == provider_id and parity.get("profile_id") == profile_id, "reporting parity receipt provider/route/profile binding mismatch")
    _require(parity.get("comparison_branch_id") == reporting_branch.get("branch_id"), "reporting parity receipt branch mismatch")
    comparison = reporting_branch.get("comparison")
    _require(parity.get("comparison_receipt_sha256") == canonical_sha256(comparison), "reporting parity receipt does not bind its branch comparison")
    return files, {"status": "supported", "branch_id": reporting_branch.get("branch_id"), "continuations": continuation_evidence}


def expected_cells_ordinal(cell_id: str, left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    if cell_id == left.get("cell_id"):
        return 0
    if cell_id == right.get("cell_id"):
        return 1
    raise PairedUsageError("unknown paired cell ID")


def compare_run(
    run_root: str | Path,
    *,
    expected_request_sha256: str,
    expected_result_sha256: str,
    expected_disposition_sha256: str,
) -> dict[str, Any]:
    root = Path(run_root).expanduser().resolve(strict=True)
    request = _read_json(root, "execution_request.json")
    _require(request.get("schema_version") == "eiel_execution_request_receipt_v2", "execution request receipt schema mismatch")
    nested_request = request.get("request")
    _require(isinstance(nested_request, Mapping), "canonical ExecutionRequestV2 is missing")
    parsed_request = ExecutionRequestV2.from_mapping(nested_request, base_dir=root)
    request_sha = parsed_request.request_sha256
    _check_trusted_anchor(request_sha, expected_request_sha256, "request")
    _require(request_sha == request.get("request_sha256"), "request receipt digest mismatch")
    _require(request.get("run_root") == str(root), "request receipt run-root binding mismatch")

    plan = _read_json(root, "execution_plan.json")
    left, right = _check_pair_plan(plan)
    plan_sha = str(plan.get("plan_sha256"))
    _require(request.get("bound_plan_sha256") == plan_sha, "request receipt does not bind the exact plan")

    result = _read_json(root, "execution_result.json")
    result_sha = _sealed_hash(result, "result_sha256", "execution result")
    _check_trusted_anchor(result_sha, expected_result_sha256, "result")
    _require(result.get("schema_version") == "eiel_execution_result_v2" and result.get("ok") is True and result.get("status") == "completed" and result.get("execution_status") == "completed", "execution result is not a completed success")
    _require(result.get("request_sha256") == request_sha and result.get("plan_sha256") == plan_sha and result.get("run_root") == str(root), "execution result request/plan/root binding mismatch")
    _require(result.get("campaign_id") == plan.get("campaign_id"), "execution result campaign mismatch")
    receipt = result.get("execution_receipt")
    source_identity = result.get("source_identity")
    _require(isinstance(receipt, Mapping) and isinstance(source_identity, Mapping), "execution result lacks canonical receipt/source identity")
    _require(request.get("source_identity") == source_identity, "request/result source identity mismatch")
    _require(receipt.get("campaign_id") == plan.get("campaign_id") and receipt.get("plan_sha256") == plan_sha and receipt.get("status") == "completed", "execution receipt campaign/plan/status mismatch")
    route_sha = receipt.get("execution_route_sha256")
    provider_id = receipt.get("provider_id")
    profile_id = receipt.get("profile_id")
    _require(result.get("profile_id") == profile_id, "result and execution receipt profile mismatch")
    _require(isinstance(route_sha, str) and len(route_sha) == 64 and all(ch in "0123456789abcdef" for ch in route_sha), "execution route digest is malformed")
    _require(provider_id == "eiel-production-provider-v1" and profile_id == "eiel-production-runtime-v1-cpu-native", "execution did not use the declared CPU-native provider/profile")

    disposition_raw = _read_json(root, "artifact_disposition_manifest.json")
    try:
        disposition = validate_artifact_disposition_manifest_v1(disposition_raw)
    except ArtifactLifecycleError as exc:
        raise PairedUsageError(f"artifact disposition validation failed: {exc}") from exc
    disposition_sha = str(disposition.get("manifest_sha256"))
    _check_trusted_anchor(disposition_sha, expected_disposition_sha256, "disposition")
    _require(disposition.get("plan_sha256") == plan_sha and disposition.get("campaign_id") == plan.get("campaign_id") and disposition.get("terminal_status") == "completed", "disposition campaign/plan/status mismatch")
    terminal_commit, current_storage = _check_terminal_bindings(root, request_sha, result_sha, disposition_sha, plan_sha, str(plan.get("campaign_id")), source_identity)

    preflight = _read_json(root, "preflight_receipt.json")
    try:
        mechanical = verify_mechanical_execution_v1(
            run_root=root,
            plan=plan,
            preflight=preflight,
            execution_receipt=receipt,
        )
    except (ExecutionFrontDoorError, OSError, ValueError) as exc:
        raise PairedUsageError(f"canonical mechanical verification failed: {exc}") from exc
    _require(mechanical.get("status") == "PASS", "canonical mechanical verification did not pass")
    _require((result.get("mechanical_verification") or {}).get("status") == "PASS", "stored execution result mechanical verification did not pass")

    rows, generation_rows, inventory_files = _check_controls_and_rows(
        root, plan, left, right, route_sha256=str(route_sha),
        provider_id=str(provider_id), profile_id=str(profile_id),
    )
    comparison_files, continuation = _check_continuations(
        root, plan, left, right, route_sha256=str(route_sha),
        provider_id=str(provider_id), profile_id=str(profile_id),
    )
    all_files = inventory_files + comparison_files
    analysis_rows = [
        {"cell_id": row["cell_id"], "metrics": {key: value for key, value in row.items() if key != "cell_id"}}
        for row in rows
    ]
    analysis = analyze_campaign(plan, analysis_rows, generation_metric_rows=generation_rows)
    measured = _paired_metric_summary(rows, str(left["cell_id"]), str(right["cell_id"]))
    return {
        "schema_version": "eiel_paired_cpu_usage_comparison_v1",
        "authenticated": True,
        "anchors": {
            "request_sha256": request_sha,
            "result_sha256": result_sha,
            "disposition_manifest_sha256": disposition_sha,
            "plan_sha256": plan_sha,
            "source_commit": source_identity.get("git", {}).get("commit"),
        },
        "campaign_id": plan.get("campaign_id"),
        "profile_id": receipt.get("profile_id"),
        "provider_id": receipt.get("provider_id"),
        "execution_route_sha256": receipt.get("execution_route_sha256"),
        "storage_lifecycle_receipt": {
            "terminal_commit_receipt_sha256": (terminal_commit.get("storage_receipt") or {}).get("receipt_sha256"),
            "current_receipt_sha256": current_storage.get("receipt_sha256"),
            "advanced_after_terminal_commit": current_storage.get("receipt_sha256") != (terminal_commit.get("storage_receipt") or {}).get("receipt_sha256"),
            "authorization_reason": current_storage.get("authorization_reason"),
            "bytes_released": current_storage.get("bytes_released"),
            "bytes_retained": current_storage.get("bytes_retained"),
            "bytes_unresolved_RX": current_storage.get("bytes_unresolved_RX"),
        },
        "settings": {
            "horizon": "legacy 6 x 160 ticks (960 ticks total)",
            "checkpoints": [3, 6],
            "seeds": dict(left["seeds"]),
            "control": {"cell_id": left["cell_id"], _PAIR_PARAM: 0.0},
            "treatment": {"cell_id": right["cell_id"], _PAIR_PARAM: 0.15},
        },
        "producer_batch_metrics": rows,
        "paired_metrics": measured,
        "canonical_campaign_analysis": analysis,
        "continuation_and_reporting": continuation,
        "mechanical_verification": {"status": mechanical.get("status"), "checks_passed": sum(row.get("status") == "PASS" for row in mechanical.get("checks", [])), "checks_total": len(mechanical.get("checks", []))},
        "inventory_verified_files": all_files,
        "qualification_status": "not_evaluated",
        "limitations": [
            "This is a bounded legacy Patch CPU usage demonstration, not Full Evolution or formal Qualification.",
            "Only the declared reproduction-probability pair and one shared seed pair were exercised.",
            "Metrics are taken from authenticated producer batch rows; absent/null values remain absent/null.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--expected-request-sha256", required=True)
    parser.add_argument("--expected-result-sha256", required=True)
    parser.add_argument("--expected-disposition-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        value = compare_run(
            args.run_root,
            expected_request_sha256=args.expected_request_sha256,
            expected_result_sha256=args.expected_result_sha256,
            expected_disposition_sha256=args.expected_disposition_sha256,
        )
    except (PairedUsageError, ExecutionFrontDoorError, OSError, ValueError) as exc:
        print(json.dumps({"schema_version": "eiel_paired_cpu_usage_comparison_v1", "authenticated": False, "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(value, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
