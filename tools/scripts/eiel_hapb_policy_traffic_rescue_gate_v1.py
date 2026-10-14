#!/usr/bin/env python3
"""Mechanical review of the bounded HAPB policy-traffic rescue receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from engine.campaigns.orchestration import atomic_write_json


SCHEMA_VERSION = "eiel_hapb_policy_traffic_rescue_gate_v1"
ROW_BYTES = 39_964


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _check_traffic(label: str, traffic: Any) -> None:
    _require(isinstance(traffic, dict), f"{label} traffic is malformed")
    _require(traffic.get("policy_row_bytes") == ROW_BYTES, f"{label} row bytes differ")
    rows = int(traffic.get("policy_h2d_rows", -1))
    bytes_ = int(traffic.get("policy_h2d_bytes", -1))
    _require(rows >= 0 and bytes_ == rows * ROW_BYTES, f"{label} physical row/byte accounting differs")
    _require(int(traffic.get("policy_d2h_rows", -1)) == 0, f"{label} unexpected policy D2H rows")
    _require(int(traffic.get("policy_d2h_bytes", -1)) == 0, f"{label} unexpected policy D2H bytes")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def _check_h4_receipt(
    *,
    label: str,
    receipt_path: Path,
    expected_alias: bool,
    artifact_sha256: str,
    manifest_sha256: str,
    source_root: str,
) -> dict[str, Any]:
    receipt = _load_json(receipt_path)
    _require(
        receipt.get("schema_version", "").startswith(
            "eiel_b48_wdl_jungle_eed_hapb_h4_continuation_v1_repeat_tick_diagnostic_v1"
        ),
        f"{label} H4 receipt schema differs",
    )
    _require(receipt.get("content_alias_enabled") is expected_alias, f"{label} alias mode differs")
    _require(receipt.get("artifact_sha256") == artifact_sha256, f"{label} artifact binding differs")
    _require(receipt.get("manifest_sha256") == manifest_sha256, f"{label} manifest binding differs")
    _require(
        receipt.get("reused_generation_zero_source_root") == source_root,
        f"{label} source-root binding differs",
    )
    _require(receipt.get("production_execution_admission") is False, f"{label} claims production admission")
    _require(receipt.get("qualification_claim") is False, f"{label} claims qualification")
    _require(receipt.get("terminal_projection_digests_exact") is True, f"{label} H4 repeat is not exact")
    _require(int(receipt.get("mismatch_count", -1)) == 0, f"{label} H4 repeat has mismatches")
    digests = receipt.get("terminal_projection_digests")
    _require(isinstance(digests, dict), f"{label} terminal digest lists are missing")
    first_digests = digests.get("first")
    second_digests = digests.get("second")
    _require(isinstance(first_digests, list) and first_digests, f"{label} first digest list is missing")
    _require(first_digests == second_digests, f"{label} repeated terminal digests differ")

    observations = receipt.get("policy_traffic_observations")
    _require(isinstance(observations, dict), f"{label} policy traffic observations are missing")
    for trace_label in ("first", "second"):
        observation = observations.get(trace_label)
        _require(isinstance(observation, dict), f"{label} {trace_label} traffic observation is missing")
        _require(
            observation.get("content_alias_enabled") is expected_alias,
            f"{label} {trace_label} alias mode differs",
        )
        if expected_alias:
            _require(
                observation.get("content_alias_relation_sha256") not in (None, "disabled"),
                f"{label} {trace_label} alias relation is missing",
            )
        else:
            _require(
                observation.get("content_alias_relation_sha256") == "disabled",
                f"{label} {trace_label} baseline relation is not disabled",
            )
        _check_traffic(f"{label} {trace_label}", observation.get("counters"))

    # The process-global state before preparation may differ between separately
    # launched diagnostic processes.  The post-prepare and post-execute RNG
    # identities must nevertheless be identical across the two route modes.
    rng = receipt.get("process_rng_traces")
    _require(isinstance(rng, dict), f"{label} RNG traces are missing")
    for trace_label in ("first", "second"):
        trace = rng.get(trace_label)
        _require(isinstance(trace, dict), f"{label} {trace_label} RNG trace is missing")
        _require(isinstance(trace.get("after_prepare"), dict), f"{label} {trace_label} after-prepare RNG trace is missing")
        _require(isinstance(trace.get("after_execute"), dict), f"{label} {trace_label} after-execute RNG trace is missing")

    trace_digests: dict[str, dict[str, Any]] = {}
    for trace_label in ("first", "second"):
        trace_path = receipt_path.parent / f"repeat_tick_{trace_label}_birth_transaction_trace.json"
        _require(trace_path.is_file(), f"{label} {trace_label} birth trace is missing")
        trace_payload = _load_json(trace_path)
        _require(isinstance(trace_payload.get("events"), list), f"{label} {trace_label} birth trace is malformed")
        _require(
            all(
                isinstance(event, dict)
                and all(
                    isinstance(item, dict)
                    and item.get("policy_birth_tape_cursor_end")
                    == item.get("policy_birth_tape_cursor_start", -1) + item.get("birth_count", -1)
                    for item in event.get("active_cells", [])
                )
                for event in trace_payload["events"]
            ),
            f"{label} {trace_label} birth cursor accounting is invalid",
        )
        trace_digests[trace_label] = trace_payload
    return {
        "path": str(receipt_path),
        "sha256": _sha256(receipt_path),
        "receipt": receipt,
        "terminal_projection_digests": digests,
        "process_rng_traces": rng,
        "birth_traces": trace_digests,
    }


def _check_continuation_receipt(
    *,
    label: str,
    receipt_path: Path,
    expected_alias: bool,
    artifact_sha256: str,
    manifest_sha256: str,
    source_root: str,
) -> dict[str, Any]:
    receipt = _load_json(receipt_path)
    _require(
        receipt.get("schema_version") == "eiel_b48_wdl_jungle_eed_hapb_h4_continuation_v1",
        f"{label} continuation receipt schema differs",
    )
    _require(receipt.get("content_alias_enabled") is expected_alias, f"{label} continuation alias mode differs")
    _require(receipt.get("artifact_sha256") == artifact_sha256, f"{label} continuation artifact binding differs")
    _require(receipt.get("manifest_sha256") == manifest_sha256, f"{label} continuation manifest binding differs")
    _require(
        receipt.get("reused_generation_zero_source_root") == source_root,
        f"{label} continuation source-root binding differs",
    )
    _require(receipt.get("production_execution_admission") is False, f"{label} continuation claims production admission")
    _require(receipt.get("qualification_claim") is False, f"{label} continuation claims qualification")
    _require(receipt.get("fresh_owner_restore") is True, f"{label} fresh-owner restore is missing")
    _require(receipt.get("continuation_exact") is True, f"{label} continuation is not exact")
    _require(int(receipt.get("checkpoint_restore_boundary_mismatch_count", -1)) == 0, f"{label} checkpoint restore differs")
    # The harness records an explicit null when no separately comparable
    # midpoint reference exists; that is the non-applicable, zero-mismatch
    # case.  A missing field still defaults to -1 and fails closed.
    midpoint_mismatch = receipt.get("reference_midpoint_boundary_mismatch_count", -1)
    _require(
        midpoint_mismatch is None or int(midpoint_mismatch) == 0,
        f"{label} midpoint boundary differs",
    )
    reference = receipt.get("reference_terminal_projection_digests")
    resumed = receipt.get("resumed_terminal_projection_digests")
    _require(isinstance(reference, list) and reference, f"{label} reference terminal digests are missing")
    _require(reference == resumed, f"{label} resumed terminal digests differ")
    checkpoint_path = Path(str(receipt.get("midpoint_checkpoint_path", "")))
    _require(checkpoint_path.is_file(), f"{label} midpoint checkpoint is missing")
    _require(_sha256(checkpoint_path) == receipt.get("midpoint_checkpoint_sha256"), f"{label} midpoint checkpoint hash differs")
    rng = receipt.get("process_rng_traces")
    _require(isinstance(rng, dict), f"{label} continuation RNG traces are missing")
    for trace_label in ("reference", "resumed"):
        trace = rng.get(trace_label)
        _require(isinstance(trace, dict), f"{label} {trace_label} continuation RNG trace is missing")
        _require(isinstance(trace.get("after_prepare"), dict), f"{label} {trace_label} after-prepare RNG trace is missing")
        _require(isinstance(trace.get("after_execute"), dict), f"{label} {trace_label} after-execute RNG trace is missing")
    return {
        "path": str(receipt_path),
        "sha256": _sha256(receipt_path),
        "receipt": receipt,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-receipt", required=True, type=Path)
    parser.add_argument("--alias-h4-receipt", required=True, type=Path)
    parser.add_argument("--baseline-h4-receipt", required=True, type=Path)
    parser.add_argument("--alias-continuation-receipt", required=True, type=Path)
    parser.add_argument("--baseline-continuation-receipt", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.output_root.exists():
        raise RuntimeError("refusing to overwrite an immutable rescue gate receipt")
    fixture_path = args.fixture_receipt.resolve()
    fixture = _load_json(fixture_path)
    _require(fixture.get("schema_version") == "eiel_b48_hapb_policy_traffic_fixture_v1", "fixture schema differs")
    artifact = Path(fixture["artifact"]["path"])
    manifest = Path(fixture["artifact"]["manifest_path"])
    _require(_sha256(artifact) == fixture["artifact"]["sha256"], "artifact hash changed")
    _require(_sha256(manifest) == fixture["artifact"]["manifest_sha256"], "manifest hash changed")
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    _require(manifest_payload.get("artifact_sha256") == fixture["artifact"]["sha256"], "manifest artifact binding differs")
    baseline = fixture.get("baseline")
    alias = fixture.get("alias")
    _require(isinstance(baseline, dict) and isinstance(alias, dict), "fixture cases are missing")
    _require(baseline.get("content_alias_enabled") is False, "baseline is not explicitly disabled")
    _require(alias.get("content_alias_enabled") is True, "alias is not explicitly enabled")
    _require(baseline.get("relation_sha256") == "disabled", "baseline relation is not disabled")
    _require(alias.get("relation_sha256") not in (None, "disabled"), "alias relation is missing")
    baseline_traffic = baseline.get("traffic")
    alias_traffic = alias.get("traffic")
    _check_traffic("baseline", baseline_traffic)
    _check_traffic("alias", alias_traffic)
    baseline_hashes = baseline.get("output_sha256")
    alias_hashes = alias.get("output_sha256")
    _require(isinstance(baseline_hashes, list) and baseline_hashes == alias_hashes, "exact output digests differ")
    _require(fixture.get("exact_output_parity") is True, "fixture did not prove exact output parity")
    _require(all(fixture.get("exact_output_parity_by_iteration", [])), "an iteration failed exact parity")
    baseline_bytes = int(baseline_traffic["policy_h2d_bytes"])
    alias_bytes = int(alias_traffic["policy_h2d_bytes"])
    ratio = baseline_bytes / alias_bytes if alias_bytes else None
    _require(ratio is not None and ratio >= 10.0, "measured policy-weight traffic reduction is below 10x")
    semantics = fixture.get("semantics")
    _require(isinstance(semantics, dict), "fixture semantics are missing")
    _require(semantics.get("production_execution_admission") is False, "fixture claims production admission")
    _require(semantics.get("qualification_claim") is False, "fixture claims qualification")
    source_root = str(fixture["source"]["source_root"])
    alias_h4 = _check_h4_receipt(
        label="alias",
        receipt_path=args.alias_h4_receipt.resolve(),
        expected_alias=True,
        artifact_sha256=fixture["artifact"]["sha256"],
        manifest_sha256=fixture["artifact"]["manifest_sha256"],
        source_root=source_root,
    )
    baseline_h4 = _check_h4_receipt(
        label="baseline",
        receipt_path=args.baseline_h4_receipt.resolve(),
        expected_alias=False,
        artifact_sha256=fixture["artifact"]["sha256"],
        manifest_sha256=fixture["artifact"]["manifest_sha256"],
        source_root=source_root,
    )
    alias_continuation = _check_continuation_receipt(
        label="alias",
        receipt_path=args.alias_continuation_receipt.resolve(),
        expected_alias=True,
        artifact_sha256=fixture["artifact"]["sha256"],
        manifest_sha256=fixture["artifact"]["manifest_sha256"],
        source_root=source_root,
    )
    baseline_continuation = _check_continuation_receipt(
        label="baseline",
        receipt_path=args.baseline_continuation_receipt.resolve(),
        expected_alias=False,
        artifact_sha256=fixture["artifact"]["sha256"],
        manifest_sha256=fixture["artifact"]["manifest_sha256"],
        source_root=source_root,
    )
    _require(
        alias_h4["terminal_projection_digests"] == baseline_h4["terminal_projection_digests"],
        "current-source alias/baseline terminal trajectories differ",
    )
    for trace_label in ("first", "second"):
        _require(
            alias_h4["birth_traces"][trace_label] == baseline_h4["birth_traces"][trace_label],
            f"current-source alias/baseline {trace_label} birth traces differ",
        )
        for rng_stage in ("after_prepare", "after_execute"):
            _require(
                alias_h4["process_rng_traces"][trace_label][rng_stage]
                == baseline_h4["process_rng_traces"][trace_label][rng_stage],
                f"current-source alias/baseline {trace_label} {rng_stage} RNG traces differ",
            )
    for digest_field in ("reference_terminal_projection_digests", "resumed_terminal_projection_digests"):
        _require(
            alias_continuation["receipt"][digest_field]
            == baseline_continuation["receipt"][digest_field],
            f"current-source alias/baseline continuation {digest_field} differ",
        )
    for trace_label in ("reference", "resumed"):
        for rng_stage in ("after_prepare", "after_execute"):
            _require(
                alias_continuation["receipt"]["process_rng_traces"][trace_label][rng_stage]
                == baseline_continuation["receipt"]["process_rng_traces"][trace_label][rng_stage],
                f"current-source alias/baseline continuation {trace_label} {rng_stage} RNG traces differ",
            )
    result = {
        "schema_version": SCHEMA_VERSION,
        "decision": "PASS",
        "fixture_receipt": str(fixture_path),
        "fixture_receipt_sha256": _sha256(fixture_path),
        "artifact": fixture["artifact"],
        "source": fixture["source"],
        "exact_output_parity": True,
        "policy_weight_h2d_bytes_baseline": baseline_bytes,
        "policy_weight_h2d_bytes_alias": alias_bytes,
        "policy_weight_h2d_bytes_reduction_ratio": ratio,
        "policy_weight_h2d_bytes_reduction_ge_10x": True,
        "current_source_h4_alias_baseline_exact": True,
        "current_source_h4_alias_receipt": alias_h4["path"],
        "current_source_h4_baseline_receipt": baseline_h4["path"],
        "current_source_h4_alias_continuation_receipt": alias_continuation["path"],
        "current_source_h4_baseline_continuation_receipt": baseline_continuation["path"],
        "current_source_h4_terminal_projection_digests_exact": True,
        "current_source_h4_birth_cursor_and_recombination_exact": True,
        "current_source_h4_rng_after_prepare_and_execute_exact": True,
        "current_source_h4_checkpoint_restore_exact": True,
        "current_source_h4_fresh_owner_continuation_exact": True,
        "current_source_h4_policy_traffic": {
            "alias_first": alias_h4["receipt"]["policy_traffic_observations"]["first"]["counters"],
            "baseline_first": baseline_h4["receipt"]["policy_traffic_observations"]["first"]["counters"],
        },
        "qualification_claim": False,
        "production_execution_admission": False,
        "scope": "bounded B48/C512 policy-boundary rescue; Gate-5 lifecycle evidence remains separate",
    }
    args.output_root.mkdir(parents=True, exist_ok=False)
    atomic_write_json(args.output_root / "policy_traffic_rescue_gate_receipt.json", result)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
