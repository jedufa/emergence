#!/usr/bin/env python3
from __future__ import annotations

"""Define and validate env/policy PCG64 operation receipts."""

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final


SCHEMA_VERSION: Final[str] = "eiel_option_d_m28_env_policy_receipt_contract_v1"
DECISION_WITHHELD: Final[str] = (
    "M28-A-ENV-POLICY-PCG64-RECEIPT-CONTRACT-DEFINED-PROMOTION-WITHHELD"
)
ENV_POLICY_STREAM_ID: Final[str] = "numpy.environment_policy"
REQUIRED_OPERATORS: Final[tuple[str, ...]] = (
    "environment.food_spawn.uniform_or_bounded",
    "policy.action_gumbel.float32",
)
COMMON_REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "operator_id",
    "physical_stream_id",
    "operation_receipt_schema_version",
    "status_ok",
    "promotion_qualified",
    "promotion_blockers",
    "raw_pcg64_delta_available",
    "semantic_draws",
    "raw64_draws",
    "raw32_draws",
    "state_before_digest",
    "state_after_digest",
    "buffer_before_digest",
    "buffer_after_digest",
    "has_uint32_before",
    "has_uint32_after",
    "uinteger_before",
    "uinteger_after",
    "checkpoint_continuation_covered",
    "failure_atomicity_covered",
    "authority_class",
)
OPERATOR_EXTRA_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    "environment.food_spawn.uniform_or_bounded": (
        "food_spawn_attempts",
        "food_spawn_successes",
        "bounded_rejection_attempts",
        "bounded_rejection_checkpoint_continuation_covered",
    ),
    "policy.action_gumbel.float32": (
        "policy_rows_requested",
        "policy_cache_hit_rows",
        "policy_cache_miss_rows",
        "policy_rows_generated",
        "gumbel_float32_draws",
        "post_action_state_buffer_bound",
    ),
}
INTEGER_FIELDS: Final[tuple[str, ...]] = (
    "semantic_draws",
    "raw64_draws",
    "raw32_draws",
    "uinteger_before",
    "uinteger_after",
    "food_spawn_attempts",
    "food_spawn_successes",
    "bounded_rejection_attempts",
    "policy_rows_requested",
    "policy_cache_hit_rows",
    "policy_cache_miss_rows",
    "policy_rows_generated",
    "gumbel_float32_draws",
)
BOOL_FIELDS: Final[tuple[str, ...]] = (
    "status_ok",
    "promotion_qualified",
    "raw_pcg64_delta_available",
    "has_uint32_before",
    "has_uint32_after",
    "checkpoint_continuation_covered",
    "failure_atomicity_covered",
    "bounded_rejection_checkpoint_continuation_covered",
    "post_action_state_buffer_bound",
)


class M28Rejected(RuntimeError):
    """The supplied receipt is malformed for the M28 contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise M28Rejected(f"{name} is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise M28Rejected(f"{name} must contain one JSON object")
    return payload


def _hex_digest(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdef" for char in value)


def _discover_m27_operations(m27_receipt: Mapping[str, Any]) -> list[str]:
    operations: list[str] = []
    for row in list(m27_receipt.get("discovered_operations") or []):
        if not isinstance(row, Mapping):
            continue
        operator_id = str(row.get("operator_id") or "")
        if operator_id:
            operations.append(operator_id)
    return sorted(set(operations))


def _contract_definition() -> dict[str, Any]:
    return {
        "physical_stream_id": ENV_POLICY_STREAM_ID,
        "required_operators": list(REQUIRED_OPERATORS),
        "common_required_fields": list(COMMON_REQUIRED_FIELDS),
        "operator_extra_fields": {
            operator_id: list(fields)
            for operator_id, fields in OPERATOR_EXTRA_FIELDS.items()
        },
        "integer_fields": list(INTEGER_FIELDS),
        "boolean_fields": list(BOOL_FIELDS),
        "semantic_rules": [
            "receipt must be default-off measurement/qualification evidence",
            "raw_pcg64_delta_available must be true",
            "promotion_blockers must be empty for operation qualification",
            "semantic_draws may be zero only when raw64_draws and raw32_draws are zero",
            "nonzero semantic_draws require nonzero raw64_draws or raw32_draws",
            "state and buffer before/after digests must be 64-char lowercase sha256",
            "has_uint32/uinteger before and after values must be explicitly recorded",
            "checkpoint continuation and failure atomicity must be bound per operation",
            "route promotion remains separately withheld after operation receipts pass",
        ],
    }


def _missing_fields(
    receipt: Mapping[str, Any],
    operator_id: str,
) -> list[str]:
    required = list(COMMON_REQUIRED_FIELDS)
    required.extend(OPERATOR_EXTRA_FIELDS[operator_id])
    return sorted(field for field in required if field not in receipt)


def _validate_integer_fields(receipt: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    for field in INTEGER_FIELDS:
        if field not in receipt:
            continue
        value = receipt.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            failures.append(f"{field}_must_be_non_negative_integer")
    return failures


def _validate_bool_fields(receipt: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    for field in BOOL_FIELDS:
        if field not in receipt:
            continue
        if not isinstance(receipt.get(field), bool):
            failures.append(f"{field}_must_be_boolean")
    return failures


def _validate_operation_receipt(
    receipt: Mapping[str, Any],
    operator_id: str,
) -> dict[str, Any]:
    failures: list[str] = []
    missing = _missing_fields(receipt, operator_id)
    if missing:
        failures.extend(f"missing_{field}" for field in missing)
    if str(receipt.get("operator_id") or "") != operator_id:
        failures.append("operator_id_mismatch")
    if str(receipt.get("physical_stream_id") or "") != ENV_POLICY_STREAM_ID:
        failures.append("physical_stream_id_mismatch")
    failures.extend(_validate_integer_fields(receipt))
    failures.extend(_validate_bool_fields(receipt))
    blockers = receipt.get("promotion_blockers")
    if not isinstance(blockers, list):
        failures.append("promotion_blockers_must_be_list")
    elif blockers:
        failures.append("promotion_blockers_not_empty")
    for field in (
        "state_before_digest",
        "state_after_digest",
        "buffer_before_digest",
        "buffer_after_digest",
    ):
        if field in receipt and not _hex_digest(receipt.get(field)):
            failures.append(f"{field}_must_be_sha256")
    if str(receipt.get("authority_class") or "") != (
        "measurement_only_not_route_promotion"
    ):
        failures.append("authority_class_must_be_measurement_only")
    if receipt.get("status_ok") is not True:
        failures.append("status_ok_not_true")
    if receipt.get("promotion_qualified") is not True:
        failures.append("promotion_qualified_not_true")
    if receipt.get("raw_pcg64_delta_available") is not True:
        failures.append("raw_pcg64_delta_not_available")
    if receipt.get("checkpoint_continuation_covered") is not True:
        failures.append("checkpoint_continuation_not_covered")
    if receipt.get("failure_atomicity_covered") is not True:
        failures.append("failure_atomicity_not_covered")

    semantic = receipt.get("semantic_draws")
    raw64 = receipt.get("raw64_draws")
    raw32 = receipt.get("raw32_draws")
    if (
        isinstance(semantic, int)
        and not isinstance(semantic, bool)
        and isinstance(raw64, int)
        and not isinstance(raw64, bool)
        and isinstance(raw32, int)
        and not isinstance(raw32, bool)
    ):
        if semantic == 0 and (raw64 != 0 or raw32 != 0):
            failures.append("zero_semantic_draws_must_have_zero_raw_draws")
        if semantic > 0 and raw64 == 0 and raw32 == 0:
            failures.append("nonzero_semantic_draws_need_raw_delta")

    if operator_id == "policy.action_gumbel.float32":
        if receipt.get("post_action_state_buffer_bound") is not True:
            failures.append("post_action_state_buffer_not_bound")
        if (
            isinstance(receipt.get("semantic_draws"), int)
            and isinstance(receipt.get("gumbel_float32_draws"), int)
            and receipt.get("semantic_draws") != receipt.get("gumbel_float32_draws")
        ):
            failures.append("semantic_draws_must_equal_gumbel_float32_draws")
        rows_total = (
            receipt.get("policy_cache_hit_rows"),
            receipt.get("policy_cache_miss_rows"),
            receipt.get("policy_rows_requested"),
        )
        if all(isinstance(value, int) for value in rows_total):
            if rows_total[0] + rows_total[1] != rows_total[2]:
                failures.append("policy_cache_rows_must_sum_to_requested_rows")

    if operator_id == "environment.food_spawn.uniform_or_bounded":
        if receipt.get("bounded_rejection_checkpoint_continuation_covered") is not True:
            failures.append("bounded_rejection_checkpoint_continuation_not_covered")
        if (
            isinstance(receipt.get("food_spawn_successes"), int)
            and isinstance(receipt.get("food_spawn_attempts"), int)
            and receipt.get("food_spawn_successes") > receipt.get("food_spawn_attempts")
        ):
            failures.append("food_spawn_successes_exceed_attempts")

    return {
        "operator_id": operator_id,
        "qualified": not failures,
        "failures": sorted(set(failures)),
    }


def decide(
    *,
    m27_receipt_path: Path,
    operation_receipt_paths: list[Path],
) -> dict[str, Any]:
    m27_receipt = _read_json_object(m27_receipt_path, name="M27 receipt")
    discovered = _discover_m27_operations(m27_receipt)
    blockers: list[str] = []
    if m27_receipt.get("decision") != (
        "M27-A-ENV-POLICY-PCG64-SURFACES-IDENTIFIED-PROMOTION-WITHHELD"
    ):
        blockers.append("m27_decision_not_bound")
    if discovered != sorted(REQUIRED_OPERATORS):
        blockers.append("m27_required_operations_not_exactly_bound")

    receipts_by_operator: dict[str, Mapping[str, Any]] = {}
    receipt_sources: list[dict[str, str]] = []
    for path in operation_receipt_paths:
        payload = _read_json_object(path, name="operation receipt")
        operator_id = str(payload.get("operator_id") or "")
        if operator_id not in REQUIRED_OPERATORS:
            blockers.append(f"unexpected_operation_receipt:{operator_id or '<missing>'}")
            continue
        if operator_id in receipts_by_operator:
            blockers.append(f"duplicate_operation_receipt:{operator_id}")
            continue
        receipts_by_operator[operator_id] = payload
        receipt_sources.append({"path": str(path), "sha256": _sha256(path)})

    missing_operation_receipts = [
        operator_id
        for operator_id in REQUIRED_OPERATORS
        if operator_id not in receipts_by_operator
    ]
    if missing_operation_receipts:
        blockers.append("env_policy_operation_receipts_missing")

    validations = [
        _validate_operation_receipt(receipts_by_operator[operator_id], operator_id)
        for operator_id in REQUIRED_OPERATORS
        if operator_id in receipts_by_operator
    ]
    for validation in validations:
        if not validation["qualified"]:
            blockers.append(f"operation_receipt_unqualified:{validation['operator_id']}")

    qualified_operations = [
        validation["operator_id"]
        for validation in validations
        if validation["qualified"]
    ]
    operation_receipts_qualified = sorted(qualified_operations) == sorted(
        REQUIRED_OPERATORS
    )
    if operation_receipts_qualified:
        blockers.append("route_level_checkpoint_projection_migration_not_bound")
        blockers.append("route_level_removal_and_performance_evidence_absent")

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "decision": DECISION_WITHHELD,
        "ok": not any(
            blocker.startswith("m27_") or blocker.startswith("unexpected_")
            for blocker in blockers
        ),
        "contract_defined": True,
        "operation_receipts_qualified": operation_receipts_qualified,
        "qualified_operations": sorted(qualified_operations),
        "missing_operation_receipts": missing_operation_receipts,
        "operation_validations": validations,
        "contract": _contract_definition(),
        "m27_receipt": {
            "path": str(m27_receipt_path),
            "sha256": _sha256(m27_receipt_path),
            "discovered_operations": discovered,
        },
        "operation_receipts": receipt_sources,
        "route_promotion_authorized": False,
        "promotion_blockers": sorted(set(blockers)),
        "recommended_next_step": (
            "instrument default-off env/policy operation receipts for food-spawn "
            "and policy-action PCG64 consumption, then prove continuation and "
            "failure atomicity before any route-level PCG64 removal"
        ),
        "no_claims": [
            "no PCG64 route promotion",
            "no checkpoint projection migration",
            "no production route removal",
            "no speedup claim",
            "no CUDA behavior change",
        ],
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result["receipt_sha256"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m27-receipt", type=Path, required=True)
    parser.add_argument(
        "--operation-receipt",
        type=Path,
        action="append",
        default=[],
        help="candidate env/policy operation receipt JSON; may be supplied twice",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = decide(
        m27_receipt_path=args.m27_receipt,
        operation_receipt_paths=list(args.operation_receipt),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0 if payload.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
