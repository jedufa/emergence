from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "eiel_hapb_frag_oracle_compare_v2"
EXACTNESS_SCHEMA_VERSION = "eiel_hapb_h4_exactness_contract_v1"
EXPECTED_CELL_COUNT = 48
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    import hashlib

    return hashlib.sha256(encoded).hexdigest()


def _require_digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a 64-hex digest")
    return value.lower()


def _terminal_digests(receipt: Mapping[str, Any]) -> list[str]:
    raw: Any = None
    for key in (
        "terminal_projection_digests",
        "reference_terminal_projection_digests",
        "resumed_terminal_projection_digests",
    ):
        if key in receipt:
            raw = receipt[key]
            break
    if isinstance(raw, Mapping):
        for key in ("single", "reference", "resumed", "first"):
            if key in raw:
                raw = raw[key]
                break
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ValueError("terminal projection digests are missing")
    result = [_require_digest(value, "terminal_projection_digest") for value in raw]
    if not result:
        raise ValueError("terminal projection digests are empty")
    return result


def _cell_digest_map(
    raw: Any,
    *,
    cell_ids: Sequence[str],
    field: str,
) -> dict[str, str]:
    expected = set(cell_ids)
    if isinstance(raw, Mapping):
        actual = {str(key): value for key, value in raw.items()}
        if set(actual) != expected:
            raise ValueError(f"{field} is not identity-bound to the exact cell set")
        return {
            cell_id: _require_digest(actual[cell_id], f"{field}[{cell_id}]")
            for cell_id in cell_ids
        }
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        if len(raw) != len(cell_ids):
            raise ValueError(f"{field} does not contain one value per cell")
        return {
            cell_id: _require_digest(value, f"{field}[{cell_id}]")
            for cell_id, value in zip(cell_ids, raw)
        }
    raise ValueError(f"{field} is missing")


def _strict_contract(receipt: Mapping[str, Any]) -> dict[str, Any]:
    raw = receipt.get("h4_exactness_contract")
    if not isinstance(raw, Mapping):
        raise ValueError("h4_exactness_contract is missing")
    if raw.get("schema_version") != EXACTNESS_SCHEMA_VERSION:
        raise ValueError("unsupported H4 exactness contract schema")
    if raw.get("cell_count") != EXPECTED_CELL_COUNT:
        raise ValueError("H4 exactness contract must contain exactly 48 cells")
    raw_ids = raw.get("cell_ids")
    if (
        not isinstance(raw_ids, Sequence)
        or isinstance(raw_ids, (str, bytes, bytearray))
        or len(raw_ids) != EXPECTED_CELL_COUNT
    ):
        raise ValueError("H4 exactness contract cell_ids must contain 48 IDs")
    cell_ids = [str(value) for value in raw_ids]
    if len(set(cell_ids)) != EXPECTED_CELL_COUNT:
        raise ValueError("H4 exactness contract cell_ids are not unique")
    projections = raw.get("projections")
    if not isinstance(projections, Mapping):
        raise ValueError("H4 exactness projections are missing")
    projection_fields = (
        "reference_terminal",
        "midpoint",
        "reference_midpoint",
        "resumed_terminal",
    )
    normalized_projections = {
        field: _cell_digest_map(
            projections.get(field),
            cell_ids=cell_ids,
            field=f"projections.{field}",
        )
        for field in projection_fields
    }
    identity = raw.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("H4 exactness identity is missing")
    identity_fields = (
        "host_harness_source_commit",
        "artifact_source_commit",
        "artifact_sha256",
        "manifest_sha256",
        "execution_route_identity",
        "tick_native_control_sha256",
    )
    normalized_identity = {
        field: identity.get(field) for field in identity_fields
    }
    for field, value in normalized_identity.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"exactness identity field {field} is missing")
    observations = raw.get("observations")
    if not isinstance(observations, Mapping):
        raise ValueError("H4 exactness observations are missing")
    observation_fields = (
        "policy_traffic_sha256",
        "rng_trace_sha256",
        "action_cache_sha256",
    )
    normalized_observations = {
        field: observations.get(field) for field in observation_fields
    }
    for field, value in normalized_observations.items():
        _require_digest(value, f"observations.{field}")
    checks = raw.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("H4 exactness checks are missing")
    check_fields = (
        "cell_count_exact",
        "midpoint_reference_exact",
        "checkpoint_restore_exact",
        "fresh_owner_restore",
        "continuation_exact",
        "compatibility_alias_deletion_proof",
        "action_cache_observed",
    )
    normalized_checks = {}
    for field in check_fields:
        value = checks.get(field)
        if value is not True:
            raise ValueError(f"exactness check {field} is not proven")
        normalized_checks[field] = value
    return {
        "cell_ids": cell_ids,
        "projections": normalized_projections,
        "identity": normalized_identity,
        "observations": normalized_observations,
        "checks": normalized_checks,
    }


def campaign_run_to_h4_compare_receipt_v1(run_root: str | Path) -> dict[str, Any]:
    root = Path(run_root)
    if not root.exists():
        raise FileNotFoundError(root)
    candidates: list[dict[str, Any]] = []
    paths = [root] if root.is_file() else sorted(root.rglob("*.json"))
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        if "h4_exactness_contract" in payload:
            result = dict(payload)
            result["run_root"] = str(root)
            return result
        if any(
            key in payload
            for key in (
                "terminal_projection_digests",
                "reference_terminal_projection_digests",
                "resumed_terminal_projection_digests",
            )
        ):
            candidates.append(dict(payload))
    if not candidates:
        raise ValueError(f"no H4 projection receipt found under {root}")
    result = candidates[0]
    result["run_root"] = str(root)
    result["terminal_projection_digests"] = _terminal_digests(result)
    return result


def _comparison_failure(message: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "error": message,
        "terminal_projection_digests_equal": False,
        "digest_count": 0,
        "process_rng_after_execute_equal": False,
        "action_cache_equal": False,
        "policy_row_traffic_equal": False,
        "bit_equal": False,
        "qualification_claim": False,
        "production_execution_admission": False,
    }


def compare_h4_receipts_v1(
    oracle_receipt: Mapping[str, Any],
    candidate_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        oracle = _strict_contract(oracle_receipt)
        candidate = _strict_contract(candidate_receipt)
    except (TypeError, ValueError) as exc:
        return _comparison_failure(str(exc))

    projections_equal = oracle["projections"] == candidate["projections"]
    rng_equal = (
        oracle["observations"]["rng_trace_sha256"]
        == candidate["observations"]["rng_trace_sha256"]
    )
    action_cache_equal = (
        oracle["observations"]["action_cache_sha256"]
        == candidate["observations"]["action_cache_sha256"]
    )
    policy_traffic_equal = (
        oracle["observations"]["policy_traffic_sha256"]
        == candidate["observations"]["policy_traffic_sha256"]
    )
    identity_equal = oracle["identity"] == candidate["identity"]
    checks_equal = oracle["checks"] == candidate["checks"]
    bit_equal = (
        projections_equal
        and rng_equal
        and action_cache_equal
        and policy_traffic_equal
        and identity_equal
        and checks_equal
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "cell_ids": list(oracle["cell_ids"]),
        "digest_count": len(oracle["cell_ids"]),
        "digest0": oracle["projections"]["reference_terminal"][oracle["cell_ids"][0]],
        "terminal_projection_digests_equal": projections_equal,
        "midpoint_boundary_equal": (
            oracle["projections"]["midpoint"]
            == candidate["projections"]["midpoint"]
            and oracle["projections"]["reference_midpoint"]
            == candidate["projections"]["reference_midpoint"]
        ),
        "restore_boundary_exact": (
            oracle["checks"]["checkpoint_restore_exact"]
            and candidate["checks"]["checkpoint_restore_exact"]
        ),
        "continuation_exact": (
            oracle["checks"]["continuation_exact"]
            and candidate["checks"]["continuation_exact"]
        ),
        "source_artifact_route_control_identity_equal": identity_equal,
        "process_rng_after_execute_equal": rng_equal,
        "action_cache_equal": action_cache_equal,
        "policy_row_traffic_equal": policy_traffic_equal,
        "exactness_checks_equal": checks_equal,
        "bit_equal": bit_equal,
        "qualification_claim": False,
        "production_execution_admission": False,
    }
