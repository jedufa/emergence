#!/usr/bin/env python3
from __future__ import annotations

"""Audit the unresolved numpy.environment_policy PCG64 owner."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Final


SCHEMA_VERSION: Final[str] = "eiel_option_d_m27_env_policy_pcg64_readiness_audit_v1"
DECISION_WITHHELD: Final[str] = (
    "M27-A-ENV-POLICY-PCG64-SURFACES-IDENTIFIED-PROMOTION-WITHHELD"
)
REQUIRED_SOURCE_PATTERNS: Final[dict[str, tuple[str, ...]]] = {
    "engine/perf/cuda_architecture_b_flattened_controller_v1_kernels.cu": (
        "spawn_flat_food_exact",
        "consume_patch_policy_rows_serial",
        "service_patch_mlp_gumbel_policy_serial",
        "service_fixed_mlp_gumbel_policy_serial",
        "full_tick_state.shared_env_policy_pcg64_state_words",
        "full_tick_state.shared_env_policy_pcg64_buffer_words",
        "full_tick_state.policy_np_semantic_draws",
        "full_tick_outputs.policy_pcg64_state_after_action",
        "full_tick_outputs.policy_pcg64_buffer_after_action",
    ),
    "engine/authoritative_state/contracts/full_tick_event_v1.py": (
        '"pcg64_state_words"',
        '"pcg64_buffer_words"',
        '"policy_draws"',
        '"policy_rng_state"',
    ),
    "tools/scripts/eiel_option_d_m24_pcg64_route_promotion_decision.py": (
        '"numpy.environment_policy"',
        "bind or explicitly exclude numpy.environment_policy",
    ),
}


class M27Rejected(RuntimeError):
    """The audit input is missing or inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_hits(path: Path, patterns: tuple[str, ...]) -> dict[str, list[int]]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    hits: dict[str, list[int]] = {}
    for pattern in patterns:
        pattern_hits = [
            index
            for index, line in enumerate(lines, start=1)
            if pattern in line
        ]
        hits[pattern] = pattern_hits
    return hits


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise M27Rejected(f"{name} is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise M27Rejected(f"{name} must contain one JSON object")
    return payload


def decide(
    *,
    repo_root: Path,
    m15_receipt: Path,
    m24_receipt: Path,
) -> dict[str, Any]:
    repo_root = repo_root.expanduser().resolve()
    if not repo_root.is_dir():
        raise M27Rejected(f"repo root is missing: {repo_root}")
    m15 = _load_json(m15_receipt, name="M15 receipt")
    m24 = _load_json(m24_receipt, name="M24 receipt")
    active_streams = {
        str(row.get("physical_stream_id"))
        for row in m15.get("active_physical_streams", [])
        if isinstance(row, dict)
    }
    m24_missing = {
        str(item)
        for item in m24.get("missing_required_streams", [])
    }
    source_evidence: dict[str, Any] = {}
    missing_patterns: list[str] = []
    for relative, patterns in REQUIRED_SOURCE_PATTERNS.items():
        path = repo_root / relative
        if not path.is_file():
            raise M27Rejected(f"required source file is missing: {relative}")
        hits = _line_hits(path, patterns)
        for pattern, lines in hits.items():
            if not lines:
                missing_patterns.append(f"{relative}:{pattern}")
        source_evidence[relative] = {
            "sha256": _sha256(path),
            "pattern_hits": hits,
        }

    discovered_operations = [
        {
            "operator_id": "environment.food_spawn.uniform_or_bounded",
            "physical_stream_id": "numpy.environment_policy",
            "source": "spawn_flat_food_exact",
            "required_counter_contract": (
                "semantic food-spawn draws plus raw PCG64 delta and "
                "has_uint32/uinteger continuation"
            ),
        },
        {
            "operator_id": "policy.action_gumbel.float32",
            "physical_stream_id": "numpy.environment_policy",
            "source": (
                "consume_patch_policy_rows_serial / "
                "service_*_mlp_gumbel_policy_serial"
            ),
            "required_counter_contract": (
                "semantic gumbel draws, raw PCG64 delta, cache-hit/miss rows, "
                "and post-action state/buffer continuation"
            ),
        },
    ]
    blockers = []
    if "numpy.environment_policy" not in active_streams:
        blockers.append("m15_does_not_expose_numpy_environment_policy")
    if "numpy.environment_policy" not in m24_missing:
        blockers.append("m24_does_not_withhold_numpy_environment_policy")
    if missing_patterns:
        blockers.append("required_source_patterns_missing")
    blockers.extend(
        [
            "env_policy_operation_receipts_absent",
            "env_policy_raw_counter_contract_absent",
            "env_policy_alias_checkpoint_save_restore_once_unproven",
            "env_policy_odd_even_uint32_continuation_unproven",
            "env_policy_cancellation_failure_atomicity_unproven",
        ]
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "decision": DECISION_WITHHELD,
        "ok": not missing_patterns
        and "numpy.environment_policy" in active_streams
        and "numpy.environment_policy" in m24_missing,
        "repo_root": str(repo_root),
        "m15_receipt": {
            "path": str(m15_receipt),
            "sha256": _sha256(m15_receipt),
        },
        "m24_receipt": {
            "path": str(m24_receipt),
            "sha256": _sha256(m24_receipt),
        },
        "active_stream_confirmed": "numpy.environment_policy" in active_streams,
        "m24_withheld_confirmed": "numpy.environment_policy" in m24_missing,
        "source_evidence": source_evidence,
        "missing_source_patterns": missing_patterns,
        "discovered_operations": discovered_operations,
        "promotion_authorized": False,
        "receipt_implementation_authorized": True,
        "blockers": blockers,
        "recommended_next_step": (
            "add a default-off env/policy PCG64 operation receipt that records "
            "food-spawn and policy-action semantic/raw counters without "
            "changing RNG authority"
        ),
        "no_claims": [
            "no PCG64 route promotion",
            "no preparation or checkpoint surface removal",
            "no production speedup claim",
            "no W32/B48 qualification",
        ],
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result["receipt_sha256"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--m15-receipt", type=Path, required=True)
    parser.add_argument("--m24-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = decide(
        repo_root=args.repo_root,
        m15_receipt=args.m15_receipt,
        m24_receipt=args.m24_receipt,
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
