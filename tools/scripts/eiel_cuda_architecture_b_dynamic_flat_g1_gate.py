#!/usr/bin/env python3
from __future__ import annotations

"""Consolidate the fail-closed Dynamic Flat stochastic-g1 gate."""

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


GATE_ID = "CUDA-STAGE3-DYNAMIC-FLAT-G1-A/EXACT-CPU-AUTHORITY"
AUTHORITY_POLICY_ID = (
    "CUDA-STAGE3-AUTHORITY-POLICY-A/"
    "CPP-CPU-SOLE-INDEPENDENT-DYNAMIC-SEMANTIC-AUTHORITY"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def _deterministic_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    receipts = dict(payload.get("receipts") or {})
    cpu = dict(receipts.get("cpu_oracle") or {})
    cpu.pop("wall_time_ns", None)
    checkpoint = dict(
        dict(payload.get("authority_policy") or {}).get(
            "candidate_cpu_checkpoint_exact"
        )
        or {}
    )
    for field in (
        "device_to_host_bytes",
        "device_to_host_operations",
        "device_synchronizations",
    ):
        checkpoint.pop(field, None)
    return {
        "source_projection_digests": payload.get("source_projection_digests"),
        "cpu_oracle": cpu,
        "checkpoint": checkpoint,
        "generation": dict(payload.get("authority_policy") or {}).get(
            "cpu_generation_contract_exact"
        ),
        "birth_exactness": payload.get("birth_exactness"),
        "turnover_exactness": payload.get("turnover_exactness"),
    }


def _check_primary(payload: Mapping[str, Any]) -> dict[str, bool]:
    authority = dict(payload.get("authority_policy") or {})
    checkpoint = dict(authority.get("candidate_cpu_checkpoint_exact") or {})
    semantic = dict(checkpoint.get("authoritative_semantic_parity") or {})
    canonical = dict(checkpoint.get("canonical_checkpoint_storage_parity") or {})
    generation = dict(authority.get("cpu_generation_contract_exact") or {})
    launch = dict(dict(payload.get("receipts") or {}).get("persistent_launch") or {})
    controller = dict(launch.get("native_controller_receipt") or {})
    cpu = dict(dict(payload.get("receipts") or {}).get("cpu_oracle") or {})
    return {
        "receipt_ok": payload.get("ok") is True,
        "status_exact": payload.get("status") == "exact_dynamic_flat_g1_primary_route",
        "authority_policy": authority.get("decision") == AUTHORITY_POLICY_ID,
        "checkpoint_exact": checkpoint.get("ok") is True
        and checkpoint.get("first_mismatch") is None,
        "authoritative_semantic_exact": semantic.get("ok") is True
        and semantic.get("first_mismatch") is None,
        "canonical_storage_exact": canonical.get("ok") is True
        and canonical.get("first_mismatch") is None,
        "generation_contract_exact": generation.get("all_exact") is True
        and generation.get("first_mismatch") is None,
        "cpu_no_hidden_fallback": cpu.get("hidden_fallback") is False,
        "one_native_dispatch": controller.get("native_dispatches") == 1,
        "no_python_tick_iterations": controller.get("python_tick_iterations") == 0,
        "no_aten_hot_loop": controller.get("aten_operations_inside_tick_loop") == 0
        and launch.get("aten_allocations_in_extension_call") == 0
        and launch.get("aten_dispatches_in_extension_call") == 0,
        "no_runtime_jit": controller.get("runtime_jit") is False,
        "no_hidden_fallback": controller.get("hidden_fallback") is False,
        "no_host_sync_inside_loop": controller.get("host_synchronizations_inside_loop")
        == 0,
        "one_final_host_sync": controller.get("final_host_synchronizations") == 1,
        "complete_tick_and_turnover_launches": controller.get("total_kernel_launches")
        == 244,
    }


def _check_prefix(payload: Mapping[str, Any]) -> dict[str, bool]:
    authority = dict(payload.get("authority_policy") or {})
    stage2 = dict(authority.get("immutable_stage2") or {})
    return {
        "receipt_ok": payload.get("ok") is True,
        "stage2_independent": stage2.get("authority_label")
        == "independent_bounded_comparator"
        and stage2.get("independent_eligible") is True,
        "one_tick_prefix": authority.get("stage2_bounded_prefix_end_tick_exclusive")
        == 1,
        "raw_stage2_prefix_exact": authority.get("raw_stage2_comparator_exact")
        is True,
        "no_birth_in_prefix": authority.get("first_birth_tick") is None,
    }


def _check_sanitizer(payload: Mapping[str, Any], log_text: str) -> dict[str, bool]:
    authority = dict(payload.get("authority_policy") or {})
    checkpoint = dict(authority.get("candidate_cpu_checkpoint_exact") or {})
    generation = dict(authority.get("cpu_generation_contract_exact") or {})
    return {
        "receipt_ok": payload.get("ok") is True,
        "first_mismatch_null": checkpoint.get("first_mismatch") is None
        and generation.get("first_mismatch") is None,
        "memcheck_zero_errors": "ERROR SUMMARY: 0 errors" in log_text,
        "memcheck_no_error_record": "Invalid" not in log_text
        and "========= Error:" not in log_text,
    }


def _artifact_receipt(primary: Mapping[str, Any]) -> dict[str, Any]:
    identity = dict(primary.get("artifact_identity") or {})
    stage3 = dict(identity.get("persistent_stage3") or {})
    artifact = Path(str(stage3.get("artifact"))).resolve()
    manifest_path = Path(str(stage3.get("manifest"))).resolve()
    manifest = _read_json(manifest_path)
    actual_sha256 = _sha256(artifact)
    checks = {
        "artifact_exists": artifact.is_file(),
        "artifact_hash_matches_receipt": actual_sha256
        == stage3.get("artifact_sha256"),
        "artifact_hash_matches_manifest": actual_sha256
        == manifest.get("artifact_sha256"),
        "source_snapshot_stable": manifest.get("source_snapshot_stable") is True,
        "source_hashes_present": len(dict(manifest.get("source_hashes") or {})) == 27,
        "sass_only_sm75": manifest.get("sass_only") is True
        and manifest.get("cuda_arch") == "sm_75",
        "manifest_no_runtime_jit": manifest.get("runtime_jit_compilation") is False
        and manifest.get("runtime_jit_required_by_validation_route") is False,
    }
    return {
        "artifact": str(artifact),
        "artifact_sha256": actual_sha256,
        "manifest": str(manifest_path),
        "elf_build_id": manifest.get("elf_build_id"),
        "checks": checks,
    }


def _path_receipt(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def build_gate(
    *,
    primary_path: Path,
    repeat_path: Path,
    prefix_path: Path,
    sanitizer_path: Path,
    sanitizer_log_path: Path,
) -> dict[str, Any]:
    primary = _read_json(primary_path)
    repeat = _read_json(repeat_path)
    prefix = _read_json(prefix_path)
    sanitizer = _read_json(sanitizer_path)
    sanitizer_log = sanitizer_log_path.read_text(encoding="utf-8")
    primary_checks = _check_primary(primary)
    repeat_checks = _check_primary(repeat)
    repeat_checks["deterministic_receipts_equal"] = (
        _deterministic_receipt(primary) == _deterministic_receipt(repeat)
    )
    prefix_checks = _check_prefix(prefix)
    sanitizer_checks = _check_sanitizer(sanitizer, sanitizer_log)
    artifact = _artifact_receipt(primary)
    check_groups = {
        "primary": primary_checks,
        "repeat": repeat_checks,
        "stage2_prefix": prefix_checks,
        "sanitizer": sanitizer_checks,
        "artifact": artifact["checks"],
    }
    failed = [
        f"{group}.{name}"
        for group, checks in check_groups.items()
        for name, passed in checks.items()
        if not passed
    ]
    return {
        "schema_version": "eiel_cuda_dynamic_flat_g1_gate_v1",
        "gate_id": GATE_ID,
        "authority_policy_id": AUTHORITY_POLICY_ID,
        "ok": not failed,
        "first_mismatch": failed[0] if failed else None,
        "failed_checks": failed,
        "checks": check_groups,
        "artifact": artifact,
        "evidence": {
            "primary": _path_receipt(primary_path),
            "repeat": _path_receipt(repeat_path),
            "stage2_prefix": _path_receipt(prefix_path),
            "sanitizer": _path_receipt(sanitizer_path),
            "sanitizer_log": _path_receipt(sanitizer_log_path),
        },
        "decision": "exact_stochastic_g1_closed" if not failed else "g1_rejected",
        "next_gate": "exact_checkpoint_fresh_owner_g2" if not failed else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--repeat", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--sanitizer", type=Path, required=True)
    parser.add_argument("--sanitizer-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_gate(
        primary_path=args.primary,
        repeat_path=args.repeat,
        prefix_path=args.prefix,
        sanitizer_path=args.sanitizer,
        sanitizer_log_path=args.sanitizer_log,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": payload["ok"],
                "decision": payload["decision"],
                "first_mismatch": payload["first_mismatch"],
                "output": str(args.output.resolve()),
            }
        )
    )
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
