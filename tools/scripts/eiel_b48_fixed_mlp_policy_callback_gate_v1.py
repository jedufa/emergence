#!/usr/bin/env python3
"""Reduce exact fixed-MLP callback profiles into one fail-closed gate."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.campaigns.canonical import canonical_sha256  # noqa: E402
from engine.campaigns.orchestration import atomic_write_json  # noqa: E402
from tools.scripts.eiel_b48_fixed_mlp_policy_birth_refill_profile_v1 import (  # noqa: E402
    BENCHMARK_ARMS,
    CELLS,
    EPISODES,
    M7_BUILD_SCHEMA,
    M8_BUILD_SCHEMA,
    PACKED_ROW_BYTES,
    POLICY_WIDTH,
    REQUIRED_REPEATS,
)


SCHEMA_VERSION = "eiel_b48_fixed_mlp_policy_callback_gate_v1"
AUTHORITY = "fixed_mlp_component_callback_promotion_gate_v1"
MINIMUM_MEDIAN_SPEEDUP = 1.10
PROFILE_SCHEMA = "eiel_b48_fixed_mlp_policy_birth_refill_profile_v1"
PROFILE_AUTHORITY = "fixed_mlp_callback_component_profile_v1"
HARDWARE_FIELDS = {
    "device_index",
    "device_count",
    "device_name",
    "device_capability",
    "device_total_memory_bytes",
    "device_multiprocessor_count",
    "gpu_uuid",
    "driver_version",
    "torch_version",
    "torch_cuda_runtime_version",
    "identity_sha256",
}
EXACT_FIXTURE = {
    "cells": CELLS,
    "episodes": EPISODES,
    "policy_width": POLICY_WIDTH,
    "packed_row_bytes": PACKED_ROW_BYTES,
    "mutation_enable": False,
    "mutation_mode": "dense",
    "compact_mutation_width": 1,
}


class ComponentGateError(RuntimeError):
    """Raised when component profiles cannot support the callback gate."""


def _validated_digest_mapping(
    value: Any, *, schema: str, enabled_field: str, expected_enabled: bool
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ComponentGateError(f"component profile lacks {schema} identity")
    identity = dict(value)
    observed = identity.pop("identity_sha256", None)
    macro = {
        "option_d_m7_batched_cpu_policy_refill_enabled": (
            "EIEL_OPTION_D_M7_BATCHED_CPU_POLICY_REFILL_ENABLED"
        ),
        "option_d_m8_resident_policy_suffix_reuse_enabled": (
            "EIEL_OPTION_D_M8_RESIDENT_POLICY_SUFFIX_REUSE_ENABLED"
        ),
    }[enabled_field]
    expected_authority = {
        "option_d_m7_batched_cpu_policy_refill_enabled": (
            "candidate_batched_exact_cpu_lsb0_policy_refill"
        ),
        "option_d_m8_resident_policy_suffix_reuse_enabled": (
            "candidate_resident_policy_suffix_reuse"
        ),
    }[enabled_field]
    runtime_sha256 = identity.get("runtime_resource_receipt_sha256")
    if observed != canonical_sha256(identity):
        raise ComponentGateError(f"component {schema} identity digest mismatch")
    if (
        identity.get("schema_version") != schema
        or identity.get(enabled_field) is not expected_enabled
        or identity.get("compile_time_macro") != macro
        or identity.get("receipt_field") != enabled_field
        or identity.get("default_enabled") is not False
        or identity.get("compile_definition")
        != f"-D{macro}={1 if expected_enabled else 0}"
        or identity.get("authority")
        != (expected_authority if expected_enabled else "disabled")
        or identity.get("identity_authority") != "compiled_binary_resource_receipt"
        or identity.get("runtime_resource_receipt_confirmed") is not True
        or not isinstance(runtime_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", runtime_sha256) is None
    ):
        raise ComponentGateError(f"component profile has wrong {schema} identity")
    identity["identity_sha256"] = observed
    return identity


def _load_profile(path: Path, *, expected_arm: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ComponentGateError(f"component profile is absent: {resolved}")
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ComponentGateError("component profile is not a mapping")
    profile = dict(raw)
    observed_receipt_sha256 = profile.pop("receipt_sha256", None)
    if observed_receipt_sha256 != canonical_sha256(profile):
        raise ComponentGateError(f"component profile digest mismatch: {resolved}")
    profile["receipt_sha256"] = observed_receipt_sha256
    source_commit = profile.get("artifact_source_commit")
    host_commit = profile.get("host_profiler_source_commit")
    if (
        not isinstance(source_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
        or not isinstance(host_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", host_commit) is None
    ):
        raise ComponentGateError("component profile lacks exact source commit")
    for field in (
        "artifact_sha256",
        "manifest_sha256",
        "compiled_source_snapshot_sha256",
    ):
        value = profile.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ComponentGateError(f"component profile lacks exact {field}")
    if (
        profile.get("schema_version") != PROFILE_SCHEMA
        or profile.get("authority") != PROFILE_AUTHORITY
        or profile.get("status") != "success"
        or profile.get("benchmark_arm") != expected_arm
        or profile.get("required_repeats") != REQUIRED_REPEATS
        or profile.get("fixture") != EXACT_FIXTURE
    ):
        raise ComponentGateError(
            f"component profile is not the exact frozen {expected_arm} surface"
        )
    hardware = profile.get("hardware_identity")
    if not isinstance(hardware, Mapping) or set(hardware) != HARDWARE_FIELDS:
        raise ComponentGateError("component profile lacks exact hardware identity")
    hardware_payload = dict(hardware)
    hardware_sha256 = hardware_payload.pop("identity_sha256", None)
    if hardware_sha256 != canonical_sha256(hardware_payload):
        raise ComponentGateError("component hardware identity digest mismatch")
    build = profile.get("build_identity")
    if not isinstance(build, Mapping):
        raise ComponentGateError("component profile lacks build identity")
    expected_m7 = expected_arm != "legacy"
    expected_m8 = expected_arm == "m8_suffix_reuse"
    expected_operation = (
        "resident_policy_suffix_reuse" if expected_m8 else "physical_policy_refill"
    )
    if profile.get("callback_operation") != expected_operation or profile.get(
        "required_future_episodes"
    ) != (3 if expected_m8 else None):
        raise ComponentGateError("component profile has wrong callback operation")
    m7 = _validated_digest_mapping(
        build.get("m7"),
        schema=M7_BUILD_SCHEMA,
        enabled_field="option_d_m7_batched_cpu_policy_refill_enabled",
        expected_enabled=expected_m7,
    )
    m8 = _validated_digest_mapping(
        build.get("m8"),
        schema=M8_BUILD_SCHEMA,
        enabled_field="option_d_m8_resident_policy_suffix_reuse_enabled",
        expected_enabled=expected_m8,
    )
    result_identity = profile.get("result_identity")
    if (
        not isinstance(result_identity, Mapping)
        or result_identity.get("option_d_m7_batched_cpu_policy_refill_enabled")
        is not expected_m7
        or result_identity.get("option_d_m7_batched_cpu_policy_refill_used")
        is not (expected_m7 and not expected_m8)
    ):
        raise ComponentGateError("component runtime M7 identity mismatch")
    if expected_m8 and (
        result_identity.get("option_d_m8_resident_policy_suffix_reuse_enabled")
        is not True
        or result_identity.get("option_d_m8_resident_policy_suffix_reuse_used")
        is not True
        or result_identity.get("physical_refill_performed") is not False
        or result_identity.get("resident_window_reused") is not True
    ):
        raise ComponentGateError("component runtime M8 reuse identity mismatch")
    timing = profile.get("timing_ns")
    host = timing.get("host_wall_time_ns") if isinstance(timing, Mapping) else None
    if (
        not isinstance(host, Mapping)
        or host.get("count") != REQUIRED_REPEATS
        or isinstance(host.get("median_ns"), bool)
        or not isinstance(host.get("median_ns"), (int, float))
        or not math.isfinite(float(host["median_ns"]))
        or float(host["median_ns"]) <= 0.0
    ):
        raise ComponentGateError(
            "component profile lacks exact-five timing distribution"
        )
    output_digest = profile.get("output_digest")
    if not isinstance(output_digest, str) or len(output_digest) != 64:
        raise ComponentGateError("component profile lacks output digest")
    if expected_arm != "legacy" and (
        profile.get("expected_output_digest") != output_digest
        or profile.get("exact_control_digest_match") is not True
    ):
        raise ComponentGateError(
            "candidate profile lacks mandatory control digest match"
        )
    profile["build_identity"] = {**dict(build), "m7": m7, "m8": m8}
    profile["_path"] = str(resolved)
    return profile


def reduce_component_profiles(
    *, legacy_path: Path, m7_path: Path, m8_path: Path
) -> dict[str, Any]:
    paths = {
        "legacy": legacy_path,
        "m7_physical_refill": m7_path,
        "m8_suffix_reuse": m8_path,
    }
    profiles = {
        arm: _load_profile(paths[arm], expected_arm=arm) for arm in BENCHMARK_ARMS
    }
    profile_digests = [profiles[arm]["receipt_sha256"] for arm in BENCHMARK_ARMS]
    if len(set(profile_digests)) != len(profile_digests):
        raise ComponentGateError("component arms are not distinct profile receipts")
    source_commits = {
        profiles[arm].get("artifact_source_commit") for arm in BENCHMARK_ARMS
    }
    host_commits = {
        profiles[arm].get("host_profiler_source_commit") for arm in BENCHMARK_ARMS
    }
    hardware_identities = {
        canonical_sha256(profiles[arm]["hardware_identity"]) for arm in BENCHMARK_ARMS
    }
    if len(source_commits) != 1 or len(host_commits) != 1:
        raise ComponentGateError("component source commit differs between arms")
    if len(hardware_identities) != 1:
        raise ComponentGateError("component hardware identity differs between arms")
    output_digests = {profiles[arm]["output_digest"] for arm in BENCHMARK_ARMS}
    if len(output_digests) != 1:
        raise ComponentGateError("fixed-MLP callback output differs between arms")
    control_digest = profiles["legacy"]["output_digest"]
    for arm in ("m7_physical_refill", "m8_suffix_reuse"):
        if profiles[arm]["expected_output_digest"] != control_digest:
            raise ComponentGateError("candidate was not bound to this control digest")

    medians = {
        arm: float(profiles[arm]["timing_ns"]["host_wall_time_ns"]["median_ns"])
        for arm in BENCHMARK_ARMS
    }
    speedups = {
        arm: medians["legacy"] / medians[arm]
        for arm in ("m7_physical_refill", "m8_suffix_reuse")
    }
    gate_passed = all(value >= MINIMUM_MEDIAN_SPEEDUP for value in speedups.values())
    source_commit = next(iter(source_commits))
    host_commit = next(iter(host_commits))
    if source_commit != host_commit:
        raise ComponentGateError(
            "component profiler source commit does not match artifact source commit"
        )
    hardware_identity = dict(profiles["legacy"]["hardware_identity"])
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "status": "success",
        "component_surface": dict(EXACT_FIXTURE),
        "required_repeats_per_arm": REQUIRED_REPEATS,
        "artifact_source_commit": source_commit,
        "host_profiler_source_commit": host_commit,
        "hardware_identity": hardware_identity,
        "control_output_digest": control_digest,
        "exact_output_digest_match_across_arms": True,
        "m8_build_binds_m7_exact_physical_refill_fallback": True,
        "metric": "ratio_of_arm_medians_callback_host_wall_time_ns",
        "minimum_median_speedup_inclusive": MINIMUM_MEDIAN_SPEEDUP,
        "callback_median_gate_passed": gate_passed,
        "arms": {
            arm: {
                "profile_path": profiles[arm]["_path"],
                "profile_receipt_sha256": profiles[arm]["receipt_sha256"],
                "artifact_sha256": profiles[arm]["artifact_sha256"],
                "manifest_sha256": profiles[arm]["manifest_sha256"],
                "compiled_source_snapshot_sha256": profiles[arm][
                    "compiled_source_snapshot_sha256"
                ],
                "build_identity": {
                    "m7": profiles[arm]["build_identity"]["m7"],
                    "m8": profiles[arm]["build_identity"]["m8"],
                },
                "runtime_identity": profiles[arm]["result_identity"],
                "host_wall_time_ns": profiles[arm]["timing_ns"]["host_wall_time_ns"],
            }
            for arm in BENCHMARK_ARMS
        },
        "median_speedup_over_legacy": speedups,
        "qualification_claim": False,
        "automatic_promotion_authorized_by_this_receipt": False,
    }
    result["receipt_sha256"] = canonical_sha256(result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-profile", required=True, type=Path)
    parser.add_argument("--m7-profile", required=True, type=Path)
    parser.add_argument("--m8-profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ComponentGateError(f"refusing to overwrite component gate: {output}")
    receipt = reduce_component_profiles(
        legacy_path=args.legacy_profile,
        m7_path=args.m7_profile,
        m8_path=args.m8_profile,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
