"""Route-bound Option D M15 live RNG inventory classifier.

The external Option-D V3 scanner is intentionally broad.  It treats generated
ABI rows, tests, tools, legacy Python surfaces, and active native-route state
fields as unresolved until a live EIEL pass supplies local ownership context.
This helper keeps that broad scan intact and adds the narrower M15 view: which
hits are active route authority, active route operators, generated echoes,
tooling/tests, legacy surfaces, or still blocked for production Option D.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "eiel_option_d_m15_live_rng_inventory_classification_v1"

ACTIVE_PHYSICAL_STREAMS: tuple[dict[str, Any], ...] = (
    {
        "physical_stream_id": "torch.mutation.policy_birth.cpu",
        "algorithm_id": "pytorch.cpu.mt19937.commit_449b1768",
        "checkpoint_fields": [
            "rng_manager_mutation_torch_cpu_state",
            "mutation_torch_cpu_semantic_draws",
        ],
        "m15_status": "d1_fixture_bound_device_candidate_exists",
    },
    {
        "physical_stream_id": "cpython.mutation.scalar",
        "algorithm_id": "cpython.random.mt19937.v3.3.14.4",
        "checkpoint_fields": [
            "python_random_words",
            "python_random_index",
            "python_random_has_gauss",
            "python_random_gauss_next",
            "python_random_raw_draws",
            "mutation_py_semantic_draws",
        ],
        "m15_status": "d2_d3_fixture_bound_cuda_authority_not_promoted",
    },
    {
        "physical_stream_id": "numpy.environment_policy",
        "algorithm_id": "numpy.pcg64.xsl_rr_128_64",
        "logical_aliases": ["numpy.environment", "numpy.policy"],
        "checkpoint_fields": [
            "native_runtime_numpy_env_state",
            "native_runtime_numpy_policy_state",
            "pcg64_state_words",
            "pcg64_buffer_words",
            "policy_np_semantic_draws",
        ],
        "m15_status": "aliased_physical_state_pcg64_promotion_blocked",
    },
    {
        "physical_stream_id": "numpy.runner",
        "algorithm_id": "numpy.pcg64.xsl_rr_128_64",
        "checkpoint_fields": [
            "native_runtime_numpy_runner_state",
            "runner_pcg64_state_words",
            "runner_pcg64_buffer_words",
            "metrics_np_semantic_draws",
        ],
        "m15_status": "pcg64_distribution_receipts_required",
    },
    {
        "physical_stream_id": "numpy.population",
        "algorithm_id": "numpy.pcg64.xsl_rr_128_64",
        "checkpoint_fields": [
            "native_runtime_numpy_population_state",
            "population_pcg64_state_words",
            "population_pcg64_buffer_words",
            "mutation_np_semantic_draws",
        ],
        "m15_status": "pcg64_newborn_drive_distribution_gap",
    },
    {
        "physical_stream_id": "numpy.birth_spawn_environment",
        "algorithm_id": "numpy.pcg64.xsl_rr_128_64",
        "checkpoint_fields": [
            "native_runtime_numpy_birth_spawn_env_state",
            "birth_planner_pcg64_state_before",
            "birth_planner_pcg64_buffer_before",
            "planner_pcg64_state_words",
            "planner_pcg64_buffer_words",
        ],
        "m15_status": "pcg64_integer_operator_receipt_required",
    },
)

ACTIVE_OPERATOR_GAPS: tuple[dict[str, str], ...] = (
    {
        "operator_id": "population.newborn_drive.standard_normal",
        "stream_id": "numpy.population",
        "source": "engine/perf/cuda_architecture_b_persistent_turnover_drives.cuh",
        "reason": (
            "Live source consumes NumPy-compatible standard normals through the "
            "population PCG64 cursor, but the V3.1 operator inventory does not "
            "name this operator separately."
        ),
        "required_next_evidence": (
            "Add a live operation receipt for raw/semantic draw deltas, state "
            "continuation, and checkpoint resume before PCG64 promotion."
        ),
    },
    {
        "operator_id": "runner.drive_noise.standard_normal",
        "stream_id": "numpy.runner",
        "source": "engine/perf/cuda_architecture_b_persistent_chemistry.cuh",
        "reason": (
            "The operator is inventoried, but distribution implementation and "
            "host-refill/device-materialization authority remain promotion gates."
        ),
        "required_next_evidence": (
            "Bind live receipts for refill path, device path, draw counts, and "
            "state continuation under the accepted route."
        ),
    },
    {
        "operator_id": "turnover.spawn_coordinate",
        "stream_id": "numpy.birth_spawn_environment",
        "source": "engine/perf/cuda_architecture_b_persistent_turnover_plan.cuh",
        "reason": (
            "The operator is inventoried, but the live route still needs a "
            "production receipt for bounded integer rejection behavior."
        ),
        "required_next_evidence": (
            "Record raw32 rejection counts, semantic coordinate draws, failure "
            "atomicity, and checkpoint continuation."
        ),
    },
)


@dataclass(frozen=True)
class Classification:
    category: str
    route_relevance: str
    stream_id: str | None = None
    operator_id: str | None = None
    note: str | None = None


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} did not contain a JSON object")
    return value


def _git_value(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=repo_root,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _fingerprint_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def _path_text(hit: Mapping[str, Any]) -> tuple[str, str]:
    return str(hit.get("path", "")), str(hit.get("text", ""))


def classify_hit(hit: Mapping[str, Any]) -> Classification:
    path, text = _path_text(hit)
    kind = str(hit.get("kind", ""))
    line = int(hit.get("line", 0) or 0)

    if path.startswith("tests/"):
        return Classification(
            "test_reference", "excluded_from_production_route_inventory"
        )
    if path.startswith("tools/"):
        return Classification(
            "tooling_or_review_reference",
            "excluded_from_production_route_inventory",
        )
    if "/generated/" in path:
        return Classification(
            "generated_schema_echo",
            "derived_from_authoritative_contracts_not_independent_stream",
        )
    if path.startswith("engine/agents/"):
        return Classification(
            "legacy_agent_python_surface",
            "not_selected_by_current_cuda_architecture_b_production_route",
            note="Must be inventoried separately if legacy agent execution is targeted.",
        )

    if path.endswith("full_tick_event_v1.py"):
        if "runner_pcg64" in text:
            return Classification(
                "active_route_state_contract",
                "active_production_route_state",
                "numpy.runner",
            )
        if "pcg64_" in text:
            return Classification(
                "active_route_state_contract",
                "active_production_route_state",
                "numpy.environment_policy",
            )
    if path.endswith("ordinary_turnover_v1.py"):
        if "python_random" in text or "mutation_py" in text:
            return Classification(
                "active_route_state_contract",
                "active_production_route_state",
                "cpython.mutation.scalar",
            )
        if "population_pcg64" in text or "mutation_np" in text:
            return Classification(
                "active_route_state_contract",
                "active_production_route_state",
                "numpy.population",
            )
        if 218 <= line <= 228 and "PCG64" in text:
            return Classification(
                "active_route_state_contract",
                "active_production_route_state",
                "numpy.population",
            )
        if "policy_birth_" in text or "mutation_torch" in text:
            return Classification(
                "active_route_state_contract",
                "active_production_route_state",
                "torch.mutation.policy_birth.cpu",
            )
        if "pcg64_" in text or "policy_np" in text:
            return Classification(
                "active_route_state_contract",
                "active_production_route_state",
                "numpy.environment_policy",
            )
        if 290 <= line <= 300 and "PCG64" in text:
            return Classification(
                "active_route_state_contract",
                "active_production_route_state",
                "numpy.environment_policy",
            )
    if path.endswith("outer_output_patch_v1.py") and "birth_planner_pcg64" in text:
        return Classification(
            "active_route_projection_contract",
            "active_projection_input_not_checkpoint_authority",
            "numpy.birth_spawn_environment",
        )

    if path.endswith("cuda_architecture_b_persistent_generation.hpp"):
        if "runner_pcg64" in text:
            return Classification(
                "active_route_native_abi_pointer",
                "active_production_route_state",
                "numpy.runner",
            )
        if "shared_env_policy_pcg64" in text:
            return Classification(
                "active_route_native_abi_pointer",
                "active_production_route_state",
                "numpy.environment_policy",
            )
        if "population_pcg64" in text:
            return Classification(
                "active_route_native_abi_pointer",
                "active_production_route_state",
                "numpy.population",
            )
        if "python_random" in text:
            return Classification(
                "active_route_native_abi_pointer",
                "active_production_route_state",
                "cpython.mutation.scalar",
            )
        if "policy_birth" in text or "mutation_torch" in text:
            return Classification(
                "active_route_native_abi_pointer",
                "active_production_route_state",
                "torch.mutation.policy_birth.cpu",
            )

    if path.endswith("cuda_architecture_b_checkpoint.py"):
        if "native_runtime_numpy_runner_state" in text or "runner_pcg64" in text:
            return Classification(
                "active_checkpoint_projection",
                "active_checkpoint_metadata",
                "numpy.runner",
            )
        if (
            "native_runtime_numpy_env_state" in text
            or "native_runtime_numpy_policy_state" in text
            or "env_pcg64" in text
            or "pcg64_state_words" in text
        ):
            return Classification(
                "active_checkpoint_projection",
                "active_checkpoint_metadata",
                "numpy.environment_policy",
            )
        if "native_runtime_numpy_population_state" in text:
            return Classification(
                "active_checkpoint_projection",
                "active_checkpoint_metadata",
                "numpy.population",
            )
        if "native_runtime_numpy_birth_spawn_env_state" in text:
            return Classification(
                "active_checkpoint_projection",
                "active_checkpoint_metadata",
                "numpy.birth_spawn_environment",
            )
        if "native_runtime_python_random_state" in text or "python_random" in text:
            return Classification(
                "active_checkpoint_projection",
                "active_checkpoint_metadata",
                "cpython.mutation.scalar",
            )

    if path.endswith("cuda_architecture_b_direct_resident_source_allocator_v1.py"):
        if "runner_pcg64" in text:
            return Classification(
                "active_prepared_source_mapping",
                "active_boundary_mapping",
                "numpy.runner",
            )
        if "population_pcg64" in text:
            return Classification(
                "active_prepared_source_mapping",
                "active_boundary_mapping",
                "numpy.population",
            )
        if "pcg64_" in text:
            return Classification(
                "active_prepared_source_mapping",
                "active_boundary_mapping",
                "numpy.environment_policy",
            )

    if path.endswith("cuda_architecture_b_persistent_turnover_plan.cuh"):
        if "pcg64" in text.lower():
            return Classification(
                "active_route_operator",
                "active_operator_receipt_required",
                "numpy.birth_spawn_environment",
                "turnover.spawn_coordinate",
            )
        if "getrandbits" in text or "randbelow" in text:
            return Classification(
                "active_route_operator",
                "active_operator_receipt_required",
                "cpython.mutation.scalar",
                "scalar_mutation.randbelow",
            )
    if path.endswith("cuda_architecture_b_turnover.cu"):
        if "pcg64" in text.lower():
            return Classification(
                "active_route_operator",
                "active_operator_receipt_required",
                "numpy.birth_spawn_environment",
                "turnover.spawn_coordinate",
            )
        if "getrandbits" in text or "randbelow" in text:
            return Classification(
                "active_route_operator",
                "active_operator_receipt_required",
                "cpython.mutation.scalar",
                "scalar_mutation.randbelow",
            )
    if path.endswith("cuda_architecture_b_persistent_turnover_drives.cuh"):
        return Classification(
            "active_route_operator_gap",
            "blocks_pcg64_promotion_until_named_and_receipted",
            "numpy.population",
            "population.newborn_drive.standard_normal",
        )
    if path.endswith("cuda_architecture_b_persistent_chemistry.cuh"):
        return Classification(
            "active_route_shared_rng_helper",
            "active_operator_receipt_required",
            "numpy.runner_or_population",
            "pcg64.standard_normal.helper",
        )
    if path.endswith("cuda_architecture_b_flattened_wdl_controller_v2_kernels.cu"):
        if "runner_pcg64" in text or "drive_noise" in text:
            return Classification(
                "adjacent_or_active_native_kernel_surface",
                "requires_route_specific_receipt_before_promotion",
                "numpy.runner",
                "runner.drive_noise.standard_normal",
            )
        if "population_pcg64" in text:
            return Classification(
                "adjacent_or_active_native_kernel_surface",
                "requires_route_specific_receipt_before_promotion",
                "numpy.population",
            )
        if "shared_env_policy_pcg64" in text or "policy_rng" in text:
            return Classification(
                "adjacent_or_active_native_kernel_surface",
                "requires_route_specific_receipt_before_promotion",
                "numpy.environment_policy",
            )

    if path.endswith("cuda_architecture_b_child_policy.py"):
        return Classification(
            "active_policy_torch_operator_source",
            "active_operator_receipt_required",
            "torch.mutation.policy_birth.cpu",
            "ordinary_or_turnover_policy_crossover",
        )
    if "DeviceCpython" in text or "getrandbits" in text or "randbelow" in text:
        if path.startswith("engine/perf/") or path.startswith("engine/perf/native/"):
            return Classification(
                "active_or_adjacent_cpython_native_surface",
                "requires_route_specific_receipt_before_promotion",
                "cpython.mutation.scalar",
            )
    if "python_random" in text or "gauss" in text:
        if path.startswith("engine/perf/") or path.startswith("engine/perf/native/"):
            return Classification(
                "active_or_adjacent_cpython_native_surface",
                "requires_route_specific_receipt_before_promotion",
                "cpython.mutation.scalar",
            )

    if path.startswith("engine/perf/native/cuda_architecture_b_pcg64"):
        return Classification(
            "active_pcg64_native_helper",
            "requires_route_specific_receipt_before_promotion",
            "numpy.pcg64.shared_helper",
        )
    if path.startswith("engine/perf/native/cxxbackbone10_resident_state_owner.cpp"):
        return Classification(
            "native_resident_state_owner_rng_surface",
            "route_projection_context_not_independent_rng_stream",
        )
    if path.startswith("engine/perf/native/resident_state_contract.hpp"):
        return Classification(
            "native_resident_state_contract_echo",
            "route_projection_context_not_independent_rng_stream",
        )
    if path.startswith("engine/perf/cuda_architecture_b_pcg64.py"):
        return Classification(
            "active_pcg64_python_helper",
            "requires_route_specific_receipt_before_promotion",
            "numpy.pcg64.shared_helper",
        )
    if path.startswith("engine/perf/native/cuda_architecture_b"):
        return Classification(
            "active_or_adjacent_cuda_architecture_b_native_surface",
            "requires_route_specific_receipt_before_promotion",
        )
    if path.startswith("engine/perf/cuda_architecture_b"):
        return Classification(
            "active_or_adjacent_cuda_architecture_b_surface",
            "requires_route_specific_receipt_before_promotion",
        )
    if path.startswith("engine/perf/gmnw_live_gru_contract_v1.py"):
        return Classification(
            "adjacent_gmnw_contract_surface",
            "not_selected_by_current_w32_cuda_architecture_b_route",
        )
    if path.startswith("engine/native_runtime/"):
        return Classification(
            "native_runtime_projection_or_reporting_surface",
            "route_projection_context_not_independent_rng_stream",
        )
    if path.startswith("engine/authoritative_state/runtime_abi.py"):
        return Classification(
            "runtime_abi_schema_echo",
            "derived_from_authoritative_contracts_not_independent_stream",
        )
    if path.startswith("engine/campaigns/"):
        return Classification(
            "campaign_or_provider_seed_surface",
            "outside_cuda_inner_loop_but_provider_init_must_remain_bound",
        )
    if (
        path.startswith("engine/simulation/")
        or path.startswith("engine/evolution/")
        or path.startswith("engine/env/")
        or path == "engine/main.py"
    ):
        return Classification(
            "legacy_or_non_cuda_python_runtime_surface",
            "not_selected_by_current_cuda_architecture_b_production_route",
        )

    if path.startswith("engine/perf/eiel_phase_") or path.startswith("engine/perf/cxx"):
        return Classification(
            "legacy_perf_or_fixture_surface",
            "not_selected_by_current_cuda_architecture_b_production_route",
        )
    if path.startswith("engine/"):
        return Classification(
            "unresolved_engine_rng_surface",
            "requires_manual_route_authority_decision",
            note=f"kind={kind}",
        )
    return Classification("unresolved_non_engine_surface", "requires_manual_review")


def _stream_inventory_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    streams = payload.get("streams", [])
    if not isinstance(streams, Sequence) or isinstance(streams, (str, bytes)):
        raise ValueError("stream inventory streams must be an array")
    ids = [row.get("stream_id") for row in streams if isinstance(row, Mapping)]
    return {
        "stream_count": len(streams),
        "stream_ids": ids,
        "accepted_live_reconciliation": payload.get("accepted_live_reconciliation"),
        "unique_physical_state_count_if_alias_preserved": payload.get(
            "unique_physical_state_count_if_alias_preserved"
        ),
    }


def _operator_inventory_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    operators = payload.get("operators", [])
    if not isinstance(operators, Sequence) or isinstance(operators, (str, bytes)):
        raise ValueError("operator inventory operators must be an array")
    ids = [row.get("operator_id") for row in operators if isinstance(row, Mapping)]
    return {
        "operator_count": len(operators),
        "operator_ids": ids,
        "accepted_live_reconciliation": payload.get("accepted_live_reconciliation"),
    }


def build_receipt(
    *,
    repo_root: Path,
    scan_path: Path,
    stream_inventory_path: Path,
    operator_inventory_path: Path,
) -> dict[str, Any]:
    scan = _read_json(scan_path)
    stream_inventory = _read_json(stream_inventory_path)
    operator_inventory = _read_json(operator_inventory_path)
    hits: list[Mapping[str, Any]] = []
    for key in ("classified", "unknown", "blocked"):
        rows = scan.get(key, [])
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise ValueError(f"scan field {key} must be an array")
        for row in rows:
            if isinstance(row, Mapping):
                item = dict(row)
                item["_source_scan_bucket"] = key
                hits.append(item)

    classified_rows: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    route_relevance_counts: Counter[str] = Counter()
    stream_counts: Counter[str] = Counter()
    unresolved: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    for hit in hits:
        result = classify_hit(hit)
        category_counts[result.category] += 1
        route_relevance_counts[result.route_relevance] += 1
        if result.stream_id is not None:
            stream_counts[result.stream_id] += 1
        row = {
            "source_scan_bucket": hit.get("_source_scan_bucket"),
            "path": hit.get("path"),
            "line": hit.get("line"),
            "kind": hit.get("kind"),
            "category": result.category,
            "route_relevance": result.route_relevance,
            "stream_id": result.stream_id,
            "operator_id": result.operator_id,
            "note": result.note,
            "context_sha256": hit.get("context_sha256"),
        }
        classified_rows.append(row)
        if result.category.startswith("unresolved"):
            unresolved.append(row)
        if result.category == "active_route_operator_gap":
            gap_rows.append(row)

    operator_summary = _operator_inventory_summary(operator_inventory)
    missing_gap_ids = sorted(
        {
            gap["operator_id"]
            for gap in ACTIVE_OPERATOR_GAPS
            if gap["operator_id"] not in set(operator_summary["operator_ids"])
        }
    )
    passed = not unresolved
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "passed": passed,
        "decision": (
            "M15-A-ROUTE-BOUND-INVENTORY-CLASSIFIED-WITH-PCG64-PROMOTION-GAPS"
            if passed
            else "M15-B-ROUTE-BOUND-INVENTORY-HAS-UNRESOLVED-ENGINE-HITS"
        ),
        "repo": str(repo_root),
        "git_branch": _git_value(repo_root, "branch", "--show-current"),
        "git_head": _git_value(repo_root, "rev-parse", "HEAD"),
        "git_status_short": _git_value(repo_root, "status", "--short"),
        "inputs": {
            "scan_path": str(scan_path),
            "scan_sha256": _fingerprint_bytes(scan_path.read_bytes()),
            "stream_inventory_path": str(stream_inventory_path),
            "stream_inventory_sha256": _fingerprint_bytes(
                stream_inventory_path.read_bytes()
            ),
            "operator_inventory_path": str(operator_inventory_path),
            "operator_inventory_sha256": _fingerprint_bytes(
                operator_inventory_path.read_bytes()
            ),
        },
        "broad_scan_totals": {
            "hits_total": scan.get("hits_total"),
            "classified_total": scan.get("classified_total"),
            "unknown_total": scan.get("unknown_total"),
            "blocked_total": scan.get("blocked_total"),
            "suspicious_device_semantics": len(
                scan.get("suspicious_device_semantics", [])
            )
            if isinstance(scan.get("suspicious_device_semantics"), list)
            else None,
        },
        "m15_counts": {
            "hit_rows_processed": len(hits),
            "category_counts": dict(sorted(category_counts.items())),
            "route_relevance_counts": dict(sorted(route_relevance_counts.items())),
            "stream_counts": dict(sorted(stream_counts.items())),
            "unresolved_engine_hit_count": len(unresolved),
            "active_operator_gap_hit_count": len(gap_rows),
        },
        "stream_inventory_summary": _stream_inventory_summary(stream_inventory),
        "operator_inventory_summary": operator_summary,
        "active_physical_streams": ACTIVE_PHYSICAL_STREAMS,
        "physical_stream_count_if_environment_policy_alias_preserved": len(
            ACTIVE_PHYSICAL_STREAMS
        ),
        "promotion_blockers": {
            "missing_operator_gap_ids_from_v31_inventory": missing_gap_ids,
            "active_operator_gaps": ACTIVE_OPERATOR_GAPS,
            "unresolved_engine_hits": unresolved[:100],
            "unresolved_engine_hits_truncated": len(unresolved) > 100,
        },
        "classified_rows_sha256": _fingerprint_bytes(
            json.dumps(classified_rows, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ),
    }
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    receipt["receipt_sha256"] = _fingerprint_bytes(encoded)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--stream-inventory", type=Path, required=True)
    parser.add_argument("--operator-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    receipt = build_receipt(
        repo_root=args.repo_root.resolve(),
        scan_path=args.scan.resolve(),
        stream_inventory_path=args.stream_inventory.resolve(),
        operator_inventory_path=args.operator_inventory.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if receipt["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
