#!/usr/bin/env python3
"""Materialize the CPU-only Gate 7/Gate 8 HAPB readiness packet.

This script deliberately performs no CUDA import, allocation, or execution.  It
freezes the benchmark contract and validates the current source/artifact
lineage so that the later GPU work is a bounded execution step rather than a
methodology decision made while timing is in progress.  The packet is
readiness evidence only: it never claims Gate 6, Q5, qualification, or
production admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.compiler import load_campaign_spec


SCHEMA_VERSION = "eiel_hapb_gate7_gate8_cpu_packet_v1"
ROW_BYTES = 39_964
CANONICAL_B48 = 48
HAPB_HOT_PREFIX_ROWS = 256
HAPB_DEFAULT_TILE_ROWS = 512
HAPB_EXISTING_TILE_ROWS = (512, 4096)
REQUIRED_BENCHMARK_CANDIDATES = (48, 60, 72, 84, 96)
OPTIONAL_BENCHMARK_CANDIDATES = (108, 120)
UPPER_FRONTIER_CANDIDATES = (128, 130, 131)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def _git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _git_non_runtime_delta(root: Path, runtime_commit: str, current_head: str) -> list[str]:
    if runtime_commit == current_head:
        return []
    subprocess.check_call(["git", "cat-file", "-e", f"{runtime_commit}^{{commit}}"], cwd=root)
    output = subprocess.check_output(
        ["git", "diff", "--name-only", f"{runtime_commit}..{current_head}"],
        cwd=root,
        text=True,
    )
    paths = [line for line in output.splitlines() if line]
    allowed = {
        "tools/scripts/eiel_hapb_gate7_gate8_cpu_packet_v1.py",
        "tools/scripts/eiel_hapb_future_gate_automation_v1.py",
        "tools/scripts/eiel_b48_wdl_jungle_eed_hapb_h4_continuation_v1.py",
        "tools/scripts/eiel_hapb_gate8_benchmark_analysis_v1.py",
        "tools/scripts/eiel_hapb_gate8_measurement_from_h4_v1.py",
        "tests/test_eiel_hapb_gate7_gate8_cpu_packet_v1.py",
        "tests/test_eiel_hapb_contended_shared_gpu_v1.py",
        "tests/test_eiel_hapb_gate8_measurement_from_h4_v1.py",
        "tests/test_eiel_hapb_high_b_abi_v1.py",
        "tests/test_eiel_cuda_architecture_b_hapb_v1.py",
        "engine/perf/hapb_contended_shared_gpu_v1.py",
        "engine/perf/cuda_architecture_b_direct_resident_wdl_jungle_v1.py",
        "engine/campaigns/production_engine_native_bindings.py",
        "_spec/campaigns/v1/wdl_jungle_eed_gv2_birth_w32p256c512_b48g200t160cuda_active_impact_allocation_diagnostic_v1.yaml",
    }
    _require(set(paths) <= allowed, "runtime source commit has an unapproved non-runtime delta")
    return paths


def _validate_artifact(
    *, artifact: Path, manifest: Path, expected_source_commit: str
) -> dict[str, Any]:
    _require(artifact.is_file(), f"artifact is missing: {artifact}")
    _require(manifest.is_file(), f"manifest is missing: {manifest}")
    artifact_sha = _sha256(artifact)
    manifest_sha = _sha256(manifest)
    payload = _load_json(manifest)
    _require(payload.get("artifact_sha256") == artifact_sha, "manifest artifact digest differs")
    snapshot = payload.get("committed_source_snapshot")
    _require(isinstance(snapshot, Mapping), "manifest source snapshot is missing")
    source_commit = snapshot.get("build_source_commit")
    _require(source_commit == expected_source_commit, "artifact source commit differs from current HEAD")
    return {
        "path": str(artifact),
        "sha256": artifact_sha,
        "manifest_path": str(manifest),
        "manifest_sha256": manifest_sha,
        "source_commit": source_commit,
        "abi": payload.get("abi"),
        "manifest_schema_version": payload.get("schema_version"),
    }


def _validate_source_factory(
    *, source_root: Path, expected_source_commit: str, expected_artifact_sha256: str
) -> dict[str, Any]:
    receipt_path = source_root / "source_factory_receipt.json"
    _require(receipt_path.is_file(), f"source factory receipt is missing: {receipt_path}")
    receipt = _load_json(receipt_path)
    _require(receipt.get("schema_version") == "eiel_campaign_source_factory_v2", "source factory schema differs")
    rows = receipt.get("rows")
    _require(isinstance(rows, list) and len(rows) == CANONICAL_B48, "source factory is not exactly 48 rows")
    cell_ids = [row.get("cell_id") for row in rows if isinstance(row, Mapping)]
    _require(len(cell_ids) == CANONICAL_B48 and len(set(cell_ids)) == CANONICAL_B48, "source factory cell identities are not distinct")
    projection_digests: list[str] = []
    for index, row in enumerate(rows):
        _require(isinstance(row, Mapping), f"source factory row {index} is malformed")
        projection_path = Path(str(row.get("projection_path", "")))
        _require(projection_path.is_file(), f"source projection {index} is missing")
        actual = _sha256(projection_path)
        _require(actual == row.get("projection_sha256"), f"source projection {index} digest differs")
        _require(row.get("sharing_mode") == "distinct_per_cell", f"source projection {index} is shared")
        projection_digests.append(actual)
    direct_receipt_path = source_root / "hapb_ordinary_birth_direct_receipt.json"
    _require(direct_receipt_path.is_file(), f"source factory direct receipt is missing: {direct_receipt_path}")
    direct_receipt = _load_json(direct_receipt_path)
    factory_artifact_sha256 = direct_receipt.get("artifact_sha256")
    _require(
        isinstance(factory_artifact_sha256, str) and len(factory_artifact_sha256) == 64,
        "source factory direct receipt lacks an artifact digest",
    )
    factory_runtime_artifact_bound = factory_artifact_sha256 == expected_artifact_sha256
    _require(direct_receipt.get("production_execution_admission") is False, "source factory direct receipt claims production admission")
    _require(direct_receipt.get("qualification_claim") is False, "source factory direct receipt claims qualification")
    return {
        "root": str(source_root),
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha256(receipt_path),
        "direct_receipt_path": str(direct_receipt_path),
        "direct_receipt_sha256": _sha256(direct_receipt_path),
        "direct_receipt_artifact_sha256": factory_artifact_sha256,
        "runtime_artifact_bound": factory_runtime_artifact_bound,
        "schema_version": receipt.get("schema_version"),
        "rows": len(rows),
        "distinct_cell_ids": len(set(cell_ids)),
        "sharing_mode": "distinct_per_cell",
        "source_contract_sha256": receipt.get("source_contract_sha256"),
        "plan_sha256": receipt.get("plan_sha256"),
        "projection_digests_sha256": hashlib.sha256(
            "".join(projection_digests).encode("ascii")
        ).hexdigest(),
        "source_commit": expected_source_commit,
    }


def _validate_interruption(root: Path) -> dict[str, Any]:
    path = root / "continuation_failure_diagnostic.json"
    _require(path.is_file(), f"preserved interruption diagnostic is missing: {path}")
    payload = _load_json(path)
    _require(payload.get("error", "").startswith("KeyboardInterrupt"), "preserved root is not an operator interruption")
    return {
        "root": str(root),
        "diagnostic_path": str(path),
        "diagnostic_sha256": _sha256(path),
        "stage": payload.get("stage"),
        "error": payload.get("error"),
        "usable_as_pass_evidence": False,
    }


def _benchmark_contract() -> dict[str, Any]:
    return {
        "authority": "merged HAPB end-to-end closure prompt",
        "control": "current Resident B48 on the same canonical EIEL_FULL_EVOLUTION_V1 route",
        "hapb_reference": "current implemented HAPB B48: P256 hot-prefix plus pageable overflow tile",
        "required_candidates": list(REQUIRED_BENCHMARK_CANDIDATES),
        "optional_candidates": list(OPTIONAL_BENCHMARK_CANDIDATES),
        "upper_frontier_candidates": list(UPPER_FRONTIER_CANDIDATES),
        "upper_frontier_rule": (
            "Evaluate B128/B130/B131 when throughput is still improving and admission remains valid; "
            "do not impose a B%8 restriction. B120 is a coarse point, not a ceiling."
        ),
        "treatment_extension": {
            "status": "PARAMETERIZED_B_FROM_BANK_EXTENT",
            "base_matrix": "the exact ordered current-source B48 source-factory rows",
            "extension": (
                "Physical B is the leading extent of the canonical host bank / source batch. "
                "C512/P256/9991 stay fixed. No B%8 restriction. B48 uses factory 200."
            ),
            "authority": "hapb_physical_cell_count_v1 plus native physical_cells_v1",
            "not_yet_implemented": False,
        },
        "memory_admission": {
            "physical_margin_formula": "minimum_free_at_peak - ceil(device_total * 0.10)",
            "experimental_margin_formula": "physical_margin - provisional_route_residual",
            "rule": "reject candidates that violate the frozen headroom rule or unsafe host/pinned limits",
            "projection_is_not_authority": True,
        },
        "timing": {
            "primary_metric": "B * 3600 / amortized_production_seconds_per_generation",
            "effective_time": "((K-1)*T_ordinary + T_checkpoint)/K",
            "minimum_steady_state_generations": 10,
            "process_policy": "clean process per candidate, fixed allocator/settings, interleave when practical",
            "uncertainty_target": "approximately <=2-3% or a predefined bounded maximum window",
            "mandatory_costs": [
                "policy settlement",
                "host-authority commits",
                "required reporting",
                "checkpoint projection/save",
                "mandatory synchronization",
                "normal result transfer",
            ],
        },
        "telemetry": {
            "gpu_memory": [
                "device_total",
                "minimum_free_at_peak",
                "process_gpu_memory",
                "torch_allocated_reserved_peak",
                "hapb_hot_prefix_allocation",
                "hapb_tile_allocation",
                "scratch_workspace",
                "no_full_authoritative_resident_cuda_bank_proof",
            ],
            "host": [
                "canonical_host_policy_bank_bytes",
                "pinned_current_high_water",
                "process_rss",
                "staging_transient_high_water",
                "duplicate_bank_check",
            ],
            "transfer": [
                "policy_h2d_bytes_per_generation",
                "policy_d2h_bytes_per_generation",
                "requested_rows",
                "transferred_rows",
                "unique_rows",
                "tile_fill_hit_miss",
                "total_policy_traffic",
            ],
            "timing": [
                "ordinary_generation",
                "checkpoint_generation",
                "selection_miss_preparation",
                "descriptor_d2h",
                "policy_h2d",
                "cuda_policy_compute",
                "settlement",
                "result_d2h",
                "host_commit",
                "synchronization_stall",
                "total_hapb_boundary",
                "mandatory_reporting",
                "checkpoint_projection_save",
                "amortized_generation",
            ],
            "unavailable_policy": "record unavailable explicitly; never fabricate telemetry",
        },
        "refinement": {
            "coarse_first": True,
            "tile_values_near_ridge": [256, 512, 1024, 2048],
            "finalist_repeats": 3,
            "rehearsal": "bounded longer run for the best one or two candidates",
            "stop_rules": [
                "memory/headroom failure",
                "semantic correctness failure",
                "throughput peak/decline",
                "upper uncertainty bound cannot beat winner",
                "PCIe/HAPB overhead recovery implausible",
                "unsafe host/pinned behavior",
                "non-equivalent treatment composition",
                "only answering largest-that-boots",
            ],
        },
        "promotion": {
            "threshold": ">=1.25x Resident-B48 production-equivalent throughput or indispensable documented capability",
            "smaller_throughput_gain": "insufficient by itself",
            "failure_classification": "HAPB-PERF-B/non-promotion unless evidence is incomplete (PERF-C) or invalid (PERF-D)",
        },
    }


def _gate7_readiness(*, factory_runtime_artifact_bound: bool) -> dict[str, Any]:
    return {
        "gate": 7,
        "decision": "G7-A_READY_FOR_GATE8",
        "q0_scope_and_authority": "READY",
        "q1_current_source_and_artifact_lineage": "READY",
        "q2_route_contract_and_telemetry": "READY",
        "q3_lifecycle_checkpoint_parity": "PROVEN_215_EQ_203",
        "q4_benchmark_method_and_admission_rule": "FROZEN_FOR_MEASUREMENT",
        "q5_ready_for_qualification": "NOT_ISSUED",
        "qualification_claim": False,
        "production_execution_admission": False,
        "factory_runtime_artifact_bound": factory_runtime_artifact_bound,
        "notes": [
            "Gate 6 remains CLOSED PASS on V1 pair 203/211. FRAG-A is absent from that closure.",
            "Factory 200 is the frozen G0 B48 source actually used by 203/211/215.",
            "If factory_runtime_artifact_bound is false, factory 200 remains bound to its historical native (189); the Gate-8 runtime artifact is the parameterized-B rebuild.",
            "High-B ABI is parameterized from bank/source extent. C512 kept. No B%8.",
            "This packet does not issue Q5 or READY_FOR_QUALIFICATION.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--interrupted-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--runtime-source-commit",
        type=str,
        help="frozen runtime commit when current HEAD contains only the allowlisted packet/tooling delta",
    )
    args = parser.parse_args()

    for path in (args.spec, args.prompt, args.artifact, args.manifest, args.source_root, args.interrupted_root):
        _require(path.expanduser().exists(), f"required input is missing: {path}")
    output_root = args.output_root.expanduser().resolve()
    _require(not output_root.exists(), f"refusing to overwrite immutable packet: {output_root}")
    repo_root = Path(__file__).resolve().parents[2]
    head = _git_head(repo_root)
    runtime_source_commit = args.runtime_source_commit or head
    non_runtime_delta = _git_non_runtime_delta(repo_root, runtime_source_commit, head)
    # Parse the spec through the repository compiler as a CPU-only structural
    # check.  Do not invoke any runtime or CUDA path here.
    spec = load_campaign_spec(args.spec.expanduser().resolve())
    parameters = spec.get("parameters")
    _require(isinstance(parameters, Mapping), "campaign spec parameters are missing")
    _require(parameters.get("patch.capacity") == 512, "spec patch capacity is not C512")
    _require(parameters.get("patch.initial_population") == 256, "spec hot-prefix source is not P256")
    artifact = _validate_artifact(
        artifact=args.artifact.expanduser().resolve(),
        manifest=args.manifest.expanduser().resolve(),
        expected_source_commit=runtime_source_commit,
    )
    source = _validate_source_factory(
        source_root=args.source_root.expanduser().resolve(),
        expected_source_commit=runtime_source_commit,
        expected_artifact_sha256=artifact["sha256"],
    )
    interruption = _validate_interruption(args.interrupted_root.expanduser().resolve())
    packet = {
        "schema_version": SCHEMA_VERSION,
        "authority": "HAPB merged end-to-end closure prompt",
        "packet_kind": "CPU-only Gate 7 readiness and Gate 8 benchmark preparation",
        "created_by_source_commit": head,
        "runtime_source_commit": runtime_source_commit,
        "allowlisted_non_runtime_delta": non_runtime_delta,
        "inputs": {
            "spec_path": str(args.spec.expanduser().resolve()),
            "spec_sha256": _sha256(args.spec.expanduser().resolve()),
            "prompt_path": str(args.prompt.expanduser().resolve()),
            "prompt_sha256": _sha256(args.prompt.expanduser().resolve()),
            "artifact": artifact,
            "source_factory": source,
            "preserved_interruption": interruption,
        },
        "gate7_readiness": _gate7_readiness(
            factory_runtime_artifact_bound=bool(source.get("runtime_artifact_bound")),
        ),
        "gate8_contract": _benchmark_contract(),
        "qualification_claim": False,
        "production_execution_admission": False,
        "gpu_execution_performed": False,
    }
    output_root.mkdir(parents=True, exist_ok=False)
    receipt = output_root / "hapb_gate7_gate8_cpu_packet_receipt.json"
    receipt.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    packet["receipt_sha256"] = canonical_sha256(packet)
    receipt.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = output_root / "README.md"
    summary.write_text(
        "# HAPB Gate 7/8 CPU-only preparation\n\n"
        "This packet freezes the benchmark contract and validates current-source lineage. "
        "It performs no CUDA work and issues no qualification or production-admission claim.\n\n"
        f"- Current helper HEAD: `{head}`\n"
        f"- Frozen runtime source commit: `{runtime_source_commit}`\n"
        f"- Artifact SHA256: `{artifact['sha256']}`\n"
        f"- Source factory rows: `{source['rows']}` distinct cells\n"
        f"- Preserved interrupted root: `{interruption['root']}`\n"
        "- Gate 6 lifecycle parity: proven 215==203; factory 200 retained\n"
        "- Gate 7 Q5/READY_FOR_QUALIFICATION: not issued\n"
        "- Gate 8 timing: pending exclusive GPU admission\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "receipt": str(receipt),
        "receipt_sha256": packet["receipt_sha256"],
        "source_commit": head,
        "gpu_execution_performed": False,
        "gate7_decision": packet["gate7_readiness"]["decision"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
