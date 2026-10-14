#!/usr/bin/env python3
"""Replay the governed Genome V2 source and compiler/device parity suites."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Final
import xml.etree.ElementTree as ET


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
SCHEMA_VERSION: Final[str] = "genome_upgrade_v2_parity_gate_v1"
SOURCE_LANE: Final[str] = "source_unit"
ENV_LANE: Final[str] = "compiler_device"
EXPECTED_COUNTS: Final[dict[str, int]] = {
    SOURCE_LANE: 571,
    ENV_LANE: 15,
}

SOURCE_SELECTORS: Final[tuple[str, ...]] = (
    "tests/evolvability/capacity/test_genome_upgrade_v2.py",
    "tests/evolvability/capacity/test_diploid_inheritance_v1.py",
    (
        "tests/perf/test_cuda_genome_upgrade_v2_state_bank.py"
        "::test_fixed_state_bank_expression_and_checkpoint[cpu]"
    ),
    (
        "tests/perf/test_cuda_genome_upgrade_v2_state_bank.py"
        "::test_checkpoint_rejects_stale_capability_contract"
    ),
    (
        "tests/perf/test_cuda_genome_upgrade_v2_state_bank.py"
        "::test_bank_storage_is_bounded_and_linear"
    ),
    (
        "tests/perf/test_cuda_genome_upgrade_v2_state_bank.py"
        "::test_neutral_bank_uses_homozygous_neutral_quantity_and_stability"
    ),
    (
        "tests/perf/test_cuda_genome_upgrade_v2_state_bank.py"
        "::test_frozen_founder_profiles_initialize_live_slots_only"
    ),
    (
        "tests/perf/test_cuda_genome_upgrade_v2_state_bank.py"
        "::test_checkpoint_founder_profile_authority_is_exact"
    ),
    (
        "tests/perf/test_cuda_genome_upgrade_v2_state_bank.py"
        "::test_b2_founder_initialization_is_per_cell_and_ordered[cpu]"
    ),
    (
        "tests/perf/test_cuda_genome_upgrade_v2_state_bank.py"
        "::test_compact_summary_is_bounded_and_alive_only"
    ),
    "tests/test_genome_upgrade_v2_bd_capabilities_v1.py",
    "tests/test_genome_upgrade_v2_capability_contract_v1.py",
    "tests/test_genome_upgrade_v2_regulatory_network_v1.py",
    "tests/test_genome_upgrade_v2_dominance_fixtures_v1.py",
    "tests/test_genome_upgrade_v2_c1_executor_v1.py",
    "tests/test_genome_upgrade_v2_causal_gate_v1.py",
    "tests/test_genome_upgrade_v2_causal_transaction_projection_v1.py",
    "tests/test_genome_upgrade_v2_causal_cuda_output_v1.py",
    "tests/test_genome_upgrade_v2_causal_settlement_v1.py",
    "tests/test_genome_upgrade_v2_causal_cuda_gate_v1.py",
    "tests/test_genome_upgrade_v2_e1_gate_v1.py",
    "tests/test_genome_upgrade_v2_e1_collect_v1.py",
    "tests/test_genome_upgrade_v2_e1_materialize_specs_v1.py",
    "tests/test_genome_upgrade_v2_telemetry_projection_v1.py",
    "tests/test_genome_upgrade_v2_package_contract_v1.py",
    "tests/test_genome_upgrade_v2_promotion_contracts_v1.py",
    "tests/test_genome_upgrade_v2_state_inventory.py",
    "tests/test_genome_upgrade_v2_observability_gate_v1.py",
    "tests/test_genome_upgrade_v2_cancellation_lifecycle_gate_v1.py",
    "tests/test_genome_upgrade_v2_ecological_readiness_v1.py",
    "tests/test_genome_upgrade_v2_environment_switch_execute_v1.py",
    "tests/test_genome_upgrade_v2_environment_switch_gate_v1.py",
    "tests/test_genome_upgrade_v2_hereditary_capacity_collect_v1.py",
    "tests/test_genome_upgrade_v2_hereditary_capacity_gate_v1.py",
    "tests/test_genome_upgrade_v2_hereditary_capacity_materialize_spec_v1.py",
    "tests/test_genome_upgrade_v2_hereditary_capacity_source_v1.py",
    "tests/test_genome_upgrade_v2_p10_completion_collect_v1.py",
    "tests/test_genome_upgrade_v2_p10_completion_gate_v1.py",
    "tests/test_genome_upgrade_v2_p10_authority_issue_v1.py",
    (
        "tests/test_eiel_campaign_compiler_v1.py"
        "::test_generation_zero_checkpoint_is_compiled_as_a_campaign_value"
    ),
    "tests/test_authoritative_state_outer_output_patch_contract_v1.py",
    "tests/test_eiel_authoritative_state_abi_v1.py",
    "tests/test_eiel_authoritative_state_schema_runtime_v1.py",
    "tests/test_eiel_cuda_architecture_b_flattened_controller_v1_build.py",
    "tests/test_eiel_campaign_patch_runner_factory_v1.py",
    "tests/test_eiel_production_engine_native_bindings_v1.py",
    "tests/test_eiel_cuda_architecture_b_direct_resident_patch_v1.py",
    "tests/test_eiel_cuda_architecture_b_direct_resident_settlement_v1.py",
    "tests/test_eiel_cuda_architecture_b_direct_resident_source_allocator_v1.py",
    "tests/test_eiel_cuda_architecture_b_direct_resident_source_builder_v1.py",
    "tests/test_eiel_cuda_architecture_b_direct_resident_wdl_v2.py",
    (
        "tests/perf/test_genome_upgrade_v2_c1_artifact_endpoint_v1.py"
        "::test_production_binding_freezes_strict_native_and_cuda_endpoints"
    ),
    (
        "tests/perf/test_genome_upgrade_v2_c1_artifact_endpoint_v1.py"
        "::test_c1_diagnostics_do_not_expand_production_expression_layout"
    ),
    (
        "tests/perf/test_genome_upgrade_v2_c1_artifact_endpoint_v1.py"
        "::test_shared_cuda_launcher_is_emitted_for_base_and_unified_wdl_artifacts"
    ),
)
SOURCE_EXTRA_ARGS: Final[tuple[str, ...]] = (
    "-k",
    "not cuda_matches_cpu_tensor_contract_bit_exact",
)

ENV_SELECTORS: Final[tuple[str, ...]] = (
    (
        "tests/evolvability/capacity/test_genome_upgrade_v2.py"
        "::test_cuda_matches_cpu_tensor_contract_bit_exact"
    ),
    (
        "tests/perf/test_cuda_genome_upgrade_v2_state_bank.py"
        "::test_fixed_state_bank_expression_and_checkpoint[cuda]"
    ),
    (
        "tests/perf/test_cuda_genome_upgrade_v2_state_bank.py"
        "::test_b2_founder_initialization_is_per_cell_and_ordered[cuda]"
    ),
    (
        "tests/perf/test_cuda_diploid_state_bank_v1.py"
        "::test_cuda_state_bank_resides_on_device_and_round_trips"
    ),
    (
        "tests/perf/test_cuda_diploid_scalar_genome_v1.py"
        "::test_cpu_contract_matches_python_authority"
    ),
    (
        "tests/perf/test_cuda_diploid_scalar_genome_v1.py"
        "::test_sm75_cuda_primitive_executes"
    ),
    (
        "tests/perf/test_genome_upgrade_v2_c1_artifact_endpoint_v1.py"
        "::test_host_header_c1_batch_matches_all_frozen_reference_arms"
    ),
    "tests/perf/test_genome_upgrade_v2_native_parity.py",
)

LANE_SELECTORS: Final[dict[str, tuple[str, ...]]] = {
    SOURCE_LANE: SOURCE_SELECTORS,
    ENV_LANE: ENV_SELECTORS,
}
LANE_EXTRA_ARGS: Final[dict[str, tuple[str, ...]]] = {
    SOURCE_LANE: SOURCE_EXTRA_ARGS,
    ENV_LANE: (),
}
LANE_CLASSIFICATION: Final[dict[str, dict[str, Any]]] = {
    SOURCE_LANE: {
        "kind": "pure_source_and_cpu_unit_contracts",
        "requires_final_artifact": False,
        "requires_compiler": False,
        "requires_cuda_device": False,
    },
    ENV_LANE: {
        "kind": "independent_temporary_compiler_and_device_probes",
        "requires_final_artifact": False,
        "gxx_case_count": 8,
        "nvcc_and_cuda_case_count": 7,
    },
}


class GenomeV2ParityGateError(RuntimeError):
    """The governed replay could not be prepared or interpreted exactly."""


def _fail(message: str) -> GenomeV2ParityGateError:
    return GenomeV2ParityGateError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )


def _git(*args: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(ROOT), *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _clean_source_context() -> tuple[Path, str]:
    git_root = Path(_git("rev-parse", "--show-toplevel")).resolve()
    source_commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=all")
    if status:
        raise _fail("parity replay requires an exactly clean source worktree")
    if (
        len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise _fail("source commit is not a full lowercase Git object name")
    return git_root, source_commit


def _selector_path(selector: str) -> Path:
    return ROOT / selector.split("::", 1)[0]


def _validate_selectors(git_root: Path) -> None:
    selectors = tuple(
        selector
        for lane in (SOURCE_LANE, ENV_LANE)
        for selector in LANE_SELECTORS[lane]
    )
    if len(selectors) != len(set(selectors)):
        raise _fail("parity selector inventory contains duplicates")
    for selector in selectors:
        path = _selector_path(selector).resolve()
        if not path.is_file() or not path.is_relative_to(ROOT):
            raise _fail(f"parity selector path is missing or outside project: {selector}")
        relative = path.relative_to(git_root)
        completed = subprocess.run(
            (
                "git",
                "-C",
                str(git_root),
                "ls-files",
                "--error-unmatch",
                str(relative),
            ),
            check=False,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise _fail(f"parity selector is not tracked: {selector}")


def _collection_count(stdout: bytes) -> int:
    text = stdout.decode("utf-8", errors="replace")
    per_file = [
        int(match.group(1))
        for line in text.splitlines()
        if (
            match := re.fullmatch(
                r".+\.py:\s+([0-9]+)",
                line.strip(),
            )
        )
    ]
    if per_file:
        return sum(per_file)
    summary = re.findall(r"([0-9]+)\s+tests?\s+collected", text)
    if len(summary) == 1:
        return int(summary[0])
    node_ids = [
        line for line in text.splitlines() if "::test_" in line and not line.startswith(" ")
    ]
    if node_ids:
        return len(node_ids)
    raise _fail("pytest collection count is absent or ambiguous")


def _junit_counts(path: Path) -> dict[str, int]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise _fail("pytest execution did not emit valid structured JUnit counts") from exc
    if root.tag == "testsuites":
        suites = root.findall("testsuite")
        if not suites:
            raise _fail("pytest JUnit report contains no test suites")
        values = {
            name: sum(int(suite.attrib.get(name, "0")) for suite in suites)
            for name in ("tests", "failures", "errors", "skipped")
        }
    elif root.tag == "testsuite":
        values = {
            name: int(root.attrib.get(name, "0"))
            for name in ("tests", "failures", "errors", "skipped")
        }
    else:
        raise _fail("pytest JUnit root has an unsupported schema")
    values["passed"] = (
        values["tests"]
        - values["failures"]
        - values["errors"]
        - values["skipped"]
    )
    if values["passed"] < 0:
        raise _fail("pytest JUnit counts are internally contradictory")
    return values


def _run_lane(
    lane: str,
    *,
    output_directory: Path,
    collect_only: bool,
    python: Path,
    pythonpath: str,
) -> dict[str, Any]:
    expected = EXPECTED_COUNTS[lane]
    stdout_path = output_directory / f"{lane}.stdout.log"
    stderr_path = output_directory / f"{lane}.stderr.log"
    junit_path = output_directory / f".{lane}.junit.xml"
    argv = [str(python), "-m", "pytest", "-q"]
    if collect_only:
        argv.append("--collect-only")
    else:
        argv.append(f"--junitxml={junit_path}")
    argv.extend(LANE_SELECTORS[lane])
    argv.extend(LANE_EXTRA_ARGS[lane])
    environment = dict(os.environ)
    environment["PYTHONPATH"] = pythonpath
    completed = subprocess.run(
        argv,
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
    )
    stdout_path.write_bytes(completed.stdout)
    stderr_path.write_bytes(completed.stderr)
    counts: dict[str, int | None]
    count_error: str | None = None
    try:
        if collect_only:
            collected = _collection_count(completed.stdout)
            counts = {
                "collected": collected,
                "tests": None,
                "passed": None,
                "failures": None,
                "errors": None,
                "skipped": None,
            }
        else:
            execution = _junit_counts(junit_path)
            counts = {"collected": None, **execution}
    except GenomeV2ParityGateError as exc:
        count_error = str(exc)
        counts = {
            "collected": None,
            "tests": None,
            "passed": None,
            "failures": None,
            "errors": None,
            "skipped": None,
        }
    finally:
        junit_path.unlink(missing_ok=True)

    actual = counts["collected"] if collect_only else counts["tests"]
    exact_count = actual == expected
    all_passed = (
        True
        if collect_only
        else (
            counts["passed"] == expected
            and counts["failures"] == 0
            and counts["errors"] == 0
            and counts["skipped"] == 0
        )
    )
    return {
        "lane": lane,
        "mode": "collect_only" if collect_only else "execute",
        "ok": (
            completed.returncode == 0
            and count_error is None
            and exact_count
            and all_passed
        ),
        "returncode": completed.returncode,
        "expected_count": expected,
        "counts": counts,
        "count_error": count_error,
        "exact_argv": argv,
        "selectors": list(LANE_SELECTORS[lane]),
        "extra_pytest_args": list(LANE_EXTRA_ARGS[lane]),
        "classification": LANE_CLASSIFICATION[lane],
        "stdout_log": stdout_path.name,
        "stdout_sha256": _sha256_bytes(completed.stdout),
        "stdout_bytes": len(completed.stdout),
        "stderr_log": stderr_path.name,
        "stderr_sha256": _sha256_bytes(completed.stderr),
        "stderr_bytes": len(completed.stderr),
    }


def execute_gate(
    *,
    output_directory: Path,
    collect_only: bool = False,
) -> dict[str, Any]:
    git_root, source_commit = _clean_source_context()
    output_directory = output_directory.expanduser().resolve()
    if output_directory.exists():
        raise FileExistsError(output_directory)
    if output_directory.is_relative_to(git_root):
        raise _fail("parity output directory must be outside the source repository")
    _validate_selectors(git_root)
    output_directory.mkdir(parents=True, exist_ok=False)
    python = Path(sys.executable).absolute()
    pythonpath = f"{ROOT / 'engine'}:{ROOT}"
    lanes = {
        lane: _run_lane(
            lane,
            output_directory=output_directory,
            collect_only=collect_only,
            python=python,
            pythonpath=pythonpath,
        )
        for lane in (SOURCE_LANE, ENV_LANE)
    }
    post_root, post_commit = _clean_source_context()
    source_stable = post_root == git_root and post_commit == source_commit
    payload = {
        "schema_version": SCHEMA_VERSION,
        "ok": source_stable and all(row["ok"] for row in lanes.values()),
        "mode": "collect_only" if collect_only else "execute",
        "source": {
            "git_root": str(git_root),
            "project_root": str(ROOT),
            "source_commit": source_commit,
            "clean_before_and_after": source_stable,
        },
        "runtime": {
            "python": str(python),
            "pythonpath": pythonpath,
            "cwd": str(ROOT),
        },
        "expected_total_count": sum(EXPECTED_COUNTS.values()),
        "observed_total_count": sum(
            int(
                row["counts"]["collected"]
                if collect_only
                else row["counts"]["tests"]
            )
            for row in lanes.values()
            if (
                row["counts"]["collected"]
                if collect_only
                else row["counts"]["tests"]
            )
            is not None
        ),
        "lanes": lanes,
        "final_artifact_proof": {
            "status": "not_provided_by_this_gate",
            "reason": (
                "compiler/device tests build independent temporary probes and "
                "do not load the promoted production CUDA artifact"
            ),
            "required_followup": (
                "run genome_upgrade_v2_causal_cuda_gate_v1.py with the final "
                "artifact, adjacent manifest, and exact post-route authority binding"
            ),
        },
        "claims_boundary": (
            "source-contract replay and independent compiler/device parity only; "
            "no final-artifact, route-qualification, ecological, intelligence, "
            "emergence, consciousness, or broad cognition claim"
        ),
    }
    receipt = {
        **payload,
        "receipt_sha256": hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest(),
    }
    _write_json(output_directory / "genome_v2_parity_gate_receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="collect and count both governed lanes without executing tests",
    )
    args = parser.parse_args()
    receipt = execute_gate(
        output_directory=args.output_directory,
        collect_only=args.collect_only,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
