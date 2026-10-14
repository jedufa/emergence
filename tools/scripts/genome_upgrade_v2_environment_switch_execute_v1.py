#!/usr/bin/env python3
"""Execute exact production-artifact evidence for the Genome V2 switch gate."""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Final, Mapping, Sequence


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.authoritative_state.schema import content_sha256  # noqa: E402
from engine.evolution.genome_upgrade_v2 import (  # noqa: E402
    GenomeUpgradeV2RuntimeProjectionV1,
    PRODUCT_COUNT,
)
from engine.evolution.genome_upgrade_v2_c1_executor_v1 import (  # noqa: E402
    reference_dimensions_for_fixture_v1,
)
from engine.evolution.genome_upgrade_v2_causal_gate_v1 import (  # noqa: E402
    REQUIRED_DIMENSIONS,
    fixture_input_dimensions_v1,
)
from engine.evolution.genome_upgrade_v2_environment_switch_gate_v1 import (  # noqa: E402
    BACKENDS,
    CONDITIONS,
    EVIDENCE_SCHEMA_VERSION,
    PREREGISTRATION_SHA256,
    REPEAT_INDICES,
    SNAPSHOT_SCHEMA_VERSION,
    evaluate_environment_switch_v1,
)
from engine.evolution.genome_upgrade_v2_hereditary_capacity_gate_v1 import (  # noqa: E402
    AUTHORITY_FIELDS,
    validate_authority_v1 as validate_reducer_authority_v1,
)
from tools.scripts.genome_upgrade_v2_c1_artifact_gate_v1 import (  # noqa: E402
    _load_artifact,
    invoke_artifact_backend_v1,
    validate_authority_v1 as validate_c1_authority_v1,
)


SCHEMA_VERSION: Final[str] = "genome_v2_environment_switch_execute_v1"
FIXTURE_FAMILY: Final[str] = "regulatory_activation_inhibition"
RUNTIME_COST_FIELD: Final[str] = "expression_energy_debit_q10"
RUNTIME_PROJECTION_FIELDS: Final[tuple[str, ...]] = tuple(
    field.name for field in fields(GenomeUpgradeV2RuntimeProjectionV1)
)
TRAJECTORIES: Final[dict[str, tuple[str, str]]] = {
    "stable_a": ("A", "A"),
    "stable_b": ("B", "B"),
    "a_to_b": ("A", "B"),
    "b_to_a": ("B", "A"),
}
OUTPUT_FILES: Final[dict[str, str]] = {
    "authority": "genome_v2_environment_switch_authority.json",
    "evidence": "genome_v2_environment_switch_evidence.json",
    "decision": "genome_v2_environment_switch_decision.json",
    "manifest": "genome_v2_environment_switch_execution_manifest.json",
}
BackendInvoker = Callable[
    [Any, str, Sequence[Mapping[str, Any]]],
    list[dict[str, tuple[int, ...]]],
]


class GenomeV2EnvironmentSwitchExecuteError(RuntimeError):
    """The exact environment-switch artifact execution failed closed."""


def _fail(message: str) -> GenomeV2EnvironmentSwitchExecuteError:
    return GenomeV2EnvironmentSwitchExecuteError(message)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authority(binding: Mapping[str, str]) -> dict[str, str]:
    try:
        value = {name: binding[name] for name in AUTHORITY_FIELDS}
        return validate_reducer_authority_v1(value)
    except (KeyError, ValueError) as exc:
        raise _fail("C1 binding cannot form environment-switch authority") from exc


def fixed_switch_inputs_v1() -> tuple[
    tuple[int, ...], dict[str, tuple[int, ...]], list[dict[str, Any]]
]:
    """Materialize the fixed C1 genotype and eight trajectory endpoints."""

    genotype_a, context_a = fixture_input_dimensions_v1(
        FIXTURE_FAMILY, "control"
    )
    genotype_b, context_b = fixture_input_dimensions_v1(
        FIXTURE_FAMILY, "treatment"
    )
    if genotype_a != genotype_b or context_a == context_b:
        raise _fail("environment-switch C1 fixture identity drifted")
    contexts = {"A": context_a, "B": context_b}
    inputs = [
        {
            "condition": condition,
            "endpoint": endpoint,
            "environment": environment,
            "genotype": genotype_a,
            "context": contexts[environment],
        }
        for condition in CONDITIONS
        for endpoint, environment in zip(
            ("initial", "terminal"),
            TRAJECTORIES[condition],
            strict=True,
        )
    ]
    return genotype_a, contexts, inputs


def _reference_backend(
    inputs: Sequence[Mapping[str, Any]],
) -> list[dict[str, tuple[int, ...]]]:
    reference = {
        "A": reference_dimensions_for_fixture_v1(FIXTURE_FAMILY, "control"),
        "B": reference_dimensions_for_fixture_v1(FIXTURE_FAMILY, "treatment"),
    }
    rows = []
    for item in inputs:
        environment = str(item["environment"])
        row = reference[environment]
        if (
            tuple(item["genotype"]) != row["genotype"]
            or tuple(item["context"]) != row["context"]
        ):
            raise _fail("reference endpoint changed fixed switch inputs")
        rows.append(row)
    return rows


def invoke_switch_backend_v1(
    module: Any,
    backend: str,
    inputs: Sequence[Mapping[str, Any]],
) -> list[dict[str, tuple[int, ...]]]:
    """Invoke the existing C1 reference or immutable artifact endpoint."""

    if backend == "reference":
        return _reference_backend(inputs)
    return invoke_artifact_backend_v1(module, backend, inputs)


def _snapshot(
    dimensions: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    if set(dimensions) != set(REQUIRED_DIMENSIONS):
        raise _fail("switch endpoint dimensions differ from the C1 contract")
    runtime = tuple(int(value) for value in dimensions["runtime_consequence"])
    costs = tuple(int(value) for value in dimensions["cost"])
    if (
        len(costs) != 2
        or len(runtime) != len(RUNTIME_PROJECTION_FIELDS)
        or RUNTIME_PROJECTION_FIELDS[-1] != RUNTIME_COST_FIELD
    ):
        raise _fail("switch endpoint omitted exact cost dimensions")
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "context_q10": list(dimensions["context"]),
        "products_q10": list(dimensions["product"][:PRODUCT_COUNT]),
        "regulatory_q10": list(dimensions["regulatory"]),
        "phenotypes_q10": list(dimensions["phenotype"]),
        "runtime_consequences_q10": list(runtime),
        "expression_cost_q10": costs[0],
        "regulatory_cost_q10": costs[1],
        "runtime_cost_q10": runtime[-1],
    }


def execute_evidence_v1(
    *,
    module: Any,
    binding: Mapping[str, str],
    backend_invoker: BackendInvoker = invoke_switch_backend_v1,
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    """Execute two fresh exact batches per reference/native/CUDA backend."""

    authority = _authority(binding)
    genotype, _contexts, inputs = fixed_switch_inputs_v1()
    genotype_sha256 = content_sha256(list(genotype))
    evidence: list[dict[str, Any]] = []
    invocation_counts = {backend: 0 for backend in BACKENDS}
    for repeat in REPEAT_INDICES:
        results: dict[str, list[dict[str, tuple[int, ...]]]] = {}
        for backend in BACKENDS:
            rows = backend_invoker(module, backend, inputs)
            invocation_counts[backend] += 1
            if len(rows) != len(inputs):
                raise _fail(f"{backend} returned the wrong switch batch size")
            results[backend] = rows
        for backend in BACKENDS:
            endpoints: dict[str, dict[str, dict[str, Any]]] = {
                condition: {} for condition in CONDITIONS
            }
            for item, dimensions in zip(
                inputs, results[backend], strict=True
            ):
                if (
                    tuple(dimensions["genotype"]) != genotype
                    or tuple(dimensions["context"]) != tuple(item["context"])
                ):
                    raise _fail(f"{backend} changed genotype or context")
                endpoints[str(item["condition"])][str(item["endpoint"])] = (
                    _snapshot(dimensions)
                )
            for condition in CONDITIONS:
                value = {
                    "schema_version": EVIDENCE_SCHEMA_VERSION,
                    "preregistration_sha256": PREREGISTRATION_SHA256,
                    "authority": authority,
                    "condition": condition,
                    "backend": backend,
                    "repeat_index": repeat,
                    "genotype_sha256": genotype_sha256,
                    "initial": endpoints[condition]["initial"],
                    "terminal": endpoints[condition]["terminal"],
                    "no_fallback": True,
                    "no_jit": True,
                }
                evidence.append(
                    {**value, "evidence_sha256": content_sha256(value)}
                )
    if invocation_counts != {backend: 2 for backend in BACKENDS}:
        raise _fail("each switch backend must be invoked exactly twice")
    decision = evaluate_environment_switch_v1(
        evidence, authority=authority
    )
    if decision.get("decision") != "ENVIRONMENT-SWITCH-A":
        raise _fail(
            "environment-switch reducer rejected artifact evidence: "
            f"{decision.get('decision')} {decision.get('reasons')}"
        )
    return authority, evidence, decision


def execute(
    *,
    artifact: Path,
    manifest_path: Path,
    binding_path: Path,
    package_receipt_path: Path,
    route_receipt_path: Path,
    output_directory: Path,
    backend_invoker: BackendInvoker = invoke_switch_backend_v1,
) -> dict[str, Any]:
    """Validate authority, execute evidence, and publish immutable JSON files."""

    output = output_directory.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    binding, package, manifest, route = validate_c1_authority_v1(
        artifact=artifact,
        manifest_path=manifest_path,
        binding_path=binding_path,
        package_receipt_path=package_receipt_path,
        route_receipt_path=route_receipt_path,
    )
    module = _load_artifact(artifact.resolve(), manifest)
    authority, evidence, decision = execute_evidence_v1(
        module=module,
        binding=binding,
        backend_invoker=backend_invoker,
    )
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / OUTPUT_FILES["authority"], authority)
    _write_json(output / OUTPUT_FILES["evidence"], evidence)
    _write_json(output / OUTPUT_FILES["decision"], decision)
    manifest_value = {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "row_count": len(evidence),
        "backend_invocation_counts": {backend: 2 for backend in BACKENDS},
        "authority": authority,
        "package_authority_state": package["authority_state"],
        "route_authority_state": route["authority_state"],
        "artifact_identity": {
            "module_name": manifest["module_name"],
            "artifact_sha256": binding["artifact_sha256"],
            "manifest_sha256": binding["manifest_sha256"],
            "runtime_jit_compilation": False,
            "fallback": False,
        },
        "decision": decision["decision"],
        "artifacts": {
            name: {
                "path": filename,
                "sha256": _sha256(output / filename),
            }
            for name, filename in OUTPUT_FILES.items()
            if name != "manifest"
        },
    }
    manifest_value["manifest_sha256"] = content_sha256(manifest_value)
    _write_json(output / OUTPUT_FILES["manifest"], manifest_value)
    return manifest_value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--authority-binding", type=Path, required=True)
    parser.add_argument("--package-receipt", type=Path, required=True)
    parser.add_argument("--route-receipt", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = execute(
            artifact=args.artifact,
            manifest_path=args.manifest,
            binding_path=args.authority_binding,
            package_receipt_path=args.package_receipt,
            route_receipt_path=args.route_receipt,
            output_directory=args.output_directory,
        )
    except (OSError, ValueError, GenomeV2EnvironmentSwitchExecuteError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
