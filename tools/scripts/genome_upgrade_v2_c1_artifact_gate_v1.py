#!/usr/bin/env python3
"""Run the frozen Genome V2 C1 matrix through the production artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Callable, Final, Mapping, Sequence

import torch


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.authoritative_state.schema import content_sha256  # noqa: E402
from engine.evolution.genome_upgrade_v2_c1_executor_v1 import (  # noqa: E402
    BackendCaseOutputV1,
    execute_c1_v1,
    reference_dimensions_for_fixture_v1,
)
from engine.evolution.genome_upgrade_v2_causal_gate_v1 import (  # noqa: E402
    DIMENSION_LENGTHS,
    MANDATORY_FAMILIES,
    PREREGISTRATION_SHA256,
    REQUIRED_BACKENDS,
    REQUIRED_DIMENSIONS,
    REQUIRED_REPEAT_INDICES,
    AuthorityBindingV1,
    CheckpointReceiptV1,
    fixture_by_family_v1,
    fixture_input_dimensions_v1,
    preregistration_payload_v1,
)
from engine.evolution.genome_upgrade_v2_package_v1 import (  # noqa: E402
    package_identity_sha256_v1,
    validate_checkpoint_authority_binding_v1,
)
from engine.qualification_plans.v1 import (  # noqa: E402
    EXECUTION_ROUTE_OBLIGATIONS,
)


SCHEMA_VERSION: Final[str] = "genome_v2_p10_c1_artifact_gate_v2"
NATIVE_ENTRYPOINT: Final[str] = (
    "eiel_cuda_architecture_b_genome_upgrade_v2_c1_native_batch_v1"
)
CUDA_ENTRYPOINT: Final[str] = (
    "eiel_cuda_architecture_b_genome_upgrade_v2_c1_cuda_batch_v1"
)
ARTIFACT_OUTPUT_SCHEMA: Final[str] = "genome_v2_p10_c1_artifact_batch_v1"
CHECKPOINT_SCHEMA: Final[str] = (
    "genome_v2_p10_c1_fixture_matrix_checkpoint_v1"
)
EXECUTION_RECEIPT_SCHEMA: Final[str] = (
    "genome_v2_p10_c1_artifact_execution_receipt_v1"
)
CHECKPOINT_EXECUTION_SCHEMA: Final[str] = (
    "genome_v2_p10_c1_checkpoint_execution_receipt_v2"
)
CHECKPOINT_RECEIPT_SET_SCHEMA: Final[str] = (
    "genome_v2_p10_c1_checkpoint_receipts_v1"
)
CHECKPOINT_RESTORE_EXECUTION_SCHEMA: Final[str] = (
    "genome_v2_p10_c1_checkpoint_restore_execution_v1"
)
OUTPUT_NAMES: Final[dict[str, str]] = {
    "preregistration": "genome_v2_c1_preregistration.json",
    "checkpoint": "genome_v2_c1_fixture_checkpoint.json",
    "checkpoint_receipts": "genome_v2_c1_checkpoint_receipts.json",
    "matrix_json": "genome_v2_c1_execution_matrix.json",
    "matrix_csv": "genome_v2_c1_execution_matrix.csv",
    "decision": "genome_v2_c1_decision.json",
    "manifest": "genome_v2_c1_artifact_gate_manifest.json",
}
BackendInvoker = Callable[
    [Any, str, Sequence[Mapping[str, Any]]],
    list[dict[str, tuple[int, ...]]],
]


class GenomeV2C1ArtifactGateError(RuntimeError):
    """The production-artifact C1 gate failed closed."""


def _fail(message: str) -> GenomeV2C1ArtifactGateError:
    return GenomeV2C1ArtifactGateError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _fail(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise _fail(f"{label} must be a JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )


def validate_authority_v1(
    *,
    artifact: Path,
    manifest_path: Path,
    binding_path: Path,
    package_receipt_path: Path,
    route_receipt_path: Path,
) -> tuple[
    dict[str, str],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Validate the artifact, package receipt, and 17-field route binding."""

    for path, label in (
        (artifact, "artifact"),
        (manifest_path, "artifact manifest"),
        (binding_path, "authority binding"),
        (package_receipt_path, "package receipt"),
        (route_receipt_path, "route receipt"),
    ):
        if path.is_symlink() or not path.is_file():
            raise _fail(f"{label} must be an existing non-symlink file")
    if manifest_path.resolve() != Path(f"{artifact.resolve()}.manifest.json"):
        raise _fail("artifact manifest must be adjacent to the production .so")
    manifest = _read_json(manifest_path, label="artifact manifest")
    package_receipt = _read_json(
        package_receipt_path,
        label="package receipt",
    )
    route_receipt = _read_json(
        route_receipt_path,
        label="route receipt",
    )
    try:
        binding = validate_checkpoint_authority_binding_v1(
            _read_json(binding_path, label="authority binding")
        )
    except ValueError as exc:
        raise _fail("authority binding is not the exact 17-field contract") from exc
    if (
        package_receipt.get("ok") is not True
        or package_receipt.get("authority_state")
        != "package_candidate_complete"
        or package_identity_sha256_v1(package_receipt)
        != package_receipt.get("package_id_sha256")
        or binding["package_id_sha256"]
        != package_receipt.get("package_id_sha256")
        or binding["package_receipt_sha256"]
        != _sha256(package_receipt_path)
        or binding["artifact_sha256"] != _sha256(artifact)
        or binding["manifest_sha256"] != _sha256(manifest_path)
    ):
        raise _fail("package, artifact, and route authority do not join exactly")
    identity = package_receipt.get("identity")
    if not isinstance(identity, Mapping):
        raise _fail("package receipt has no identity object")
    identity_join = {
        "provider_id": "provider_id",
        "runtime_profile_id": "runtime_profile_id",
        "global_genome_profile_id": "global_genome_profile_id",
        "source_commit": "source_commit",
        "executable_package_core_sha256": (
            "executable_package_core_sha256"
        ),
        "artifact_sha256": "cuda_artifact_sha256",
        "manifest_sha256": "cuda_manifest_sha256",
        "state_schema_sha256": "authoritative_state_schema_sha256",
        "mechanic_contract_sha256": "mechanic_contract_sha256",
        "capability_contract_sha256": "capability_contract_sha256",
        "cancellation_abi_sha256": "cancellation_abi_sha256",
        "observability_schema_sha256": (
            "observability_storage_budget_sha256"
        ),
        "capacity_layout_contract_sha256": (
            "capacity_layout_contract_sha256"
        ),
    }
    if any(
        binding[binding_name] != identity.get(identity_name)
        for binding_name, identity_name in identity_join.items()
    ):
        raise _fail("package identity contradicts route authority")
    if (
        manifest.get("artifact_sha256") != binding["artifact_sha256"]
        or manifest.get("runtime_jit_compilation") is not False
        or manifest.get("immutable_output") is not True
        or manifest.get("source_snapshot_stable") is not True
    ):
        raise _fail("artifact manifest does not prove immutable no-JIT output")
    covered = route_receipt.get("covered_obligations")
    missing = route_receipt.get("missing_obligations", [])
    if (
        route_receipt.get("authority_state") != "route_qualified"
        or route_receipt.get("execution_route_sha256")
        != binding["execution_route_sha256"]
        or route_receipt.get("artifact_sha256")
        != binding["artifact_sha256"]
        or route_receipt.get("provider_id") != binding["provider_id"]
        or route_receipt.get("profile_id")
        != binding["runtime_profile_id"]
        or not isinstance(covered, list)
        or set(covered) != EXECUTION_ROUTE_OBLIGATIONS
        or len(covered) != len(EXECUTION_ROUTE_OBLIGATIONS)
        or any(not isinstance(value, str) or not value for value in covered)
        or not isinstance(missing, list)
        or missing
    ):
        raise _fail("route receipt does not prove the exact qualified route")
    return binding, package_receipt, manifest, route_receipt


def _load_artifact(path: Path, manifest: Mapping[str, Any]) -> ModuleType:
    module_name = manifest.get("module_name")
    if not isinstance(module_name, str) or not module_name:
        raise _fail("artifact manifest has no module name")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise _fail("cannot construct the production artifact loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for endpoint in (NATIVE_ENTRYPOINT, CUDA_ENTRYPOINT):
        if not callable(getattr(module, endpoint, None)):
            raise _fail(f"production artifact omitted {endpoint}")
    provenance = getattr(
        module,
        "eiel_authoritative_state_build_provenance_v1",
        None,
    )
    if not callable(provenance):
        raise _fail("production artifact omitted build provenance")
    return module


def _fixture_checkpoint_v1(
    binding: Mapping[str, str],
) -> dict[str, Any]:
    rows = []
    for family in MANDATORY_FAMILIES:
        fixture = fixture_by_family_v1(family)
        arms = {}
        for arm in ("control", "treatment"):
            genotype, context = fixture_input_dimensions_v1(family, arm)
            reference_dimensions_for_fixture_v1(family, arm)
            arms[arm] = {
                "genotype": list(genotype),
                "context": list(context),
            }
        rows.append(
            {
                "family_id": family,
                "pair_id": fixture["pair_id"],
                "seed": fixture["seed"],
                "fixture_sha256": fixture["fixture_sha256"],
                "arms": arms,
            }
        )
    value = {
        "schema_version": CHECKPOINT_SCHEMA,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "authority_binding_sha256": content_sha256(dict(binding)),
        "rows": rows,
        "restore_semantics": (
            "immutable fixed-fixture matrix rematerialized independently "
            "for each fresh backend repeat; not a live-world checkpoint"
        ),
    }
    return {**value, "checkpoint_sha256": content_sha256(value)}


def _checkpoint_rows(
    checkpoint: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for fixture in checkpoint["rows"]:
        for arm in ("control", "treatment"):
            rows.append(
                {
                    "family_id": fixture["family_id"],
                    "arm": arm,
                    "genotype": tuple(fixture["arms"][arm]["genotype"]),
                    "context": tuple(fixture["arms"][arm]["context"]),
                }
            )
    return rows


def _validated_artifact_output(
    raw: Any,
    *,
    backend: str,
    inputs: Sequence[Mapping[str, Any]],
) -> list[dict[str, tuple[int, ...]]]:
    if not isinstance(raw, Mapping):
        raise _fail(f"{backend} endpoint returned a non-mapping")
    exact_keys = {
        "schema_version",
        "backend",
        *REQUIRED_DIMENSIONS,
        "no_jit",
        "no_fallback",
        "production_artifact_endpoint",
    }
    if set(raw) != exact_keys:
        raise _fail(f"{backend} endpoint output contract drifted")
    if (
        raw["schema_version"] != ARTIFACT_OUTPUT_SCHEMA
        or raw["backend"] != backend
        or raw["no_jit"] is not True
        or raw["no_fallback"] is not True
        or raw["production_artifact_endpoint"] is not True
    ):
        raise _fail(f"{backend} endpoint metadata is not admissible")
    device_type = "cpu" if backend == "native" else "cuda"
    batch = len(inputs)
    dimensions: dict[str, list[list[int]]] = {}
    for name in REQUIRED_DIMENSIONS:
        tensor = raw[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.int64
            or tensor.device.type != device_type
            or not tensor.is_contiguous()
            or tensor.requires_grad
            or tuple(tensor.shape)
            != (batch, DIMENSION_LENGTHS[name])
        ):
            raise _fail(f"{backend} returned an invalid {name} tensor")
        dimensions[name] = tensor.detach().cpu().tolist()
    rows = [
        {
            name: tuple(int(value) for value in dimensions[name][index])
            for name in REQUIRED_DIMENSIONS
        }
        for index in range(batch)
    ]
    for index, row in enumerate(rows):
        if (
            row["genotype"] != tuple(inputs[index]["genotype"])
            or row["context"] != tuple(inputs[index]["context"])
        ):
            raise _fail(f"{backend} endpoint changed frozen inputs")
    return rows


def invoke_artifact_backend_v1(
    module: Any,
    backend: str,
    inputs: Sequence[Mapping[str, Any]],
) -> list[dict[str, tuple[int, ...]]]:
    """Invoke one exact production endpoint without JIT or fallback."""

    genotype = torch.tensor(
        [row["genotype"] for row in inputs],
        dtype=torch.int64,
    )
    context = torch.tensor(
        [row["context"] for row in inputs],
        dtype=torch.int64,
    )
    if backend == "native":
        endpoint = getattr(module, NATIVE_ENTRYPOINT)
    elif backend == "cuda":
        if not torch.cuda.is_available():
            raise _fail("CUDA is unavailable; C1 does not fall back to CPU")
        genotype = genotype.cuda()
        context = context.cuda()
        endpoint = getattr(module, CUDA_ENTRYPOINT)
    else:
        raise _fail(f"unsupported artifact backend: {backend}")
    return _validated_artifact_output(
        endpoint(genotype, context),
        backend=backend,
        inputs=inputs,
    )


def _authority(
    binding: Mapping[str, str],
    *,
    checkpoint_sha256: str,
) -> AuthorityBindingV1:
    if len(binding["source_commit"]) != 40:
        raise _fail("C1 requires a full 40-character Git source commit")
    return AuthorityBindingV1(
        source_commit=binding["source_commit"],
        package_id_sha256=binding["package_id_sha256"],
        package_receipt_sha256=binding["package_receipt_sha256"],
        artifact_sha256=binding["artifact_sha256"],
        execution_route_sha256=binding["execution_route_sha256"],
        checkpoint_sha256=checkpoint_sha256,
    )


def _execution_receipt_sha256(
    *,
    backend: str,
    repeat: int,
    row: Mapping[str, Any],
    dimensions: Mapping[str, Sequence[int]],
    authority: AuthorityBindingV1,
) -> str:
    return content_sha256(
        {
            "schema_version": EXECUTION_RECEIPT_SCHEMA,
            "backend": backend,
            "repeat_index": repeat,
            "family_id": row["family_id"],
            "arm": row["arm"],
            "authority": authority.payload(),
            "dimensions": {
                name: list(dimensions[name]) for name in REQUIRED_DIMENSIONS
            },
            "no_jit": True,
            "no_fallback": True,
        }
    )


def _checkpoint_restore_evidence_v1(
    checkpoint: Mapping[str, Any],
    outputs: Sequence[BackendCaseOutputV1],
) -> dict[str, list[dict[str, Any]]]:
    fixture_rows = {
        row.get("family_id"): row
        for row in checkpoint.get("rows", ())
        if isinstance(row, Mapping)
    }
    if set(fixture_rows) != set(MANDATORY_FAMILIES):
        raise _fail("C1 fixture checkpoint does not cover seven families")
    indexed = {
        (
            output.family_id,
            output.repeat_index,
            output.arm,
            output.backend,
        ): output
        for output in outputs
    }
    expected = {
        (family, repeat, arm, backend)
        for family in MANDATORY_FAMILIES
        for repeat in REQUIRED_REPEAT_INDICES
        for arm in ("control", "treatment")
        for backend in REQUIRED_BACKENDS
    }
    if len(indexed) != len(outputs) or set(indexed) != expected:
        raise _fail("C1 restore evidence does not join the 84-receipt matrix")
    evidence: dict[str, list[dict[str, Any]]] = {}
    for family in MANDATORY_FAMILIES:
        fixture = fixture_by_family_v1(family)
        family_restores = []
        for repeat in REQUIRED_REPEAT_INDICES:
            execution_receipts = [
                {
                    "arm": arm,
                    "backend": backend,
                    "execution_receipt_sha256": indexed[
                        (family, repeat, arm, backend)
                    ].execution_receipt_sha256,
                }
                for arm in ("control", "treatment")
                for backend in REQUIRED_BACKENDS
            ]
            value = {
                "schema_version": CHECKPOINT_RESTORE_EXECUTION_SCHEMA,
                "family_id": family,
                "pair_id": fixture["pair_id"],
                "repeat_index": repeat,
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "fixture_row": dict(fixture_rows[family]),
                "fixture_row_sha256": content_sha256(
                    fixture_rows[family]
                ),
                "backend_execution_receipts": execution_receipts,
                "exact_fixture_rematerialization": True,
            }
            family_restores.append(
                {
                    **value,
                    "restore_payload_sha256": content_sha256(value),
                }
            )
        evidence[family] = family_restores
    return evidence


def execute_matrix_v1(
    *,
    module: Any,
    binding: Mapping[str, str],
    backend_invoker: BackendInvoker = invoke_artifact_backend_v1,
) -> tuple[
    dict[str, Any],
    tuple[BackendCaseOutputV1, ...],
    dict[str, CheckpointReceiptV1],
    Any,
]:
    """Execute the exact 84-receipt matrix from two fresh rematerializations."""

    checkpoint = _fixture_checkpoint_v1(binding)
    authority = _authority(
        binding,
        checkpoint_sha256=checkpoint["checkpoint_sha256"],
    )
    outputs: list[BackendCaseOutputV1] = []
    for repeat in REQUIRED_REPEAT_INDICES:
        inputs = _checkpoint_rows(checkpoint)
        reference_rows = [
            reference_dimensions_for_fixture_v1(
                row["family_id"],
                row["arm"],
            )
            for row in inputs
        ]
        backend_rows = {
            "reference": reference_rows,
            "native": backend_invoker(module, "native", inputs),
            "cuda": backend_invoker(module, "cuda", inputs),
        }
        for backend in REQUIRED_BACKENDS:
            for row, dimensions in zip(
                inputs,
                backend_rows[backend],
                strict=True,
            ):
                receipt_hash = _execution_receipt_sha256(
                    backend=backend,
                    repeat=repeat,
                    row=row,
                    dimensions=dimensions,
                    authority=authority,
                )
                outputs.append(
                    BackendCaseOutputV1(
                        family_id=row["family_id"],
                        arm=row["arm"],
                        backend=backend,
                        repeat_index=repeat,
                        dimensions=dimensions,
                        execution_receipt_sha256=receipt_hash,
                        no_fallback=True,
                        no_jit=True,
                    )
                )
    restore_evidence = _checkpoint_restore_evidence_v1(
        checkpoint,
        outputs,
    )
    checkpoints = {}
    for family in MANDATORY_FAMILIES:
        fixture = fixture_by_family_v1(family)
        checkpoint_execution_hash = content_sha256(
            {
                "schema_version": CHECKPOINT_EXECUTION_SCHEMA,
                "family_id": family,
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "restore_payload_sha256s": [
                    row["restore_payload_sha256"]
                    for row in restore_evidence[family]
                ],
                "exact_fixture_rematerialization": True,
            }
        )
        checkpoints[family] = CheckpointReceiptV1(
            family_id=family,
            pair_id=fixture["pair_id"],
            control_case_id=f"{fixture['pair_id']}-control",
            treatment_case_id=f"{fixture['pair_id']}-treatment",
            authority=authority,
            restored_checkpoint_sha256s=(
                checkpoint["checkpoint_sha256"],
                checkpoint["checkpoint_sha256"],
            ),
            execution_receipt_sha256=checkpoint_execution_hash,
            exact_restore=True,
        )
    result = execute_c1_v1(
        authorities={family: authority for family in MANDATORY_FAMILIES},
        backend_outputs=outputs,
        checkpoint_receipts=checkpoints,
    )
    return checkpoint, tuple(outputs), checkpoints, result


def _matrix_rows(
    outputs: Sequence[BackendCaseOutputV1],
) -> list[dict[str, Any]]:
    rows = []
    for output in outputs:
        dimensions = {
            name: list(output.dimensions[name]) for name in REQUIRED_DIMENSIONS
        }
        rows.append(
            {
                "family_id": output.family_id,
                "arm": output.arm,
                "backend": output.backend,
                "repeat_index": output.repeat_index,
                "execution_receipt_sha256": (
                    output.execution_receipt_sha256
                ),
                "dimensions_sha256": content_sha256(dimensions),
                "dimensions": dimensions,
                "no_jit": output.no_jit,
                "no_fallback": output.no_fallback,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["family_id"],
            row["arm"],
            row["backend"],
            row["repeat_index"],
        ),
    )


def write_evidence_v1(
    *,
    output_dir: Path,
    binding: Mapping[str, str],
    checkpoint: Mapping[str, Any],
    checkpoint_receipts: Mapping[str, CheckpointReceiptV1],
    outputs: Sequence[BackendCaseOutputV1],
    result: Any,
    route_receipt_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Write one immutable JSON/CSV C1 handoff without overwriting evidence."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise _fail("C1 output directory must be absent or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        name: output_dir / filename for name, filename in OUTPUT_NAMES.items()
    }
    _write_json(
        paths["preregistration"],
        {
            **preregistration_payload_v1(),
            "preregistration_sha256": PREREGISTRATION_SHA256,
        },
    )
    _write_json(paths["checkpoint"], checkpoint)
    if set(checkpoint_receipts) != set(MANDATORY_FAMILIES):
        raise _fail("C1 checkpoint receipts do not cover seven families")
    restore_evidence = _checkpoint_restore_evidence_v1(
        checkpoint,
        outputs,
    )
    result_pairs = {
        pair.control.family_id: pair
        for pair in result.pairs
    }
    if set(result_pairs) != set(MANDATORY_FAMILIES):
        raise _fail("C1 result pairs do not cover seven families")
    checkpoint_rows = []
    for family in MANDATORY_FAMILIES:
        receipt = checkpoint_receipts[family]
        expected_execution_sha256 = content_sha256(
            {
                "schema_version": CHECKPOINT_EXECUTION_SCHEMA,
                "family_id": family,
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "restore_payload_sha256s": [
                    row["restore_payload_sha256"]
                    for row in restore_evidence[family]
                ],
                "exact_fixture_rematerialization": True,
            }
        )
        if (
            not isinstance(receipt, CheckpointReceiptV1)
            or receipt.family_id != family
            or receipt.restored_checkpoint_sha256s
            != (checkpoint["checkpoint_sha256"],) * 2
            or receipt.exact_restore is not True
            or receipt.execution_receipt_sha256
            != expected_execution_sha256
            or result_pairs[family].checkpoint.receipt_sha256()
            != receipt.receipt_sha256()
        ):
            raise _fail(f"C1 checkpoint receipt is invalid: {family}")
        checkpoint_rows.append(
            {
                "family_id": family,
                "pair_id": receipt.pair_id,
                "checkpoint_receipt": {
                    **receipt.payload(),
                    "receipt_sha256": receipt.receipt_sha256(),
                },
                "restores": restore_evidence[family],
            }
        )
    _write_json(
        paths["checkpoint_receipts"],
        {
            "schema_version": CHECKPOINT_RECEIPT_SET_SCHEMA,
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "receipt_count": len(checkpoint_rows),
            "restore_count": sum(
                len(row["restores"])
                for row in checkpoint_rows
            ),
            "rows": checkpoint_rows,
        },
    )
    rows = _matrix_rows(outputs)
    _write_json(
        paths["matrix_json"],
        {
            "schema_version": "genome_v2_p10_c1_execution_matrix_v1",
            "row_count": len(rows),
            "rows": rows,
        },
    )
    with paths["matrix_csv"].open(
        "w",
        newline="",
        encoding="ascii",
    ) as stream:
        fieldnames = (
            "family_id",
            "arm",
            "backend",
            "repeat_index",
            "execution_receipt_sha256",
            "dimensions_sha256",
            "no_jit",
            "no_fallback",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {name: row[name] for name in fieldnames} for row in rows
        )
    _write_json(paths["decision"], dict(result.decision))
    evidence_files = {}
    for name in (
        "preregistration",
        "checkpoint",
        "checkpoint_receipts",
        "matrix_json",
        "matrix_csv",
        "decision",
    ):
        evidence_files[name] = {
            "path": paths[name].name,
            "sha256": _sha256(paths[name]),
            "bytes": paths[name].stat().st_size,
        }
    value = {
        "schema_version": SCHEMA_VERSION,
        "authority_binding": dict(binding),
        "route_receipt": dict(route_receipt_binding),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "checkpoint_receipt_count": len(checkpoint_rows),
        "checkpoint_restore_count": sum(
            len(row["restores"])
            for row in checkpoint_rows
        ),
        "matrix_row_count": len(rows),
        "backend_repeat_contract": {
            "backends": list(REQUIRED_BACKENDS),
            "repeat_indices": list(REQUIRED_REPEAT_INDICES),
        },
        "decision": result.decision["decision"],
        "decision_sha256": result.decision["decision_sha256"],
        "execution_sha256": result.payload()["execution_sha256"],
        "no_jit": True,
        "no_fallback": True,
        "evidence_files": evidence_files,
        "claims_boundary": (
            "fixed-fixture technical C1 only; fixture rematerialization is "
            "not a live-world checkpoint restore"
        ),
    }
    manifest = {**value, "manifest_sha256": content_sha256(value)}
    _write_json(paths["manifest"], manifest)
    for path in paths.values():
        path.chmod(0o444)
    return manifest


def execute(args: argparse.Namespace) -> int:
    artifact = args.artifact.expanduser().resolve()
    manifest_path = Path(f"{artifact}.manifest.json")
    binding_path = args.authority_binding.expanduser().resolve()
    package_receipt_path = args.package_receipt.expanduser().resolve()
    route_receipt_path = args.route_receipt.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise _fail("C1 output directory must be absent or empty")
    binding, _package, manifest, route_receipt = validate_authority_v1(
        artifact=artifact,
        manifest_path=manifest_path,
        binding_path=binding_path,
        package_receipt_path=package_receipt_path,
        route_receipt_path=route_receipt_path,
    )
    module = _load_artifact(artifact, manifest)
    provenance = dict(
        module.eiel_authoritative_state_build_provenance_v1()
    )
    if provenance.get("build_source_commit") != binding["source_commit"]:
        raise _fail("loaded artifact source commit contradicts authority")
    checkpoint, outputs, checkpoints, result = execute_matrix_v1(
        module=module,
        binding=binding,
    )
    manifest_receipt = write_evidence_v1(
        output_dir=output_dir,
        binding=binding,
        checkpoint=checkpoint,
        checkpoint_receipts=checkpoints,
        outputs=outputs,
        result=result,
        route_receipt_binding={
            "path": str(route_receipt_path),
            "sha256": _sha256(route_receipt_path),
            "execution_route_sha256": route_receipt[
                "execution_route_sha256"
            ],
            "covered_obligation_count": len(
                route_receipt["covered_obligations"]
            ),
        },
    )
    print(json.dumps(manifest_receipt, sort_keys=True))
    return 0 if result.decision["decision"] == "C1-A" else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--authority-binding", type=Path, required=True)
    parser.add_argument("--package-receipt", type=Path, required=True)
    parser.add_argument("--route-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    try:
        return execute(_parser().parse_args())
    except (GenomeV2C1ArtifactGateError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "decision": "C1-D",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
