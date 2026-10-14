from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from engine.evolution.genome_upgrade_v2_c1_executor_v1 import (
    reference_dimensions_for_fixture_v1,
)
from engine.evolution.genome_upgrade_v2_causal_gate_v1 import (
    MANDATORY_FAMILIES,
    REQUIRED_DIMENSIONS,
)
from engine.evolution.genome_upgrade_v2_package_v1 import (
    package_identity_sha256_v1,
)
from tools.scripts import genome_upgrade_v2_c1_artifact_gate_v1 as gate


HASH = "a" * 64
COMMIT = "b" * 40


def _binding() -> dict[str, str]:
    value = {
        "schema_version": "genome_v2_checkpoint_authority_binding_v1",
        "provider_id": "eiel-production-provider-v1",
        "runtime_profile_id": "eiel-production-runtime-v1",
        "global_genome_profile_id": "genome_upgrade_v2",
        "source_commit": COMMIT,
    }
    for name in (
        "executable_package_core_sha256",
        "package_id_sha256",
        "package_receipt_sha256",
        "artifact_sha256",
        "manifest_sha256",
        "state_schema_sha256",
        "mechanic_contract_sha256",
        "capability_contract_sha256",
        "cancellation_abi_sha256",
        "observability_schema_sha256",
        "capacity_layout_contract_sha256",
        "execution_route_sha256",
    ):
        value[name] = HASH
    return value


def _reference_invoker(
    _module,
    backend: str,
    inputs,
) -> list[dict[str, tuple[int, ...]]]:
    assert backend in {"native", "cuda"}
    return [
        reference_dimensions_for_fixture_v1(
            row["family_id"],
            row["arm"],
        )
        for row in inputs
    ]


def test_matrix_runs_two_fresh_native_and_cuda_batches_and_decides_a() -> None:
    calls = []

    def invoke(module, backend, inputs):
        calls.append((module, backend, id(inputs), len(inputs)))
        return _reference_invoker(module, backend, inputs)

    checkpoint, outputs, checkpoints, result = gate.execute_matrix_v1(
        module=object(),
        binding=_binding(),
        backend_invoker=invoke,
    )

    assert result.decision["decision"] == "C1-A"
    assert len(outputs) == 84
    assert len(checkpoints) == 7
    assert [(row[1], row[3]) for row in calls] == [
        ("native", 14),
        ("cuda", 14),
        ("native", 14),
        ("cuda", 14),
    ]
    assert len({row[2] for row in calls}) == 2
    assert checkpoint["restore_semantics"].endswith(
        "not a live-world checkpoint"
    )
    assert all(
        receipt.restored_checkpoint_sha256s
        == (checkpoint["checkpoint_sha256"],) * 2
        and receipt.exact_restore is True
        for receipt in checkpoints.values()
    )
    restore_evidence = gate._checkpoint_restore_evidence_v1(
        checkpoint,
        outputs,
    )
    assert set(restore_evidence) == set(MANDATORY_FAMILIES)
    assert all(
        len(restores) == 2
        and all(
            len(row["backend_execution_receipts"]) == 6
            and row["restore_payload_sha256"]
            == gate.content_sha256(
                {
                    name: value
                    for name, value in row.items()
                    if name != "restore_payload_sha256"
                }
            )
            for row in restores
        )
        for restores in restore_evidence.values()
    )


def test_evidence_writer_emits_exact_immutable_json_csv_handoff(
    tmp_path: Path,
) -> None:
    checkpoint, outputs, checkpoints, result = gate.execute_matrix_v1(
        module=object(),
        binding=_binding(),
        backend_invoker=_reference_invoker,
    )
    output = tmp_path / "c1"
    manifest = gate.write_evidence_v1(
        output_dir=output,
        binding=_binding(),
        checkpoint=checkpoint,
        checkpoint_receipts=checkpoints,
        outputs=outputs,
        result=result,
        route_receipt_binding={
            "path": "route.json",
            "sha256": HASH,
            "execution_route_sha256": HASH,
            "covered_obligation_count": 20,
        },
    )

    assert manifest["decision"] == "C1-A"
    assert manifest["matrix_row_count"] == 84
    assert manifest["checkpoint_receipt_count"] == 7
    assert manifest["checkpoint_restore_count"] == 14
    assert manifest["claims_boundary"].endswith(
        "not a live-world checkpoint restore"
    )
    for filename in gate.OUTPUT_NAMES.values():
        path = output / filename
        assert path.is_file()
        assert path.stat().st_mode & 0o222 == 0
    matrix = json.loads(
        (output / gate.OUTPUT_NAMES["matrix_json"]).read_text()
    )
    assert matrix["row_count"] == 84
    checkpoint_receipts = json.loads(
        (output / gate.OUTPUT_NAMES["checkpoint_receipts"]).read_text()
    )
    assert checkpoint_receipts["receipt_count"] == 7
    assert checkpoint_receipts["restore_count"] == 14
    matrix_receipts = {
        (
            row["family_id"],
            row["repeat_index"],
            row["arm"],
            row["backend"],
            row["execution_receipt_sha256"],
        )
        for row in matrix["rows"]
    }
    assert all(
        len(row["restores"]) == 2
        and row["checkpoint_receipt"]["exact_restore"] is True
        and len(row["checkpoint_receipt"]["receipt_sha256"]) == 64
        and all(
            restore["restore_payload_sha256"]
            == gate.content_sha256(
                {
                    name: value
                    for name, value in restore.items()
                    if name != "restore_payload_sha256"
                }
            )
            and all(
                (
                    restore["family_id"],
                    restore["repeat_index"],
                    execution["arm"],
                    execution["backend"],
                    execution["execution_receipt_sha256"],
                )
                in matrix_receipts
                for execution in restore[
                    "backend_execution_receipts"
                ]
            )
            for restore in row["restores"]
        )
        for row in checkpoint_receipts["rows"]
    )
    with (output / gate.OUTPUT_NAMES["matrix_csv"]).open(
        newline="",
        encoding="ascii",
    ) as stream:
        assert len(list(csv.DictReader(stream))) == 84
    with pytest.raises(
        gate.GenomeV2C1ArtifactGateError,
        match="absent or empty",
    ):
        gate.write_evidence_v1(
            output_dir=output,
            binding=_binding(),
            checkpoint=checkpoint,
            checkpoint_receipts=checkpoints,
            outputs=outputs,
            result=result,
            route_receipt_binding={
                "path": "route.json",
                "sha256": HASH,
                "execution_route_sha256": HASH,
                "covered_obligation_count": 20,
            },
        )


def _authority_files(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path, dict[str, str]]:
    artifact = tmp_path / "production.so"
    artifact.write_bytes(b"production artifact")
    artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest_path = Path(f"{artifact}.manifest.json")
    manifest = {
        "module_name": "production_module",
        "artifact_sha256": artifact_sha,
        "runtime_jit_compilation": False,
        "immutable_output": True,
        "source_snapshot_stable": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="ascii",
    )
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    identity = {
        "provider_id": "eiel-production-provider-v1",
        "runtime_profile_id": "eiel-production-runtime-v1",
        "global_genome_profile_id": "genome_upgrade_v2",
        "source_commit": COMMIT,
        "executable_package_core_sha256": "1" * 64,
        "cuda_artifact_sha256": artifact_sha,
        "cuda_manifest_sha256": manifest_sha,
        "authoritative_state_schema_sha256": "2" * 64,
        "mechanic_contract_sha256": "3" * 64,
        "capability_contract_sha256": "4" * 64,
        "cancellation_abi_sha256": "5" * 64,
        "observability_storage_budget_sha256": "6" * 64,
        "capacity_layout_contract_sha256": "7" * 64,
    }
    receipt = {
        "authority_state": "package_candidate_complete",
        "ok": True,
        "identity": identity,
    }
    receipt["package_id_sha256"] = package_identity_sha256_v1(receipt)
    receipt_path = tmp_path / "package.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    binding = {
        "schema_version": "genome_v2_checkpoint_authority_binding_v1",
        "provider_id": identity["provider_id"],
        "runtime_profile_id": identity["runtime_profile_id"],
        "global_genome_profile_id": identity["global_genome_profile_id"],
        "source_commit": identity["source_commit"],
        "executable_package_core_sha256": identity[
            "executable_package_core_sha256"
        ],
        "package_id_sha256": receipt["package_id_sha256"],
        "package_receipt_sha256": hashlib.sha256(
            receipt_path.read_bytes()
        ).hexdigest(),
        "artifact_sha256": artifact_sha,
        "manifest_sha256": manifest_sha,
        "state_schema_sha256": identity[
            "authoritative_state_schema_sha256"
        ],
        "mechanic_contract_sha256": identity[
            "mechanic_contract_sha256"
        ],
        "capability_contract_sha256": identity[
            "capability_contract_sha256"
        ],
        "cancellation_abi_sha256": identity[
            "cancellation_abi_sha256"
        ],
        "observability_schema_sha256": identity[
            "observability_storage_budget_sha256"
        ],
        "capacity_layout_contract_sha256": identity[
            "capacity_layout_contract_sha256"
        ],
        "execution_route_sha256": "8" * 64,
    }
    binding_path = tmp_path / "binding.json"
    binding_path.write_text(
        json.dumps(binding, sort_keys=True) + "\n",
        encoding="ascii",
    )
    route_path = tmp_path / "route.json"
    route_path.write_text(
        json.dumps(
            {
                "authority_state": "route_qualified",
                "execution_route_sha256": binding[
                    "execution_route_sha256"
                ],
                "artifact_sha256": artifact_sha,
                "provider_id": binding["provider_id"],
                "profile_id": binding["runtime_profile_id"],
                "covered_obligations": sorted(
                    gate.EXECUTION_ROUTE_OBLIGATIONS
                ),
                "missing_obligations": [],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    return (
        artifact,
        manifest_path,
        binding_path,
        receipt_path,
        route_path,
        binding,
    )


def test_authority_validation_joins_exact_package_route_and_artifact(
    tmp_path: Path,
) -> None:
    (
        artifact,
        manifest,
        binding_path,
        receipt,
        route,
        binding,
    ) = _authority_files(tmp_path)
    actual, package, artifact_manifest, route_receipt = (
        gate.validate_authority_v1(
        artifact=artifact,
        manifest_path=manifest,
        binding_path=binding_path,
        package_receipt_path=receipt,
        route_receipt_path=route,
        )
    )
    assert actual == binding
    assert package["package_id_sha256"] == binding["package_id_sha256"]
    assert artifact_manifest["artifact_sha256"] == binding["artifact_sha256"]
    assert route_receipt["authority_state"] == "route_qualified"

    artifact.write_bytes(b"substituted")
    with pytest.raises(
        gate.GenomeV2C1ArtifactGateError,
        match="do not join",
    ):
        gate.validate_authority_v1(
            artifact=artifact,
            manifest_path=manifest,
            binding_path=binding_path,
            package_receipt_path=receipt,
            route_receipt_path=route,
        )


def test_native_artifact_output_requires_exact_metadata_tensors_and_echo() -> None:
    inputs = [
        {
            "family_id": family,
            "arm": arm,
            **{
                name: reference_dimensions_for_fixture_v1(family, arm)[name]
                for name in ("genotype", "context")
            },
        }
        for family in MANDATORY_FAMILIES
        for arm in ("control", "treatment")
    ]
    values = [
        reference_dimensions_for_fixture_v1(
            row["family_id"],
            row["arm"],
        )
        for row in inputs
    ]
    raw = {
        "schema_version": gate.ARTIFACT_OUTPUT_SCHEMA,
        "backend": "native",
        "no_jit": True,
        "no_fallback": True,
        "production_artifact_endpoint": True,
        **{
            name: torch.tensor(
                [row[name] for row in values],
                dtype=torch.int64,
            )
            for name in REQUIRED_DIMENSIONS
        },
    }
    actual = gate._validated_artifact_output(
        raw,
        backend="native",
        inputs=inputs,
    )
    assert actual == values

    raw["cost"] = raw["cost"][:, :1]
    with pytest.raises(
        gate.GenomeV2C1ArtifactGateError,
        match="invalid cost",
    ):
        gate._validated_artifact_output(
            raw,
            backend="native",
            inputs=inputs,
        )


def test_authority_validation_rejects_incomplete_route_obligations(
    tmp_path: Path,
) -> None:
    artifact, manifest, binding, package, route, _value = _authority_files(
        tmp_path
    )
    route_payload = json.loads(route.read_text(encoding="ascii"))
    route_payload["covered_obligations"].pop()
    route.write_text(
        json.dumps(route_payload, sort_keys=True) + "\n",
        encoding="ascii",
    )
    with pytest.raises(
        gate.GenomeV2C1ArtifactGateError,
        match="exact qualified route",
    ):
        gate.validate_authority_v1(
            artifact=artifact,
            manifest_path=manifest,
            binding_path=binding,
            package_receipt_path=package,
            route_receipt_path=route,
        )


def test_cuda_unavailability_fails_closed_without_calling_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    module = SimpleNamespace(
        **{
            gate.CUDA_ENTRYPOINT: lambda *_args: pytest.fail(
                "CUDA endpoint must not run"
            ),
            gate.NATIVE_ENTRYPOINT: lambda *_args: pytest.fail(
                "native fallback must not run"
            ),
        }
    )
    with pytest.raises(
        gate.GenomeV2C1ArtifactGateError,
        match="does not fall back",
    ):
        gate.invoke_artifact_backend_v1(
            module,
            "cuda",
            [
                {
                    "genotype": (0,) * 120,
                    "context": (0,) * 8,
                }
            ],
        )
