from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.evolution.genome_upgrade_v2_environment_switch_gate_v1 import (
    BACKENDS,
    CONDITIONS,
    REPEAT_INDICES,
    evaluate_environment_switch_v1,
)
from tools.scripts import genome_upgrade_v2_environment_switch_execute_v1 as execute


def _binding() -> dict[str, str]:
    value = {
        "provider_id": "eiel-production-provider-v1",
        "runtime_profile_id": "eiel-production-runtime-v1",
        "source_commit": "1" * 40,
        "schema_version": "genome_v2_checkpoint_authority_binding_v1",
        "global_genome_profile_id": "genome_upgrade_v2",
    }
    for index, name in enumerate((
        "package_id_sha256",
        "package_receipt_sha256",
        "artifact_sha256",
        "manifest_sha256",
        "execution_route_sha256",
        "executable_package_core_sha256",
        "state_schema_sha256",
        "mechanic_contract_sha256",
        "capability_contract_sha256",
        "cancellation_abi_sha256",
        "observability_schema_sha256",
        "capacity_layout_contract_sha256",
    ), start=2):
        value[name] = format(index, "x") * 64
    return value


def test_fixed_switch_inputs_share_one_genotype_and_exact_trajectories() -> None:
    genotype, contexts, inputs = execute.fixed_switch_inputs_v1()

    assert execute.FIXTURE_FAMILY == "regulatory_activation_inhibition"
    assert len(genotype) == 120
    assert contexts["A"] != contexts["B"]
    assert len(inputs) == 8
    assert {tuple(row["genotype"]) for row in inputs} == {genotype}
    assert [
        (row["condition"], row["endpoint"], row["environment"])
        for row in inputs
    ] == [
        ("stable_a", "initial", "A"),
        ("stable_a", "terminal", "A"),
        ("stable_b", "initial", "B"),
        ("stable_b", "terminal", "B"),
        ("a_to_b", "initial", "A"),
        ("a_to_b", "terminal", "B"),
        ("b_to_a", "initial", "B"),
        ("b_to_a", "terminal", "A"),
    ]


def test_executor_invokes_each_backend_twice_and_reducer_accepts_24_rows() -> None:
    calls = {backend: 0 for backend in BACKENDS}

    def invoker(
        module: object, backend: str, inputs: list[dict]
    ) -> list[dict[str, tuple[int, ...]]]:
        assert module is marker
        calls[backend] += 1
        return execute._reference_backend(inputs)

    marker = object()
    authority, evidence, decision = execute.execute_evidence_v1(
        module=marker,
        binding=_binding(),
        backend_invoker=invoker,
    )

    assert calls == {backend: 2 for backend in BACKENDS}
    assert len(evidence) == (
        len(CONDITIONS) * len(BACKENDS) * len(REPEAT_INDICES)
    ) == 24
    assert decision["decision"] == "ENVIRONMENT-SWITCH-A"
    assert (
        evaluate_environment_switch_v1(evidence, authority=authority)[
            "decision"
        ]
        == "ENVIRONMENT-SWITCH-A"
    )
    assert len({row["genotype_sha256"] for row in evidence}) == 1
    assert all(row["no_fallback"] and row["no_jit"] for row in evidence)
    assert execute.RUNTIME_PROJECTION_FIELDS[-1] == (
        "expression_energy_debit_q10"
    )
    for row in evidence:
        for endpoint in ("initial", "terminal"):
            snapshot = row[endpoint]
            assert len(snapshot["products_q10"]) == 12
            assert len(snapshot["phenotypes_q10"]) == 10
            assert len(snapshot["runtime_consequences_q10"]) == 12
            assert snapshot["runtime_cost_q10"] == snapshot[
                "runtime_consequences_q10"
            ][-1]


def test_execute_reuses_c1_authority_loader_and_writes_bound_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binding = _binding()
    manifest = {
        "module_name": "production_module",
        "runtime_jit_compilation": False,
    }
    package = {"authority_state": "package_candidate_complete"}
    route = {"authority_state": "route_qualified"}
    artifact = tmp_path / "artifact.so"
    paths = {
        "manifest_path": tmp_path / "artifact.so.manifest.json",
        "binding_path": tmp_path / "binding.json",
        "package_receipt_path": tmp_path / "package.json",
        "route_receipt_path": tmp_path / "route.json",
    }
    observed: dict[str, object] = {}

    def validate(**kwargs: object) -> tuple[dict, dict, dict, dict]:
        observed["validate"] = kwargs
        return binding, package, manifest, route

    module = SimpleNamespace(name="production")
    monkeypatch.setattr(execute, "validate_c1_authority_v1", validate)
    monkeypatch.setattr(
        execute, "_load_artifact", lambda path, value: module
    )

    def invoker(
        loaded: object, backend: str, inputs: list[dict]
    ) -> list[dict[str, tuple[int, ...]]]:
        assert loaded is module
        return execute._reference_backend(inputs)

    output = tmp_path / "output"
    result = execute.execute(
        artifact=artifact,
        output_directory=output,
        backend_invoker=invoker,
        **paths,
    )

    assert result["ok"] is True
    assert result["row_count"] == 24
    assert result["backend_invocation_counts"] == {
        backend: 2 for backend in BACKENDS
    }
    assert result["decision"] == "ENVIRONMENT-SWITCH-A"
    assert set(result["artifacts"]) == {"authority", "evidence", "decision"}
    assert len(json.loads(
        (output / execute.OUTPUT_FILES["evidence"]).read_text(encoding="ascii")
    )) == 24
    assert observed["validate"] == {
        "artifact": artifact,
        **paths,
    }
    with pytest.raises(FileExistsError):
        execute.execute(
            artifact=artifact,
            output_directory=output,
            backend_invoker=invoker,
            **paths,
        )
