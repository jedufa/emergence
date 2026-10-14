#!/usr/bin/env python3
"""Issue a source- and artifact-bound Genome V2 package candidate receipt."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from engine.authoritative_state.schema import content_sha256, load_schema_v1
from engine.evolution.genome_upgrade_v2 import CONTRACT_SHA256, contract_payload
from engine.evolution.genome_upgrade_v2_capability_contract_v1 import (
    GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256,
)
from engine.evolution.genome_upgrade_v2_causal_gate_v1 import (
    PREREGISTRATION_SHA256 as CAUSAL_GATE_PREREGISTRATION_SHA256,
    SCHEMA_VERSION as CAUSAL_GATE_SCHEMA_VERSION,
)
from engine.evolution.genome_upgrade_v2_causal_settlement_v1 import (
    CONTRACT_SHA256 as CAUSAL_SETTLEMENT_CONTRACT_SHA256,
    SCHEMA_VERSION as CAUSAL_SETTLEMENT_SCHEMA_VERSION,
)
from engine.evolution.genome_upgrade_v2_cancellation_v1 import (
    cancellation_contract_v1,
)
from engine.evolution.genome_upgrade_v2_e1_gate_v1 import (
    PREREGISTRATION_SHA256 as ECOLOGICAL_E1_PREREGISTRATION_SHA256,
    SCHEMA_VERSION as ECOLOGICAL_E1_SCHEMA_VERSION,
)
from engine.evolution.genome_upgrade_v2_environment_switch_gate_v1 import (
    PREREGISTRATION_SHA256 as ENVIRONMENT_SWITCH_PREREGISTRATION_SHA256,
    SCHEMA_VERSION as ENVIRONMENT_SWITCH_SCHEMA_VERSION,
)
from engine.evolution.genome_upgrade_v2_hereditary_capacity_gate_v1 import (
    PREREGISTRATION_SHA256 as HEREDITARY_CAPACITY_PREREGISTRATION_SHA256,
    SCHEMA_VERSION as HEREDITARY_CAPACITY_SCHEMA_VERSION,
)
from engine.evolution.genome_upgrade_v2_package_v1 import (
    PACKAGE_IDENTITY_RECEIPT_SCHEMA_VERSION,
    PRE_ROUTE_PACKAGE_OBLIGATIONS,
    capacity_layout_contract_v1,
    checkpoint_authority_contract_v1,
    executable_package_contract_v1,
    obligation_dictionary_v1,
    package_core_identity_sha256_v1,
    package_identity_sha256_v1,
    package_invalidation_decision_v1,
    package_invalidation_field_map_v1,
    route_input_contract_v1,
    supported_checkpoint_cut_ticks_v1,
    supported_world_controller_absolute_tick_envelope_v1,
)
from engine.evolution.genome_upgrade_v2_promotion_contracts_v1 import (
    promotion_contract_artifacts_v1,
)
from engine.evolution.genome_upgrade_v2_telemetry import (
    chronicle_schema_v1,
    observability_storage_budget_v1,
    reporting_schema_v1,
)
from engine.evolution.global_genome_registry_v1 import global_genome_profile_v1
from tools.scripts.genome_upgrade_v2_state_inventory import write_artifacts


SCHEMA_VERSION = "genome_v2_package_qualification_gate_v1"

# These obligations assert behavior observed through the loaded CUDA artifact.
# A receipt from a different binary cannot establish the claim for the package
# being issued, even when the receipt file itself is intact.  The remaining
# source-contract inventory obligations are deliberately excluded: they are
# structural proofs, not observations of a loaded artifact.
_ARTIFACT_BOUND_OBLIGATIONS = frozenset(
    PRE_ROUTE_PACKAGE_OBLIGATIONS
) - frozenset(
    {
        "genome_v2_default_off_exact",
        "genome_v2_neutral_profile_exact",
        "genome_v2_reference_native_cuda_expression_parity",
        "genome_v2_inheritance_mutation_parity",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ("git", *args), cwd=root, text=True
    ).strip()


def _source_tree_sha256(root: Path, commit: str) -> str:
    listing = subprocess.check_output(
        ("git", "ls-tree", "-r", "--full-tree", commit), cwd=root
    )
    return hashlib.sha256(listing).hexdigest()


def _artifact_digests(value: Any) -> set[str]:
    """Return artifact digests declared by a JSON evidence payload.

    Receipts have a few intentionally compatible envelope shapes.  Restrict
    the search to semantic artifact fields, rather than accepting an expected
    digest merely because it appears in arbitrary receipt text.
    """
    found: set[str] = set()
    if isinstance(value, dict):
        artifact_sha256 = value.get("artifact_sha256")
        if isinstance(artifact_sha256, str):
            found.add(artifact_sha256)
        artifact = value.get("artifact")
        if isinstance(artifact, dict):
            sha256 = artifact.get("sha256")
            if isinstance(sha256, str):
                found.add(sha256)
        artifact_identity = value.get("artifact_identity")
        if isinstance(artifact_identity, dict):
            sha256 = artifact_identity.get("sha256")
            if isinstance(sha256, str):
                found.add(sha256)
        for child in value.values():
            found.update(_artifact_digests(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_artifact_digests(child))
    return found


def _evidence_rows(
    values: list[str], *, artifact_sha256: str
) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for value in values:
        obligation, separator, raw_path = value.partition("=")
        if not separator or obligation not in PRE_ROUTE_PACKAGE_OBLIGATIONS:
            raise ValueError(f"invalid --evidence binding: {value!r}")
        path = Path(raw_path).expanduser().resolve()
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"evidence must be an existing non-symlink file: {path}"
            )
        if obligation in _ARTIFACT_BOUND_OBLIGATIONS:
            try:
                declared_digests = _artifact_digests(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "artifact-bound evidence must be readable JSON: "
                    f"{path}"
                ) from exc
            if artifact_sha256 not in declared_digests:
                raise ValueError(
                    "artifact-bound evidence does not declare the exact "
                    f"package artifact for {obligation}: {path}"
                )
        rows[obligation] = {
            "status": "passed",
            "evidence_path": str(path),
            "evidence_sha256": _sha256(path),
        }
    missing = sorted(set(PRE_ROUTE_PACKAGE_OBLIGATIONS) - set(rows))
    extra_count = len(values) - len(rows)
    if missing or extra_count:
        raise ValueError(
            f"evidence bindings must be exact; missing={missing}, "
            f"duplicate_count={extra_count}"
        )
    return rows


def _manifest(artifact: Path) -> tuple[Path, dict[str, Any]]:
    path = artifact.with_suffix(artifact.suffix + ".manifest.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("artifact_sha256") != _sha256(artifact)
        or payload.get("artifact_identity", {}).get("sha256")
        != _sha256(artifact)
        or payload.get("elf_build_id")
        != payload.get("artifact_identity", {}).get("elf_build_id")
        or payload.get("immutable_output") is not True
        or payload.get("sass_only") is not True
        or payload.get("runtime_jit_compilation") is not False
        or payload.get("cuda_arch") != "sm_75"
    ):
        raise ValueError("CUDA artifact manifest is invalid or stale")
    return path, payload


def _contract_hashes() -> dict[str, str]:
    mechanic = contract_payload()
    return {
        "genome_envelope_sha256": content_sha256(
            {
                "allele_parameters": mechanic["allele_parameters"],
                "product_channels": mechanic["product_channels"],
                "phenotype_channels": mechanic["phenotype_channels"],
                "context_channels": mechanic["context_channels"],
                "numeric_contract": mechanic["numeric_contract"],
            }
        ),
        "product_schema_sha256": content_sha256(
            {
                "channels": mechanic["product_channels"],
                "context_index": mechanic["product_context_index"],
                "cost_q10": mechanic["product_cost_q10"],
            }
        ),
        "phenotype_schema_sha256": content_sha256(
            {
                "channels": mechanic["phenotype_channels"],
                "weights_q10": mechanic["polygenic_weights_q10"],
                "runtime_projection": mechanic["runtime_projection"],
            }
        ),
        "context_schema_sha256": content_sha256(
            {
                "channels": mechanic["context_channels"],
                "runtime_context": mechanic["runtime_context"],
            }
        ),
        "linkage_map_sha256": content_sha256(
            mechanic["linkage_block_by_product"]
        ),
        "inheritance_mutation_draw_contract_sha256": content_sha256(
            {
                "inheritance_version": mechanic["inheritance_version"],
                "meiosis_draw_count": mechanic["meiosis_draw_count"],
                "mutation_semantic_draw_count": mechanic[
                    "mutation_semantic_draw_count"
                ],
                "hidden_rng_draws": mechanic["hidden_rng_draws"],
            }
        ),
    }


def _identity(
    *,
    root: Path,
    artifact: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    contracts: Path,
    obligations: dict[str, dict[str, str]],
) -> dict[str, Any]:
    source_commit = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain")
    committed = manifest.get("committed_source_snapshot")
    if (
        status
        or not isinstance(committed, dict)
        or committed.get("build_source_commit") != source_commit
    ):
        raise ValueError(
            "package qualification requires a clean tree and artifact built "
            "from the exact HEAD commit"
        )
    schema = load_schema_v1()
    components = schema.root["component_hashes"]
    profile = global_genome_profile_v1("genome_upgrade_v2")
    cancellation = cancellation_contract_v1()
    promotion = promotion_contract_artifacts_v1()
    environment = {
        "compiler_path": manifest.get("compiler_path"),
        "compiler_version": manifest.get("compiler_version"),
        "nvcc_path": manifest.get("nvcc_path"),
        "nvcc_version": manifest.get("nvcc_version"),
        "cuda_home": manifest.get("cuda_home"),
        "torch_version": manifest.get("torch_version"),
        "torch_cuda_version": manifest.get("torch_cuda_version"),
        "python_version": manifest.get("python_version"),
    }
    flags = {
        "cxx_flags": manifest.get("cxx_flags"),
        "nvcc_flags": manifest.get("nvcc_flags"),
        "link_flags": manifest.get("link_flags"),
    }
    identity: dict[str, Any] = {
        "source_commit": source_commit,
        "source_tree_sha256": _source_tree_sha256(root, source_commit),
        "source_clean": True,
        "diagnostic_only": False,
        "provider_id": "eiel-production-provider-v1",
        "runtime_profile_id": "eiel-production-runtime-v1",
        "authoritative_state_schema_version": schema.root["schema_version"],
        "authoritative_state_schema_sha256": schema.schema_hash,
        "global_genome_profile_id": "genome_upgrade_v2",
        "global_genome_profile_sha256": content_sha256(profile.payload()),
        "mechanic_contract_sha256": CONTRACT_SHA256,
        "capability_contract_sha256": (
            GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256
        ),
        **_contract_hashes(),
        "inheritance_authority_contract_sha256": content_sha256(
            {
                "profile": profile.payload(),
                "cardinality": "exactly_one_global_genome_profile",
                "mixed_authority": "forbidden_without_composition_contract",
            }
        ),
        "ordinary_birth_abi_schema": promotion[
            "genome_v2_birth_transaction_contract.json"
        ]["schema_version"],
        "ordinary_birth_abi_sha256": content_sha256(
            promotion["genome_v2_birth_transaction_contract.json"]
        ),
        "turnover_abi_schema": promotion[
            "genome_v2_turnover_transaction_contract.json"
        ]["schema_version"],
        "turnover_abi_sha256": content_sha256(
            promotion["genome_v2_turnover_transaction_contract.json"]
        ),
        "cancellation_abi_schema": cancellation["schema_version"],
        "cancellation_abi_sha256": content_sha256(cancellation),
        "causal_settlement_schema_version": (
            CAUSAL_SETTLEMENT_SCHEMA_VERSION
        ),
        "causal_settlement_contract_sha256": (
            CAUSAL_SETTLEMENT_CONTRACT_SHA256
        ),
        "causal_gate_schema_version": CAUSAL_GATE_SCHEMA_VERSION,
        "causal_gate_preregistration_sha256": (
            CAUSAL_GATE_PREREGISTRATION_SHA256
        ),
        "ecological_e1_schema_version": ECOLOGICAL_E1_SCHEMA_VERSION,
        "ecological_e1_preregistration_sha256": (
            ECOLOGICAL_E1_PREREGISTRATION_SHA256
        ),
        "hereditary_capacity_schema_version": (
            HEREDITARY_CAPACITY_SCHEMA_VERSION
        ),
        "hereditary_capacity_preregistration_sha256": (
            HEREDITARY_CAPACITY_PREREGISTRATION_SHA256
        ),
        "environment_switch_schema_version": (
            ENVIRONMENT_SWITCH_SCHEMA_VERSION
        ),
        "environment_switch_preregistration_sha256": (
            ENVIRONMENT_SWITCH_PREREGISTRATION_SHA256
        ),
        "checkpoint_schema_version": schema.components[
            "checkpoint_schema.json"
        ]["schema_version"],
        "checkpoint_schema_sha256": components["checkpoint_schema.json"],
        "chronicle_schema_version": schema.components[
            "chronicle_schema.json"
        ]["schema_version"],
        "chronicle_schema_sha256": components["chronicle_schema.json"],
        "genome_v2_chronicle_schema_version": chronicle_schema_v1()[
            "schema_version"
        ],
        "genome_v2_chronicle_schema_sha256": content_sha256(
            chronicle_schema_v1()
        ),
        "reporting_schema_version": reporting_schema_v1()["schema_version"],
        "reporting_schema_sha256": content_sha256(reporting_schema_v1()),
        "observability_storage_budget_schema_version": (
            observability_storage_budget_v1()["schema_version"]
        ),
        "observability_storage_budget_sha256": content_sha256(
            observability_storage_budget_v1()
        ),
        "executable_contract_sha256": _sha256(
            contracts / "genome_v2_executable_contract.json"
        ),
        "obligation_set_sha256": obligation_dictionary_v1()[
            "obligation_set_sha256"
        ],
        "executable_package_core_sha256": "",
        "evidence_inventory_sha256": content_sha256(obligations),
        "cuda_artifact_sha256": _sha256(artifact),
        "loaded_artifact_sha256": _sha256(artifact),
        "cuda_manifest_sha256": _sha256(manifest_path),
        "elf_build_id": manifest["elf_build_id"],
        "capacity_layout_contract_sha256": content_sha256(
            capacity_layout_contract_v1()
        ),
        "checkpoint_authority_contract_sha256": content_sha256(
            checkpoint_authority_contract_v1()
        ),
        "route_contract_inputs_sha256": content_sha256(
            route_input_contract_v1()
        ),
        "compiler_cuda_torch_environment": environment,
        "compiler_cuda_torch_environment_sha256": content_sha256(environment),
        "critical_build_flags_sha256": content_sha256(flags),
        "target_architecture": "sm_75",
        "no_jit": True,
        "no_fallback": True,
        "supported_world_controller_absolute_tick_envelope": (
            supported_world_controller_absolute_tick_envelope_v1()
        ),
        "supported_checkpoint_cut_ticks": supported_checkpoint_cut_ticks_v1(),
        "supported_campaign_classes": [
            "qualification",
            "fixed_genotype_causal",
            "bounded_ecological",
        ],
    }
    identity["executable_package_core_sha256"] = (
        package_core_identity_sha256_v1(identity)
    )
    required = set(
        executable_package_contract_v1()["required_identity_fields"]
    )
    if set(identity) != required:
        raise ValueError(
            "package identity implementation differs from contract: "
            f"missing={sorted(required - set(identity))}, "
            f"extra={sorted(set(identity) - required)}"
        )
    return identity


def _write_invalidation_matrix(
    output: Path, identity: dict[str, Any]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for surface, fields in package_invalidation_field_map_v1().items():
        candidate = dict(identity)
        field = fields[0]
        value = candidate[field]
        candidate[field] = (
            not value if isinstance(value, bool) else f"changed:{value}"
        )
        decision = package_invalidation_decision_v1(identity, candidate)
        rows.append(
            {
                "case_id": f"change_{surface}",
                "changed_field": field,
                "expected": "requalify",
                "actual": decision["action"],
                "status": (
                    "PASS" if decision["action"] == "requalify" else "FAIL"
                ),
            }
        )
    unchanged = dict(identity)
    unchanged["documentation_note"] = "non-normative text changed"
    decision = package_invalidation_decision_v1(identity, unchanged)
    rows.append(
        {
            "case_id": "documentation_only_change",
            "changed_field": "documentation_note",
            "expected": "reuse",
            "actual": decision["action"],
            "status": "PASS" if decision["action"] == "reuse" else "FAIL",
        }
    )
    path = output / "genome_v2_package_invalidation_matrix.csv"
    with path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "schema_version": "genome_v2_package_invalidation_gate_v1",
        "ok": all(row["status"] == "PASS" for row in rows),
        "case_count": len(rows),
        "matrix_path": str(path),
        "matrix_sha256": _sha256(path),
    }


def execute(args: argparse.Namespace) -> int:
    root = args.repository_root.resolve()
    artifact = args.artifact.resolve()
    output = args.output_dir.resolve()
    contracts = output / "contracts"
    output.mkdir(parents=True, exist_ok=True)
    write_artifacts(contracts)
    manifest_path, manifest = _manifest(artifact)
    obligations = _evidence_rows(args.evidence, artifact_sha256=_sha256(artifact))
    identity = _identity(
        root=root,
        artifact=artifact,
        manifest_path=manifest_path,
        manifest=manifest,
        contracts=contracts,
        obligations=obligations,
    )
    invalidation = _write_invalidation_matrix(output, identity)
    receipt = {
        "schema_version": PACKAGE_IDENTITY_RECEIPT_SCHEMA_VERSION,
        "authority_state": "package_candidate_complete",
        "ok": invalidation["ok"],
        "package_contract_sha256": _sha256(
            contracts / "genome_v2_executable_package_contract_v1.json"
        ),
        "obligation_dictionary_sha256": _sha256(
            contracts / "genome_v2_package_obligation_dictionary.json"
        ),
        "invalidation_rules_sha256": _sha256(
            contracts / "genome_v2_package_invalidation_rules.json"
        ),
        "executable_contract_sha256": _sha256(
            contracts / "genome_v2_executable_contract.json"
        ),
        "identity": identity,
        "obligations": obligations,
    }
    receipt["package_id_sha256"] = package_identity_sha256_v1(receipt)
    receipt_path = output / "genome_v2_package_identity_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "decision": (
            "GENOME-V2-PACKAGE-CANDIDATE-PASS"
            if receipt["ok"]
            else "GENOME-V2-PACKAGE-CANDIDATE-FAIL"
        ),
        "ok": receipt["ok"],
        "package_id_sha256": receipt["package_id_sha256"],
        "source_commit": identity["source_commit"],
        "artifact_sha256": identity["cuda_artifact_sha256"],
        "manifest_sha256": identity["cuda_manifest_sha256"],
        "pre_route_obligation_count": len(obligations),
        "invalidation": invalidation,
        "route_qualification_status": "pending_separate_T3_authority",
    }
    (output / "genome_v2_package_qualification_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if receipt["ok"] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--evidence",
        action="append",
        default=[],
        metavar="OBLIGATION=PATH",
    )
    return execute(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
