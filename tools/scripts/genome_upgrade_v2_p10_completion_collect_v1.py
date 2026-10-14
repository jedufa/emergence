#!/usr/bin/env python3
"""Assemble exact, content-addressed Genome V2 P1.0 completion evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.authoritative_state.schema import content_sha256  # noqa: E402
from engine.campaigns.qualification import (  # noqa: E402
    EXECUTION_ROUTE_OBLIGATIONS_V1,
    EXECUTION_ROUTE_QUALIFICATION_SCHEMA_VERSION,
)
from engine.evolution.genome_upgrade_v2_capability_contract_v1 import (  # noqa: E402
    GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256,
    validate_genome_v2_p10_capability_contract_v1,
)
from engine.evolution.genome_upgrade_v2_causal_gate_v1 import (  # noqa: E402
    PREREGISTRATION_SHA256 as C1_PREREGISTRATION_SHA256,
)
from engine.evolution.genome_upgrade_v2_e1_gate_v1 import (  # noqa: E402
    CAMPAIGNS,
    DECISION_A as E1_DECISION_A,
    DECISION_B as E1_DECISION_B,
    PREREGISTRATION_SHA256 as E1_PREREGISTRATION_SHA256,
    _authority_defects,
)
from engine.evolution.genome_upgrade_v2_environment_switch_gate_v1 import (  # noqa: E402
    DECISION_A as ENVIRONMENT_DECISION_A,
    PREREGISTRATION_SHA256 as ENVIRONMENT_PREREGISTRATION_SHA256,
    SCHEMA_VERSION as ENVIRONMENT_SCHEMA_VERSION,
    validate_authority_v1 as validate_environment_authority_v1,
)
from engine.evolution.genome_upgrade_v2_hereditary_capacity_gate_v1 import (  # noqa: E402
    DECISION_A as HEREDITARY_DECISION_A,
    PREREGISTRATION_SHA256 as HEREDITARY_PREREGISTRATION_SHA256,
    SCHEMA_VERSION as HEREDITARY_SCHEMA_VERSION,
    validate_authority_v1 as validate_hereditary_authority_v1,
)
from engine.evolution.genome_upgrade_v2_p10_completion_gate_v1 import (  # noqa: E402
    EVIDENCE_SCHEMA_VERSION,
    PACKAGE_CAPABILITIES,
    PREREGISTRATION_SHA256,
    WORK_PACKAGES,
)
from engine.evolution.genome_upgrade_v2_package_v1 import (  # noqa: E402
    PACKAGE_IDENTITY_RECEIPT_SCHEMA_VERSION,
    package_identity_sha256_v1,
)
from tools.scripts.genome_upgrade_v2_e1_gate_v1 import (  # noqa: E402
    authority_from_payload_v1,
)


PROVIDER_ID = "eiel-production-provider-v1"
RUNTIME_PROFILE_ID = "eiel-production-runtime-v1"
GATE_RECEIPT_SCHEMA_VERSION = "genome_v2_p10_completion_gate_receipt_v1"
COLLECTOR_SCHEMA_VERSION = "genome_v2_p10_completion_collector_v1"
SINGLE_CELL_EXECUTION_ROUTE_OBLIGATIONS_V1 = (
    EXECUTION_ROUTE_OBLIGATIONS_V1
    - {
        "alternate_batch_partition",
        "multicell_execution",
        "odd_physical_tail",
    }
)


class CompletionCollectorError(ValueError):
    """The supplied terminal evidence cannot form one authoritative package."""


def _fail(message: str) -> None:
    raise CompletionCollectorError(message)


def _object(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be a JSON object")
    return dict(value)


def _hash(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} must be a lowercase SHA-256")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} must be an existing non-symlink file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CompletionCollectorError(f"{label} is not readable JSON") from exc
    return _object(value, label=label)


def validate_package_authority_v1(
    path: Path,
) -> tuple[dict[str, Any], str, str]:
    receipt = _read_json(path, label="package receipt")
    if (
        receipt.get("schema_version")
        != PACKAGE_IDENTITY_RECEIPT_SCHEMA_VERSION
        or receipt.get("authority_state") != "package_candidate_complete"
        or receipt.get("ok") is not True
    ):
        _fail("package receipt is not a complete package candidate")
    package_id = _hash(
        receipt.get("package_id_sha256"), label="package_id_sha256"
    )
    if package_identity_sha256_v1(receipt) != package_id:
        _fail("package receipt self-address is invalid")
    identity = _object(receipt.get("identity"), label="package identity")
    expected = {
        "provider_id": PROVIDER_ID,
        "runtime_profile_id": RUNTIME_PROFILE_ID,
        "global_genome_profile_id": "genome_upgrade_v2",
        "source_clean": True,
        "diagnostic_only": False,
        "no_fallback": True,
        "no_jit": True,
        "capability_contract_sha256": (
            GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256
        ),
        "causal_gate_preregistration_sha256": (
            C1_PREREGISTRATION_SHA256
        ),
        "ecological_e1_preregistration_sha256": (
            E1_PREREGISTRATION_SHA256
        ),
        "hereditary_capacity_schema_version": HEREDITARY_SCHEMA_VERSION,
        "hereditary_capacity_preregistration_sha256": (
            HEREDITARY_PREREGISTRATION_SHA256
        ),
        "environment_switch_schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "environment_switch_preregistration_sha256": (
            ENVIRONMENT_PREREGISTRATION_SHA256
        ),
    }
    if any(identity.get(name) != expected_value for name, expected_value in expected.items()):
        _fail("package identity differs from current Genome V2 P1.0 authority")
    source_commit = identity.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        _fail("package source commit is invalid")
    artifact = _hash(
        identity.get("cuda_artifact_sha256"), label="package artifact"
    )
    if identity.get("loaded_artifact_sha256") != artifact:
        _fail("package loaded artifact differs from its CUDA artifact")
    return identity, package_id, file_sha256(path)


def validate_execution_route_authority_v1(
    path: Path,
    *,
    label: str,
    artifact_sha256: str,
    required_obligations: frozenset[str] = EXECUTION_ROUTE_OBLIGATIONS_V1,
) -> tuple[dict[str, Any], str]:
    receipt = _read_json(path, label=f"{label} route receipt")
    exact_fields = {
        "schema_version",
        "authority_state",
        "execution_route_sha256",
        "provider_id",
        "profile_id",
        "covered_obligations",
        "qualification_plan_ids",
        "case_ids",
        "artifact_sha256",
        "checkpoint_results",
        "reporting_results",
        "exact_comparisons",
    }
    if set(receipt) != exact_fields:
        _fail(f"{label} route receipt fields differ from contract")
    route = _hash(
        receipt.get("execution_route_sha256"),
        label=f"{label} execution route",
    )
    comparisons = receipt.get("exact_comparisons")
    if (
        receipt.get("schema_version")
        != EXECUTION_ROUTE_QUALIFICATION_SCHEMA_VERSION
        or receipt.get("authority_state") != "route_qualified"
        or receipt.get("provider_id") != PROVIDER_ID
        or receipt.get("profile_id") != RUNTIME_PROFILE_ID
        or receipt.get("artifact_sha256") != artifact_sha256
        or set(receipt.get("covered_obligations", ()))
        != set(required_obligations)
        or not receipt.get("qualification_plan_ids")
        or not receipt.get("case_ids")
        or not isinstance(receipt.get("checkpoint_results"), Mapping)
        or receipt["checkpoint_results"].get("exact") is not True
        or not isinstance(receipt.get("reporting_results"), Mapping)
        or receipt["reporting_results"].get("exact") is not True
        or not isinstance(comparisons, list)
        or not comparisons
        or any(
            not isinstance(row, Mapping)
            or row.get("match") is not True
            or row.get("first_difference") is not None
            or row.get("execution_route_sha256") != route
            for row in comparisons
        )
    ):
        _fail(f"{label} route is stale, incomplete, or not exactly qualified")
    return receipt, file_sha256(path)


def _validated_decision(
    path: Path,
    *,
    label: str,
    schema_version: str,
    preregistration_sha256: str,
    accepted: set[str],
) -> tuple[dict[str, Any], str, str]:
    value = _read_json(path, label=label)
    supplied = _hash(
        value.get("decision_sha256"), label=f"{label} decision_sha256"
    )
    unsigned = dict(value)
    unsigned.pop("decision_sha256", None)
    if (
        value.get("schema_version") != schema_version
        or value.get("preregistration_sha256") != preregistration_sha256
        or value.get("decision") not in accepted
        or content_sha256(unsigned) != supplied
    ):
        _fail(f"{label} is not a current content-addressed terminal decision")
    return value, supplied, file_sha256(path)


def _gate_receipt(
    *,
    gate_id: str,
    decision: str,
    decision_sha256: str,
    authority: Mapping[str, Any],
    routes: Sequence[str],
) -> dict[str, Any]:
    value = {
        "schema_version": GATE_RECEIPT_SCHEMA_VERSION,
        "gate_id": gate_id,
        "decision": decision,
        "decision_sha256": decision_sha256,
        "source_commit": authority["source_commit"],
        "package_id_sha256": authority["package_id_sha256"],
        "package_receipt_sha256": authority["package_receipt_sha256"],
        "artifact_sha256": authority["artifact_sha256"],
        "execution_route_sha256s": list(routes),
    }
    return {**value, "receipt_sha256": content_sha256(value)}


def collect_completion_evidence_v1(
    *,
    package_receipt_path: Path,
    patch_route_receipt_path: Path,
    wdl_route_receipt_path: Path,
    c1_decision_path: Path,
    e1_authority_path: Path,
    e1_decision_path: Path,
    hereditary_capacity_decision_path: Path,
    environment_switch_decision_path: Path,
    capability_contract_path: Path,
) -> dict[str, Any]:
    """Strictly join all terminal inputs into one completion-gate document."""

    identity, package_id, package_receipt_sha = validate_package_authority_v1(
        package_receipt_path
    )
    artifact = identity["cuda_artifact_sha256"]
    patch_receipt, patch_receipt_sha = validate_execution_route_authority_v1(
        patch_route_receipt_path, label="Patch", artifact_sha256=artifact
    )
    wdl_receipt, wdl_receipt_sha = validate_execution_route_authority_v1(
        wdl_route_receipt_path,
        label="WDL",
        artifact_sha256=artifact,
        required_obligations=SINGLE_CELL_EXECUTION_ROUTE_OBLIGATIONS_V1,
    )
    patch_route = patch_receipt["execution_route_sha256"]
    wdl_route = wdl_receipt["execution_route_sha256"]
    if patch_route == wdl_route:
        _fail("Patch and WDL must have distinct qualified route identities")

    capability = _read_json(
        capability_contract_path, label="capability contract"
    )
    try:
        validated_capability = validate_genome_v2_p10_capability_contract_v1(
            capability
        )
    except ValueError as exc:
        raise CompletionCollectorError(
            "capability contract is not the current Genome V2 P1.0 contract"
        ) from exc
    if (
        validated_capability.contract_sha256
        != identity["capability_contract_sha256"]
    ):
        _fail("capability contract differs from package identity")
    capability_file_sha = file_sha256(capability_contract_path)

    c1, c1_sha, c1_file_sha = _validated_decision(
        c1_decision_path,
        label="C1 decision",
        schema_version="genome_v2_p10_c1_decision_v2",
        preregistration_sha256=C1_PREREGISTRATION_SHA256,
        accepted={"C1-A"},
    )
    e1, e1_sha, e1_file_sha = _validated_decision(
        e1_decision_path,
        label="E1 decision",
        schema_version="genome_v2_p10_e1_ecological_gate_v1",
        preregistration_sha256=E1_PREREGISTRATION_SHA256,
        accepted={E1_DECISION_A, E1_DECISION_B},
    )
    hereditary, hereditary_sha, hereditary_file_sha = _validated_decision(
        hereditary_capacity_decision_path,
        label="hereditary-capacity decision",
        schema_version=HEREDITARY_SCHEMA_VERSION,
        preregistration_sha256=HEREDITARY_PREREGISTRATION_SHA256,
        accepted={HEREDITARY_DECISION_A},
    )
    environment, environment_sha, environment_file_sha = _validated_decision(
        environment_switch_decision_path,
        label="environment-switch decision",
        schema_version=ENVIRONMENT_SCHEMA_VERSION,
        preregistration_sha256=ENVIRONMENT_PREREGISTRATION_SHA256,
        accepted={ENVIRONMENT_DECISION_A},
    )

    e1_authority_payload = _read_json(
        e1_authority_path, label="E1 authority"
    )
    e1_authority_file_sha = file_sha256(e1_authority_path)
    try:
        e1_authority = authority_from_payload_v1(e1_authority_payload)
    except ValueError as exc:
        raise CompletionCollectorError("E1 authority is malformed") from exc
    defects = _authority_defects(e1_authority)
    if defects:
        _fail("E1 authority is invalid: " + ", ".join(defects))
    expected_common = {
        "provider_id": PROVIDER_ID,
        "runtime_profile_id": RUNTIME_PROFILE_ID,
        "source_commit": identity["source_commit"],
        "package_id_sha256": package_id,
        "package_receipt_sha256": package_receipt_sha,
        "artifact_sha256": artifact,
        "c1_decision_sha256": c1_sha,
        "c1_decision": c1["decision"],
    }
    if any(
        e1_authority_payload.get(name) != expected
        for name, expected in expected_common.items()
    ):
        _fail("E1 authority is mixed with another package or C1 decision")
    expected_campaign_routes = {
        CAMPAIGNS[0]: patch_route,
        CAMPAIGNS[1]: patch_route,
        CAMPAIGNS[2]: wdl_route,
    }
    if e1_authority_payload.get("execution_route_sha256s") != expected_campaign_routes:
        _fail("E1 authority does not bind the exact Patch/WDL route assignment")

    completion_authority = {
        "provider_id": PROVIDER_ID,
        "runtime_profile_id": RUNTIME_PROFILE_ID,
        "source_commit": identity["source_commit"],
        "package_id_sha256": package_id,
        "package_receipt_sha256": package_receipt_sha,
        "artifact_sha256": artifact,
        "execution_route_sha256s": {
            "patch": patch_route,
            "wdl": wdl_route,
        },
        "c1_decision_sha256": c1_sha,
        "e1_decision_sha256": e1_sha,
    }
    terminal_authorities = (
        (
            "hereditary-capacity",
            validate_hereditary_authority_v1(hereditary.get("authority")),
        ),
        (
            "environment-switch",
            validate_environment_authority_v1(environment.get("authority")),
        ),
    )
    for label, authority in terminal_authorities:
        expected = {
            "provider_id": PROVIDER_ID,
            "runtime_profile_id": RUNTIME_PROFILE_ID,
            "source_commit": identity["source_commit"],
            "package_id_sha256": package_id,
            "package_receipt_sha256": package_receipt_sha,
            "artifact_sha256": artifact,
        }
        if any(authority.get(name) != value for name, value in expected.items()):
            _fail(f"{label} authority is mixed with another package")
        if authority["execution_route_sha256"] not in {patch_route, wdl_route}:
            _fail(f"{label} authority uses an unqualified route")

    technical_payload = {
        "schema_version": "genome_v2_p10_technical_admission_v1",
        "decision": "TECHNICAL-A",
        "package_id_sha256": package_id,
        "package_receipt_sha256": package_receipt_sha,
        "capability_contract_sha256": (
            validated_capability.contract_sha256
        ),
        "capability_contract_file_sha256": capability_file_sha,
        "route_receipt_sha256s": {
            "patch": patch_receipt_sha,
            "wdl": wdl_receipt_sha,
        },
        "e1_authority_sha256": e1_authority.authority_sha256(),
        "e1_authority_file_sha256": e1_authority_file_sha,
        "execution_route_sha256s": {
            "patch": patch_route,
            "wdl": wdl_route,
        },
    }
    technical_sha = content_sha256(technical_payload)
    routes = [patch_route, wdl_route]
    gates = {
        "technical": _gate_receipt(
            gate_id="technical",
            decision="TECHNICAL-A",
            decision_sha256=technical_sha,
            authority=completion_authority,
            routes=routes,
        ),
        "c1": _gate_receipt(
            gate_id="c1",
            decision=c1["decision"],
            decision_sha256=c1_sha,
            authority=completion_authority,
            routes=[patch_route],
        ),
        "e1": _gate_receipt(
            gate_id="e1",
            decision=e1["decision"],
            decision_sha256=e1_sha,
            authority=completion_authority,
            routes=routes,
        ),
        "hereditary_capacity": _gate_receipt(
            gate_id="hereditary_capacity",
            decision=hereditary["decision"],
            decision_sha256=hereditary_sha,
            authority=completion_authority,
            routes=[hereditary["authority"]["execution_route_sha256"]],
        ),
        "environment_switch": _gate_receipt(
            gate_id="environment_switch",
            decision=environment["decision"],
            decision_sha256=environment_sha,
            authority=completion_authority,
            routes=[environment["authority"]["execution_route_sha256"]],
        ),
    }
    evidence_hashes = {
        "GV2-A": technical_sha,
        "GV2-B": hereditary_file_sha,
        "GV2-C": c1_file_sha,
        "GV2-D": c1_file_sha,
        "GV2-E": content_sha256(
            {"c1": c1_file_sha, "environment": environment_file_sha}
        ),
        "GV2-F": c1_file_sha,
        "GV2-G": environment_file_sha,
        "GV2-H": content_sha256(
            {
                "c1": c1_file_sha,
                "e1_authority": e1_authority_file_sha,
                "e1": e1_file_sha,
            }
        ),
    }
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "authority": completion_authority,
        "gate_receipts": gates,
        "work_packages": [
            {
                "work_package": package,
                "mandatory_capabilities": list(PACKAGE_CAPABILITIES[package]),
                "status": "implemented_and_qualified",
                "evidence_sha256": evidence_hashes[package],
                "disposition": (
                    "Exact current package, qualified routes, and terminal "
                    "gate evidence are content-addressed and joined."
                ),
            }
            for package in WORK_PACKAGES
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-receipt", type=Path, required=True)
    parser.add_argument("--patch-route-receipt", type=Path, required=True)
    parser.add_argument("--wdl-route-receipt", type=Path, required=True)
    parser.add_argument("--c1-decision", type=Path, required=True)
    parser.add_argument("--e1-authority", type=Path, required=True)
    parser.add_argument("--e1-decision", type=Path, required=True)
    parser.add_argument(
        "--hereditary-capacity-decision", type=Path, required=True
    )
    parser.add_argument("--environment-switch-decision", type=Path, required=True)
    parser.add_argument("--capability-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        print("error: output already exists", file=sys.stderr)
        return 2
    try:
        evidence = collect_completion_evidence_v1(
            package_receipt_path=args.package_receipt,
            patch_route_receipt_path=args.patch_route_receipt,
            wdl_route_receipt_path=args.wdl_route_receipt,
            c1_decision_path=args.c1_decision,
            e1_authority_path=args.e1_authority,
            e1_decision_path=args.e1_decision,
            hereditary_capacity_decision_path=(
                args.hereditary_capacity_decision
            ),
            environment_switch_decision_path=args.environment_switch_decision,
            capability_contract_path=args.capability_contract,
        )
        payload = canonical_json_bytes(evidence)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(payload)
    except (CompletionCollectorError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "schema_version": COLLECTOR_SCHEMA_VERSION,
                "output": str(output),
                "output_sha256": hashlib.sha256(payload).hexdigest(),
                "evidence_sha256": content_sha256(evidence),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
