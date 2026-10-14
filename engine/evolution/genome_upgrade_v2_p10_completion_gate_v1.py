"""Terminal Genome V2 P1.0 A-H completeness and baseline-lock reducer."""

from __future__ import annotations

import csv
import io
from typing import Any, Final, Mapping, Sequence

from engine.authoritative_state.schema import content_sha256
from .genome_upgrade_v2_capability_contract_v1 import (
    GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256,
    MANDATORY_P10_CAPABILITIES,
)
from .genome_upgrade_v2_environment_switch_gate_v1 import (
    DECISION_A as ENVIRONMENT_DECISION_A,
)
from .genome_upgrade_v2_hereditary_capacity_gate_v1 import (
    DECISION_A as HEREDITARY_DECISION_A,
)


SCHEMA_VERSION: Final[str] = "genome_v2_p10_completion_gate_v1"
EVIDENCE_SCHEMA_VERSION: Final[str] = "genome_v2_p10_completion_evidence_v1"
PROVIDER_ID: Final[str] = "eiel-production-provider-v1"
RUNTIME_PROFILE_ID: Final[str] = "eiel-production-runtime-v1"
ROUTE_IDS: Final[tuple[str, str]] = ("patch", "wdl")
WORK_PACKAGES: Final[tuple[str, ...]] = (
    "GV2-A",
    "GV2-B",
    "GV2-C",
    "GV2-D",
    "GV2-E",
    "GV2-F",
    "GV2-G",
    "GV2-H",
)
ALLOWED_STATUSES: Final[tuple[str, ...]] = (
    "implemented_and_qualified",
    "implemented_not_qualified",
    "partially_implemented",
    "not_implemented",
    "deferred_by_plan",
    "superseded_with_equivalent_evidence",
    "not_applicable_with_reason",
)
PACKAGE_CAPABILITIES: Final[dict[str, tuple[str, ...]]] = {
    "GV2-A": (
        "stable_diploid_inheritance",
        "reference_native_cuda_parity",
        "checkpoint_and_chronicle_authority",
    ),
    "GV2-B": ("broader_hereditary_traits_with_real_costs",),
    "GV2-C": ("gene_product_output",),
    "GV2-D": ("polygenic_phenotype_mapping",),
    "GV2-E": ("bounded_regulatory_execution",),
    "GV2-F": (
        "emergent_dominance_fixtures",
        "modifier_gene_interaction",
    ),
    "GV2-G": ("environmental_expression",),
    "GV2-H": (
        "pleiotropic_tradeoffs",
        "ecological_usefulness_under_preregistered_campaigns",
    ),
}
DEFERRED_SCOPE: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "evolvable_regulatory_topology",
        "status": "deferred_by_plan",
        "rationale": "P1.0 qualifies the fixed bounded regulatory network first.",
        "future_prerequisite": "new versioned topology contract and package qualification",
    },
    {
        "id": "evolvable_linkage_boundaries",
        "status": "deferred_by_plan",
        "rationale": "P1.0 retains the fixed six-block linkage map.",
        "future_prerequisite": "dedicated fixed-linkage science decision",
    },
    {
        "id": "evolvable_recombination_control",
        "status": "deferred_by_plan",
        "rationale": "Crossover modifiers and suppression require a later package.",
        "future_prerequisite": "dedicated recombination architecture and causal gate",
    },
    {
        "id": "structural_diploidy",
        "status": "deferred_by_plan",
        "rationale": "Graph homolog identity and crossover are outside the fixed envelope.",
        "future_prerequisite": "separate structural genetics architecture review",
    },
)
GATE_IDS: Final[tuple[str, ...]] = (
    "technical",
    "c1",
    "e1",
    "hereditary_capacity",
    "environment_switch",
)
DECISION_A: Final[str] = "P10-A"
DECISION_B: Final[str] = "P10-B"
DECISION_C: Final[str] = "P10-C"
DECISION_D: Final[str] = "P10-D"
OUTPUT_NAMES: Final[dict[str, str]] = {
    "matrix": "genome_v2_p10_ah_completeness_matrix.csv",
    "audit": "genome_v2_p10_ah_completeness_audit.json",
    "gap_report": "genome_v2_p10_ah_gap_report.md",
    "deferred": "genome_v2_p10_deferred_scope_register.json",
    "baseline_lock": "genome_v2_p10_technical_baseline_lock.json",
}


def preregistration_payload_v1() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "programme": "GENOME-UPGRADE-V2/P1.0",
        "work_packages": {
            package: list(PACKAGE_CAPABILITIES[package])
            for package in WORK_PACKAGES
        },
        "mandatory_capabilities": list(MANDATORY_P10_CAPABILITIES),
        "required_gate_receipts": list(GATE_IDS),
        "required_routes": list(ROUTE_IDS),
        "allowed_work_package_statuses": list(ALLOWED_STATUSES),
        "mandatory_pleiotropy_rule": (
            "GV2-H pleiotropic_tradeoffs must be implemented_and_qualified; "
            "it cannot be deferred"
        ),
        "deferred_scope": [dict(row) for row in DEFERRED_SCOPE],
        "terminal_decisions": {
            DECISION_A: "all_mandatory_capabilities_and_e1_a",
            DECISION_B: "all_mandatory_capabilities_and_complete_negative_e1_b",
            DECISION_C: "valid_but_incomplete_programme",
            DECISION_D: "mixed_identity_or_integrity_failure",
        },
    }


PREREGISTRATION_SHA256: Final[str] = content_sha256(
    preregistration_payload_v1()
)


def _hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_completion_authority_v1(value: Any) -> dict[str, Any]:
    fields = {
        "provider_id",
        "runtime_profile_id",
        "source_commit",
        "package_id_sha256",
        "package_receipt_sha256",
        "artifact_sha256",
        "execution_route_sha256s",
        "c1_decision_sha256",
        "e1_decision_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("P1.0 completion authority fields differ from contract")
    routes = value.get("execution_route_sha256s")
    if (
        value.get("provider_id") != PROVIDER_ID
        or value.get("runtime_profile_id") != RUNTIME_PROFILE_ID
        or not _commit(value.get("source_commit"))
        or not isinstance(routes, Mapping)
        or set(routes) != set(ROUTE_IDS)
        or any(not _hash(item) for item in routes.values())
        or any(
            not _hash(value.get(name))
            for name in (
                "package_id_sha256",
                "package_receipt_sha256",
                "artifact_sha256",
                "c1_decision_sha256",
                "e1_decision_sha256",
            )
        )
    ):
        raise ValueError("P1.0 completion authority is invalid")
    return {
        **{name: value[name] for name in fields if name != "execution_route_sha256s"},
        "execution_route_sha256s": dict(sorted(routes.items())),
    }


def _validate_gate_receipt(
    value: Any,
    *,
    gate_id: str,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "gate_id",
        "decision",
        "decision_sha256",
        "source_commit",
        "package_id_sha256",
        "package_receipt_sha256",
        "artifact_sha256",
        "execution_route_sha256s",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{gate_id}: gate receipt fields differ from contract")
    unsigned = dict(value)
    supplied = unsigned.pop("receipt_sha256", None)
    routes = value.get("execution_route_sha256s")
    if (
        value.get("schema_version") != "genome_v2_p10_completion_gate_receipt_v1"
        or value.get("gate_id") != gate_id
        or not _hash(value.get("decision_sha256"))
        or supplied != content_sha256(unsigned)
        or value.get("source_commit") != authority["source_commit"]
        or value.get("package_id_sha256") != authority["package_id_sha256"]
        or value.get("package_receipt_sha256")
        != authority["package_receipt_sha256"]
        or value.get("artifact_sha256") != authority["artifact_sha256"]
        or isinstance(routes, (str, bytes))
        or not isinstance(routes, Sequence)
        or not routes
        or len(set(routes)) != len(routes)
        or any(route not in authority["execution_route_sha256s"].values() for route in routes)
    ):
        raise ValueError(f"{gate_id}: mixed or invalid gate authority")
    if gate_id == "c1" and value["decision_sha256"] != authority["c1_decision_sha256"]:
        raise ValueError("c1: decision identity mismatch")
    if gate_id == "e1" and value["decision_sha256"] != authority["e1_decision_sha256"]:
        raise ValueError("e1: decision identity mismatch")
    if gate_id in {"technical", "e1"} and set(routes) != set(
        authority["execution_route_sha256s"].values()
    ):
        raise ValueError(f"{gate_id}: complete Patch/WDL route identity is required")
    return dict(value)


def _matrix_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = (
        "work_package",
        "mandatory_capabilities",
        "status",
        "evidence_sha256",
        "disposition",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                **{name: row[name] for name in fields},
                "mandatory_capabilities": ";".join(row["mandatory_capabilities"]),
            }
        )
    return stream.getvalue()


def _gap_report(
    *,
    decision: str,
    rows: Sequence[Mapping[str, Any]],
    reasons: Sequence[str],
) -> str:
    gaps = [
        row
        for row in rows
        if row["status"] != "implemented_and_qualified"
    ]
    lines = [
        "# Genome V2 P1.0 A-H Gap Report",
        "",
        "## Decision",
        "",
        f"`{decision}`",
        "",
    ]
    if reasons:
        lines.extend(["## Reasons", "", *[f"- {reason}" for reason in reasons], ""])
    lines.extend(["## Work Packages", ""])
    lines.extend(
        f"- `{row['work_package']}`: `{row['status']}` - {row['disposition']}"
        for row in rows
    )
    lines.extend(["", "## Remaining Gaps", ""])
    lines.extend(
        ["- None. All mandatory P1.0 capabilities are qualified."]
        if not gaps
        else [
            f"- `{row['work_package']}`: {row['disposition']}"
            for row in gaps
        ]
    )
    lines.extend(
        [
            "",
            "## Claims Boundary",
            "",
            "This reducer establishes only the bounded P1.0 decision encoded by "
            "the supplied evidence. It makes no intelligence, cognition, "
            "consciousness, emergence, speciation, or open-ended evolution claim.",
            "",
        ]
    )
    return "\n".join(lines)


def reduce_p10_completion_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the five immutable P1.0 completion artifacts."""

    if not isinstance(value, Mapping):
        raise ValueError("P1.0 completion evidence must be an object")
    expected_fields = {
        "schema_version",
        "preregistration_sha256",
        "authority",
        "gate_receipts",
        "work_packages",
    }
    if (
        set(value) != expected_fields
        or value.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or value.get("preregistration_sha256") != PREREGISTRATION_SHA256
    ):
        raise ValueError("P1.0 completion evidence schema differs from contract")
    authority = validate_completion_authority_v1(value.get("authority"))
    raw_gates = value.get("gate_receipts")
    if not isinstance(raw_gates, Mapping) or set(raw_gates) != set(GATE_IDS):
        raise ValueError("P1.0 gate receipt set is incomplete")
    gates = {
        gate_id: _validate_gate_receipt(
            raw_gates[gate_id], gate_id=gate_id, authority=authority
        )
        for gate_id in GATE_IDS
    }
    raw_packages = value.get("work_packages")
    if (
        isinstance(raw_packages, (str, bytes))
        or not isinstance(raw_packages, Sequence)
    ):
        raise ValueError("P1.0 work packages must be a sequence")
    indexed: dict[str, dict[str, Any]] = {}
    for raw in raw_packages:
        fields = {
            "work_package",
            "mandatory_capabilities",
            "status",
            "evidence_sha256",
            "disposition",
        }
        if not isinstance(raw, Mapping) or set(raw) != fields:
            raise ValueError("P1.0 work-package row fields differ from contract")
        package = raw.get("work_package")
        capabilities = raw.get("mandatory_capabilities")
        if (
            package not in WORK_PACKAGES
            or package in indexed
            or capabilities != list(PACKAGE_CAPABILITIES[package])
            or raw.get("status") not in ALLOWED_STATUSES
            or not _hash(raw.get("evidence_sha256"))
            or not isinstance(raw.get("disposition"), str)
            or not raw["disposition"]
        ):
            raise ValueError(f"invalid P1.0 work-package row: {package!r}")
        indexed[package] = dict(raw)
    if set(indexed) != set(WORK_PACKAGES):
        raise ValueError("P1.0 work-package matrix is incomplete")
    rows = [indexed[package] for package in WORK_PACKAGES]
    reasons: list[str] = []
    expected_decisions = {
        "technical": {"TECHNICAL-A"},
        "c1": {"C1-A"},
        "e1": {"E1-A", "E1-B"},
        "hereditary_capacity": {HEREDITARY_DECISION_A},
        "environment_switch": {ENVIRONMENT_DECISION_A},
    }
    for gate_id, accepted in expected_decisions.items():
        if gates[gate_id]["decision"] not in accepted:
            reasons.append(f"{gate_id}_decision_not_terminal")
    for row in rows:
        if row["status"] != "implemented_and_qualified":
            reasons.append(f"{row['work_package']}_not_qualified")
    if (
        indexed["GV2-H"]["status"] != "implemented_and_qualified"
        or "pleiotropic_tradeoffs"
        not in indexed["GV2-H"]["mandatory_capabilities"]
    ):
        reasons.append("bounded_pleiotropy_is_mandatory")
    if reasons:
        decision = DECISION_C
    elif gates["e1"]["decision"] == "E1-B":
        decision = DECISION_B
    else:
        decision = DECISION_A
    matrix = _matrix_csv(rows)
    deferred = {
        "schema_version": "genome_v2_p10_deferred_scope_register_v1",
        "source_commit": authority["source_commit"],
        "package_id_sha256": authority["package_id_sha256"],
        "items": [dict(row) for row in DEFERRED_SCOPE],
    }
    deferred["register_sha256"] = content_sha256(deferred)
    baseline = {
        "schema_version": "genome_v2_p10_technical_baseline_lock_v1",
        "authority": authority,
        "capability_contract_sha256": (
            GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256
        ),
        "gate_receipt_sha256s": {
            gate_id: gates[gate_id]["receipt_sha256"] for gate_id in GATE_IDS
        },
        "work_package_evidence_sha256s": {
            row["work_package"]: row["evidence_sha256"] for row in rows
        },
        "decision": decision,
    }
    baseline["baseline_lock_sha256"] = content_sha256(baseline)
    audit = {
        "schema_version": "genome_v2_p10_ah_completeness_audit_v1",
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "authority": authority,
        "decision": decision,
        "reasons": sorted(set(reasons)),
        "capability_contract_sha256": (
            GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256
        ),
        "matrix_sha256": content_sha256(matrix),
        "deferred_scope_register_sha256": deferred["register_sha256"],
        "technical_baseline_lock_sha256": baseline["baseline_lock_sha256"],
        "work_packages": {
            row["work_package"]: {
                "status": row["status"],
                "evidence_sha256": row["evidence_sha256"],
                "mandatory_capabilities": list(row["mandatory_capabilities"]),
            }
            for row in rows
        },
        "claims": {
            "p10_complete": decision in {DECISION_A, DECISION_B},
            "ecological_usefulness_positive": decision == DECISION_A,
            "complete_negative_ecological_result": decision == DECISION_B,
        },
    }
    audit["audit_sha256"] = content_sha256(audit)
    return {
        OUTPUT_NAMES["matrix"]: matrix,
        OUTPUT_NAMES["audit"]: audit,
        OUTPUT_NAMES["gap_report"]: _gap_report(
            decision=decision, rows=rows, reasons=reasons
        ),
        OUTPUT_NAMES["deferred"]: deferred,
        OUTPUT_NAMES["baseline_lock"]: baseline,
    }


__all__ = [
    "ALLOWED_STATUSES",
    "DECISION_A",
    "DECISION_B",
    "DECISION_C",
    "DECISION_D",
    "DEFERRED_SCOPE",
    "EVIDENCE_SCHEMA_VERSION",
    "GATE_IDS",
    "OUTPUT_NAMES",
    "PACKAGE_CAPABILITIES",
    "PREREGISTRATION_SHA256",
    "ROUTE_IDS",
    "SCHEMA_VERSION",
    "WORK_PACKAGES",
    "preregistration_payload_v1",
    "reduce_p10_completion_v1",
    "validate_completion_authority_v1",
]
