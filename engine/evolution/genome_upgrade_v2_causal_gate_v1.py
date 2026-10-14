"""Content-addressed Genome V2 P1.0 C1 causal decision gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Mapping, Sequence

from engine.authoritative_state.schema import content_sha256
from .genome_upgrade_v2 import (
    CONTEXT_COUNT,
    CONTRACT_SHA256 as MECHANIC_CONTRACT_SHA256,
    PARAMETER_COUNT,
    PHENOTYPE_COUNT,
    PRODUCT_COUNT,
    Q10_SCALE,
)


SCHEMA_VERSION: Final[str] = "genome_v2_p10_c1_causal_gate_v2"
CASE_SCHEMA_VERSION: Final[str] = "genome_v2_p10_c1_matched_case_v2"
PAIR_SCHEMA_VERSION: Final[str] = "genome_v2_p10_c1_pair_evidence_v2"
BACKEND_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "genome_v2_p10_c1_backend_case_receipt_v1"
)
CHECKPOINT_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "genome_v2_p10_c1_checkpoint_receipt_v1"
)
SUPPORT_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "genome_v2_p10_c1_family_support_receipt_v1"
)

PROVIDER_ID: Final[str] = "eiel-production-provider-v1"
RUNTIME_PROFILE_ID: Final[str] = "eiel-production-runtime-v1"
REQUIRED_BACKENDS: Final[tuple[str, ...]] = ("reference", "native", "cuda")
REQUIRED_REPEAT_INDICES: Final[tuple[int, ...]] = (0, 1)
MANDATORY_FAMILIES: Final[tuple[str, ...]] = (
    "broader_hereditary_traits",
    "dosage_saturation",
    "polygenic_compensation",
    "regulatory_activation_inhibition",
    "modifier_dependent_expression",
    "environmental_developmental_switching",
    "pleiotropic_tradeoffs",
)
REQUIRED_DIMENSIONS: Final[tuple[str, ...]] = (
    "genotype",
    "context",
    "product",
    "regulatory",
    "phenotype",
    "runtime_consequence",
    "cost",
)
DIMENSION_LENGTHS: Final[dict[str, int]] = {
    "genotype": 2 * PRODUCT_COUNT * PARAMETER_COUNT + 2 * PRODUCT_COUNT,
    "context": CONTEXT_COUNT,
    "product": PRODUCT_COUNT + 2,
    "regulatory": 4 + PRODUCT_COUNT + 4,
    "phenotype": PHENOTYPE_COUNT,
    "runtime_consequence": 12,
    "cost": 2,
}

DECISION_A: Final[str] = "C1-A"
DECISION_B: Final[str] = "C1-B"
DECISION_C: Final[str] = "C1-C"
DECISION_D: Final[str] = "C1-D"

_NEUTRAL_CONTEXT: Final[tuple[int, ...]] = (0,) * CONTEXT_COUNT
_GENOTYPE_ALLELE_WIDTH: Final[int] = (
    2 * PRODUCT_COUNT * PARAMETER_COUNT
)
_PRODUCT_ACTIVE_COUNT_INDEX: Final[int] = PRODUCT_COUNT
_PRODUCT_SATURATED_COUNT_INDEX: Final[int] = PRODUCT_COUNT + 1
_REGULATOR_NODE_OFFSET: Final[int] = 0
_REGULATORY_COST_INDEX: Final[int] = 4 + PRODUCT_COUNT
_ACTIVE_REGULATOR_COUNT_INDEX: Final[int] = _REGULATORY_COST_INDEX + 1
_ACTIVE_EDGE_COUNT_INDEX: Final[int] = _REGULATORY_COST_INDEX + 2
_DEVELOPMENTAL_STAGE_INDEX: Final[int] = _REGULATORY_COST_INDEX + 3


def _allele_update(
    homolog: int,
    product: int,
    parameter: int,
    value: int,
) -> tuple[str, int, int, int, int]:
    return ("allele", homolog, product, parameter, value)


def _modifier_update(
    homolog: int,
    product: int,
    value: int,
) -> tuple[str, int, int, int, int]:
    return ("modifier", homolog, product, 0, value)


def _arm(
    *,
    updates: Sequence[tuple[str, int, int, int, int]] = (),
    context: Sequence[int] = _NEUTRAL_CONTEXT,
) -> dict[str, Any]:
    return {
        "genotype_updates": [list(update) for update in updates],
        "context_q10": list(context),
    }


def _fixture_rows_v1() -> tuple[dict[str, Any], ...]:
    juvenile = list(_NEUTRAL_CONTEXT)
    juvenile[6] = Q10_SCALE
    hazard = list(_NEUTRAL_CONTEXT)
    hazard[5] = Q10_SCALE
    return (
        {
            "family_id": "broader_hereditary_traits",
            "pair_id": "broader-hereditary-traits-pair-v1",
            "seed": 1701,
            "control": _arm(),
            "treatment": _arm(
                updates=(_allele_update(0, 11, 0, 768),),
            ),
            "predicate_id": "broader_runtime_projection_v1",
            "predicate_parameters": {
                "minimum_changed_phenotypes": 4,
                "minimum_changed_runtime_fields": 4,
            },
        },
        {
            "family_id": "dosage_saturation",
            "pair_id": "dosage-saturation-pair-v1",
            "seed": 1702,
            "control": _arm(
                updates=(_allele_update(0, 7, 0, 700),),
            ),
            "treatment": _arm(
                updates=(
                    _allele_update(0, 7, 0, 700),
                    _allele_update(1, 7, 0, 700),
                ),
            ),
            "predicate_id": "diploid_dosage_into_saturation_v1",
            "predicate_parameters": {
                "product_index": 7,
                "saturation_q10": Q10_SCALE,
            },
        },
        {
            "family_id": "polygenic_compensation",
            "pair_id": "polygenic-compensation-pair-v1",
            "seed": 1703,
            "control": _arm(
                updates=(_allele_update(0, 0, 0, 512),),
            ),
            "treatment": _arm(
                updates=(_allele_update(0, 1, 0, 640),),
            ),
            "predicate_id": "distinct_routes_equal_target_phenotype_v1",
            "predicate_parameters": {
                "target_phenotype_index": 0,
                "control_product_index": 0,
                "treatment_product_index": 1,
            },
        },
        {
            "family_id": "regulatory_activation_inhibition",
            "pair_id": "regulatory-activation-inhibition-pair-v1",
            "seed": 1704,
            "control": _arm(
                updates=(_allele_update(0, 10, 0, 512),),
            ),
            "treatment": _arm(
                updates=(_allele_update(0, 10, 0, 512),),
                context=hazard,
            ),
            "predicate_id": "hazard_activation_inhibition_v1",
            "predicate_parameters": {
                "regulator_index": 2,
                "activated_product_index": 9,
                "inhibited_product_index": 10,
            },
        },
        {
            "family_id": "modifier_dependent_expression",
            "pair_id": "modifier-dependent-expression-pair-v1",
            "seed": 1705,
            "control": _arm(
                updates=(_allele_update(0, 2, 0, 512),),
            ),
            "treatment": _arm(
                updates=(
                    _allele_update(0, 2, 0, 512),
                    _modifier_update(0, 2, Q10_SCALE),
                ),
            ),
            "predicate_id": "stability_modifier_expression_v1",
            "predicate_parameters": {"product_index": 2},
        },
        {
            "family_id": "environmental_developmental_switching",
            "pair_id": "environmental-developmental-switching-pair-v1",
            "seed": 1706,
            "control": _arm(
                updates=(_allele_update(0, 10, 0, 512),),
            ),
            "treatment": _arm(
                updates=(_allele_update(0, 10, 0, 512),),
                context=juvenile,
            ),
            "predicate_id": "mature_to_juvenile_switch_v1",
            "predicate_parameters": {
                "regulator_index": 3,
                "product_index": 10,
                "mature_stage_code": 0,
                "juvenile_stage_code": 1,
            },
        },
        {
            "family_id": "pleiotropic_tradeoffs",
            "pair_id": "pleiotropic-tradeoffs-pair-v1",
            "seed": 1707,
            "control": _arm(),
            "treatment": _arm(
                updates=(_allele_update(0, 0, 0, Q10_SCALE),),
            ),
            "predicate_id": "opposing_multitrait_effects_v1",
            "predicate_parameters": {
                "minimum_changed_phenotypes": 2,
                "require_opposing_phenotype_signs": True,
                "require_opposing_runtime_deltas": True,
            },
        },
    )


def preregistration_payload_v1() -> dict[str, Any]:
    """Return the exact unsigned C1 preregistration contract."""

    fixtures = []
    for row in _fixture_rows_v1():
        fixture = {
            **row,
            "required_case_arms": ["control", "treatment"],
            "required_backends": list(REQUIRED_BACKENDS),
            "required_repeat_indices": list(REQUIRED_REPEAT_INDICES),
            "required_dimensions": list(REQUIRED_DIMENSIONS),
        }
        fixtures.append(
            {
                **fixture,
                "fixture_sha256": content_sha256(fixture),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "programme": "GENOME-UPGRADE-V2/P1.0",
        "gate": "C1",
        "provider_id": PROVIDER_ID,
        "runtime_profile_id": RUNTIME_PROFILE_ID,
        "mechanic_contract_sha256": MECHANIC_CONTRACT_SHA256,
        "fixtures": fixtures,
        "dimension_contract": {
            "order": list(REQUIRED_DIMENSIONS),
            "lengths": dict(DIMENSION_LENGTHS),
            "values": "exact_finite_signed_integer_vectors_v1",
            "product_tail": [
                "active_product_count",
                "saturated_product_count",
            ],
            "regulatory_tail": [
                "regulatory_cost_q10",
                "active_regulator_count",
                "active_regulatory_edge_count",
                "developmental_stage_code",
            ],
        },
        "identity_contract": {
            "package_id_required": True,
            "package_receipt_required": True,
            "artifact_required": True,
            "execution_route_required": True,
            "shared_pair_checkpoint_required": True,
            "source_commit_required": True,
        },
        "integrity_contract": {
            "backend_parity": "reference_native_cuda_exact",
            "repeat_parity": "two_exact_repeats_per_backend_and_arm",
            "checkpoint": "two_exact_restores_of_shared_pair_checkpoint",
            "fallback": "absent_for_every_backend_receipt",
            "jit": "absent_for_every_backend_receipt",
            "confounds": (
                "genotype_and_context_must_equal_frozen_arm_inputs"
            ),
            "support": "computed_only_by_frozen_family_predicate",
        },
        "decision_map": {
            DECISION_A: "all_seven_frozen_family_predicates_supported",
            DECISION_B: "complete_valid_evidence_with_negative_predicate",
            DECISION_C: "missing_or_ambiguous_family_evidence",
            DECISION_D: "technical_identity_or_integrity_failure",
        },
        "claims_boundary": (
            "bounded fixed-genotype causal fixtures only; no ecological, "
            "intelligence, emergence, consciousness, or broad cognition claim"
        ),
    }


PREREGISTRATION_SHA256: Final[str] = content_sha256(
    preregistration_payload_v1()
)


def _valid_hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_dimensions(
    dimensions: Mapping[str, Sequence[int]],
) -> dict[str, list[int]]:
    return {
        name: [int(value) for value in dimensions[name]]
        for name in REQUIRED_DIMENSIONS
    }


@dataclass(frozen=True, slots=True)
class AuthorityBindingV1:
    source_commit: str
    package_id_sha256: str
    package_receipt_sha256: str
    artifact_sha256: str
    execution_route_sha256: str
    checkpoint_sha256: str
    preregistration_sha256: str = PREREGISTRATION_SHA256
    provider_id: str = PROVIDER_ID
    runtime_profile_id: str = RUNTIME_PROFILE_ID

    def payload(self) -> dict[str, str]:
        return {
            "provider_id": self.provider_id,
            "runtime_profile_id": self.runtime_profile_id,
            "source_commit": self.source_commit,
            "package_id_sha256": self.package_id_sha256,
            "package_receipt_sha256": self.package_receipt_sha256,
            "artifact_sha256": self.artifact_sha256,
            "execution_route_sha256": self.execution_route_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "preregistration_sha256": self.preregistration_sha256,
        }


@dataclass(frozen=True, slots=True)
class BackendCaseReceiptV1:
    family_id: str
    pair_id: str
    case_id: str
    arm: str
    seed: int
    backend: str
    repeat_index: int
    authority: AuthorityBindingV1
    dimensions: Mapping[str, Sequence[int]]
    execution_receipt_sha256: str
    no_fallback: bool
    no_jit: bool

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": BACKEND_RECEIPT_SCHEMA_VERSION,
            "family_id": self.family_id,
            "pair_id": self.pair_id,
            "case_id": self.case_id,
            "arm": self.arm,
            "seed": self.seed,
            "backend": self.backend,
            "repeat_index": self.repeat_index,
            "authority": self.authority.payload(),
            "dimensions": _canonical_dimensions(self.dimensions),
            "execution_receipt_sha256": self.execution_receipt_sha256,
            "no_fallback": self.no_fallback,
            "no_jit": self.no_jit,
        }

    def receipt_sha256(self) -> str:
        return content_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class MatchedCaseRecordV1:
    family_id: str
    pair_id: str
    case_id: str
    arm: str
    seed: int
    receipts: tuple[BackendCaseReceiptV1, ...]

    def payload(self) -> dict[str, Any]:
        rows = sorted(
            (
                {
                    **receipt.payload(),
                    "receipt_sha256": receipt.receipt_sha256(),
                }
                for receipt in self.receipts
            ),
            key=lambda row: (str(row["backend"]), int(row["repeat_index"])),
        )
        value = {
            "schema_version": CASE_SCHEMA_VERSION,
            "family_id": self.family_id,
            "pair_id": self.pair_id,
            "case_id": self.case_id,
            "arm": self.arm,
            "seed": self.seed,
            "backend_receipts": rows,
        }
        return {**value, "case_sha256": content_sha256(value)}

    def case_sha256(self) -> str:
        return str(self.payload()["case_sha256"])


@dataclass(frozen=True, slots=True)
class CheckpointReceiptV1:
    family_id: str
    pair_id: str
    control_case_id: str
    treatment_case_id: str
    authority: AuthorityBindingV1
    restored_checkpoint_sha256s: tuple[str, str]
    execution_receipt_sha256: str
    exact_restore: bool

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": CHECKPOINT_RECEIPT_SCHEMA_VERSION,
            "family_id": self.family_id,
            "pair_id": self.pair_id,
            "control_case_id": self.control_case_id,
            "treatment_case_id": self.treatment_case_id,
            "authority": self.authority.payload(),
            "restored_checkpoint_sha256s": list(
                self.restored_checkpoint_sha256s
            ),
            "execution_receipt_sha256": self.execution_receipt_sha256,
            "exact_restore": self.exact_restore,
        }

    def receipt_sha256(self) -> str:
        return content_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class MatchedPairEvidenceV1:
    control: MatchedCaseRecordV1
    treatment: MatchedCaseRecordV1
    checkpoint: CheckpointReceiptV1
    fixture_sha256: str

    def payload(self) -> dict[str, Any]:
        value = {
            "schema_version": PAIR_SCHEMA_VERSION,
            "fixture_sha256": self.fixture_sha256,
            "control": self.control.payload(),
            "treatment": self.treatment.payload(),
            "checkpoint": {
                **self.checkpoint.payload(),
                "receipt_sha256": self.checkpoint.receipt_sha256(),
            },
        }
        return {**value, "pair_sha256": content_sha256(value)}

    def pair_sha256(self) -> str:
        return str(self.payload()["pair_sha256"])


def fixture_by_family_v1(family_id: str) -> dict[str, Any]:
    for fixture in preregistration_payload_v1()["fixtures"]:
        if fixture["family_id"] == family_id:
            return fixture
    raise ValueError(f"unknown C1 family: {family_id!r}")


def _neutral_genotype_vector() -> tuple[int, ...]:
    values: list[int] = []
    for _homolog in range(2):
        for _product in range(PRODUCT_COUNT):
            values.extend((0, Q10_SCALE, Q10_SCALE, 0))
    values.extend(0 for _ in range(2 * PRODUCT_COUNT))
    return tuple(values)


def fixture_input_dimensions_v1(
    family_id: str,
    arm: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the exact preregistered genotype and context input vectors."""

    if arm not in {"control", "treatment"}:
        raise ValueError("C1 arm must be control or treatment")
    fixture = fixture_by_family_v1(family_id)
    arm_spec = fixture[arm]
    genotype = list(_neutral_genotype_vector())
    for update in arm_spec["genotype_updates"]:
        kind, homolog, product, parameter, value = update
        if kind == "allele":
            index = (
                (int(homolog) * PRODUCT_COUNT + int(product))
                * PARAMETER_COUNT
                + int(parameter)
            )
        elif kind == "modifier":
            index = (
                _GENOTYPE_ALLELE_WIDTH
                + int(homolog) * PRODUCT_COUNT
                + int(product)
            )
        else:  # pragma: no cover - preregistration is module-owned
            raise ValueError("invalid frozen genotype update kind")
        genotype[index] = int(value)
    return tuple(genotype), tuple(int(v) for v in arm_spec["context_q10"])


def _authority_defects(authority: AuthorityBindingV1) -> list[str]:
    defects: list[str] = []
    if authority.provider_id != PROVIDER_ID:
        defects.append("provider_identity_mismatch")
    if authority.runtime_profile_id != RUNTIME_PROFILE_ID:
        defects.append("runtime_profile_identity_mismatch")
    if not _valid_commit(authority.source_commit):
        defects.append("invalid_source_commit")
    for name in (
        "package_id_sha256",
        "package_receipt_sha256",
        "artifact_sha256",
        "execution_route_sha256",
        "checkpoint_sha256",
        "preregistration_sha256",
    ):
        if not _valid_hash(getattr(authority, name)):
            defects.append(f"invalid_{name}")
    if authority.preregistration_sha256 != PREREGISTRATION_SHA256:
        defects.append("preregistration_drift")
    return defects


def _dimensions_defects(
    dimensions: Any,
    *,
    prefix: str,
) -> list[str]:
    if not isinstance(dimensions, Mapping):
        return [f"{prefix}:invalid_dimensions"]
    if set(dimensions) != set(REQUIRED_DIMENSIONS):
        return [f"{prefix}:dimension_contract_drift"]
    defects: list[str] = []
    for name in REQUIRED_DIMENSIONS:
        values = dimensions[name]
        if (
            isinstance(values, (str, bytes))
            or not isinstance(values, Sequence)
            or len(values) != DIMENSION_LENGTHS[name]
        ):
            defects.append(f"{prefix}:invalid_dimension_shape:{name}")
            continue
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in values
        ):
            defects.append(f"{prefix}:noninteger_dimension:{name}")
    return defects


def _case_state(
    case: MatchedCaseRecordV1,
    *,
    expected_arm: str,
    fixture: Mapping[str, Any],
) -> tuple[BackendCaseReceiptV1 | None, list[str]]:
    defects: list[str] = []
    expected_identity = {
        "family_id": fixture["family_id"],
        "pair_id": fixture["pair_id"],
        "arm": expected_arm,
        "seed": fixture["seed"],
    }
    for name, expected in expected_identity.items():
        if getattr(case, name) != expected:
            defects.append(f"{expected_arm}:case_identity_mismatch:{name}")
    if case.case_id != f"{case.pair_id}-{expected_arm}":
        defects.append(f"{expected_arm}:case_id_not_canonical")

    expected_keys = {
        (backend, repeat)
        for backend in REQUIRED_BACKENDS
        for repeat in REQUIRED_REPEAT_INDICES
    }
    by_key: dict[tuple[str, int], BackendCaseReceiptV1] = {}
    for index, receipt in enumerate(case.receipts):
        if not isinstance(receipt, BackendCaseReceiptV1):
            defects.append(f"{expected_arm}:receipt[{index}]:invalid_type")
            continue
        key = (receipt.backend, receipt.repeat_index)
        if key in by_key:
            defects.append(f"{expected_arm}:duplicate_backend_repeat:{key}")
        by_key[key] = receipt
        for name in ("family_id", "pair_id", "case_id", "arm", "seed"):
            if getattr(receipt, name) != getattr(case, name):
                defects.append(
                    f"{expected_arm}:{receipt.backend}:{name}_mismatch"
                )
        defects.extend(
            f"{expected_arm}:{receipt.backend}:{item}"
            for item in _authority_defects(receipt.authority)
        )
        defects.extend(
            _dimensions_defects(
                receipt.dimensions,
                prefix=(
                    f"{expected_arm}:{receipt.backend}:"
                    f"repeat{receipt.repeat_index}"
                ),
            )
        )
        if not _valid_hash(receipt.execution_receipt_sha256):
            defects.append(
                f"{expected_arm}:{receipt.backend}:"
                "invalid_execution_receipt_sha256"
            )
        if receipt.no_fallback is not True:
            defects.append(
                f"{expected_arm}:{receipt.backend}:fallback_present"
            )
        if receipt.no_jit is not True:
            defects.append(f"{expected_arm}:{receipt.backend}:jit_present")
    if set(by_key) != expected_keys:
        defects.append(f"{expected_arm}:backend_repeat_matrix_incomplete")
        return None, defects
    if defects:
        return None, defects

    canonical = by_key[("reference", 0)]
    canonical_payload = _canonical_dimensions(canonical.dimensions)
    for key, receipt in by_key.items():
        if receipt.authority != canonical.authority:
            defects.append(f"{expected_arm}:authority_mismatch:{key}")
        if _canonical_dimensions(receipt.dimensions) != canonical_payload:
            defects.append(f"{expected_arm}:parity_or_repeat_failure:{key}")

    expected_genotype, expected_context = fixture_input_dimensions_v1(
        case.family_id,
        expected_arm,
    )
    if tuple(canonical.dimensions["genotype"]) != expected_genotype:
        defects.append(f"{expected_arm}:frozen_genotype_mismatch")
    if tuple(canonical.dimensions["context"]) != expected_context:
        defects.append(f"{expected_arm}:frozen_context_mismatch")
    return canonical, defects


def _checkpoint_defects(
    pair: MatchedPairEvidenceV1,
    *,
    authority: AuthorityBindingV1 | None,
) -> list[str]:
    checkpoint = pair.checkpoint
    defects: list[str] = []
    if not isinstance(checkpoint, CheckpointReceiptV1):
        return ["invalid_checkpoint_receipt_type"]
    expected = {
        "family_id": pair.control.family_id,
        "pair_id": pair.control.pair_id,
        "control_case_id": pair.control.case_id,
        "treatment_case_id": pair.treatment.case_id,
    }
    for name, value in expected.items():
        if getattr(checkpoint, name) != value:
            defects.append(f"checkpoint_identity_mismatch:{name}")
    defects.extend(
        f"checkpoint:{item}" for item in _authority_defects(checkpoint.authority)
    )
    if authority is not None and checkpoint.authority != authority:
        defects.append("checkpoint_authority_mismatch")
    if (
        checkpoint.restored_checkpoint_sha256s
        != (checkpoint.authority.checkpoint_sha256,) * 2
    ):
        defects.append("checkpoint_repeat_restore_mismatch")
    if checkpoint.exact_restore is not True:
        defects.append("checkpoint_restore_not_exact")
    if not _valid_hash(checkpoint.execution_receipt_sha256):
        defects.append("invalid_checkpoint_execution_receipt_sha256")
    return defects


def _changed_count(left: Sequence[int], right: Sequence[int]) -> int:
    return sum(a != b for a, b in zip(left, right, strict=True))


def _signed_deltas(
    control: Sequence[int],
    treatment: Sequence[int],
) -> tuple[int, ...]:
    return tuple(
        treatment_value - control_value
        for control_value, treatment_value in zip(
            control,
            treatment,
            strict=True,
        )
    )


def _family_checks(
    fixture: Mapping[str, Any],
    control: Mapping[str, Sequence[int]],
    treatment: Mapping[str, Sequence[int]],
) -> dict[str, bool]:
    predicate = fixture["predicate_id"]
    parameters = fixture["predicate_parameters"]
    genotype_equal = control["genotype"] == treatment["genotype"]
    context_equal = control["context"] == treatment["context"]
    product = _signed_deltas(control["product"], treatment["product"])
    phenotype = _signed_deltas(
        control["phenotype"],
        treatment["phenotype"],
    )
    runtime = _signed_deltas(
        control["runtime_consequence"],
        treatment["runtime_consequence"],
    )
    cost = _signed_deltas(control["cost"], treatment["cost"])

    if predicate == "broader_runtime_projection_v1":
        checks = {
            "genotype": not genotype_equal,
            "context": context_equal,
            "product": _changed_count(
                control["product"][:PRODUCT_COUNT],
                treatment["product"][:PRODUCT_COUNT],
            )
            >= 1,
            "regulatory": control["regulatory"]
            == treatment["regulatory"],
            "phenotype": sum(delta != 0 for delta in phenotype)
            >= parameters["minimum_changed_phenotypes"],
            "runtime_consequence": sum(delta != 0 for delta in runtime)
            >= parameters["minimum_changed_runtime_fields"],
            "cost": cost[0] > 0,
        }
    elif predicate == "diploid_dosage_into_saturation_v1":
        index = int(parameters["product_index"])
        checks = {
            "genotype": not genotype_equal,
            "context": context_equal,
            "product": (
                treatment["product"][index]
                == parameters["saturation_q10"]
                and treatment["product"][index] > control["product"][index]
                and treatment["product"][_PRODUCT_SATURATED_COUNT_INDEX]
                > control["product"][_PRODUCT_SATURATED_COUNT_INDEX]
            ),
            "regulatory": control["regulatory"]
            == treatment["regulatory"],
            "phenotype": any(delta != 0 for delta in phenotype),
            "runtime_consequence": any(delta != 0 for delta in runtime),
            "cost": cost[0] > 0,
        }
    elif predicate == "distinct_routes_equal_target_phenotype_v1":
        target = int(parameters["target_phenotype_index"])
        control_product = int(parameters["control_product_index"])
        treatment_product = int(parameters["treatment_product_index"])
        checks = {
            "genotype": not genotype_equal,
            "context": context_equal,
            "product": (
                control["product"][control_product] != 0
                and treatment["product"][treatment_product] != 0
                and control["product"][:PRODUCT_COUNT]
                != treatment["product"][:PRODUCT_COUNT]
            ),
            "regulatory": control["regulatory"]
            != treatment["regulatory"],
            "phenotype": (
                control["phenotype"][target]
                == treatment["phenotype"][target]
                != 0
            ),
            "runtime_consequence": (
                control["runtime_consequence"][target]
                == treatment["runtime_consequence"][target]
            ),
            "cost": control["cost"] != treatment["cost"],
        }
    elif predicate == "hazard_activation_inhibition_v1":
        regulator = int(parameters["regulator_index"])
        activated = int(parameters["activated_product_index"])
        inhibited = int(parameters["inhibited_product_index"])
        checks = {
            "genotype": genotype_equal,
            "context": (
                _changed_count(control["context"], treatment["context"]) == 1
                and treatment["context"][5] == Q10_SCALE
            ),
            "product": (
                product[activated] > 0 and product[inhibited] < 0
            ),
            "regulatory": (
                treatment["regulatory"][
                    _REGULATOR_NODE_OFFSET + regulator
                ]
                > control["regulatory"][
                    _REGULATOR_NODE_OFFSET + regulator
                ]
                and treatment["regulatory"][_ACTIVE_EDGE_COUNT_INDEX]
                > control["regulatory"][_ACTIVE_EDGE_COUNT_INDEX]
            ),
            "phenotype": any(delta != 0 for delta in phenotype),
            "runtime_consequence": any(delta != 0 for delta in runtime),
            "cost": cost[1] > 0,
        }
    elif predicate == "stability_modifier_expression_v1":
        index = int(parameters["product_index"])
        checks = {
            "genotype": (
                not genotype_equal
                and _changed_count(
                    control["genotype"],
                    treatment["genotype"],
                )
                == 1
            ),
            "context": context_equal,
            "product": product[index] > 0,
            "regulatory": (
                control["regulatory"][_DEVELOPMENTAL_STAGE_INDEX]
                == treatment["regulatory"][_DEVELOPMENTAL_STAGE_INDEX]
            ),
            "phenotype": any(delta != 0 for delta in phenotype),
            "runtime_consequence": any(delta != 0 for delta in runtime),
            "cost": cost[0] > 0,
        }
    elif predicate == "mature_to_juvenile_switch_v1":
        regulator = int(parameters["regulator_index"])
        product_index = int(parameters["product_index"])
        checks = {
            "genotype": genotype_equal,
            "context": (
                _changed_count(control["context"], treatment["context"]) == 1
                and treatment["context"][6] == Q10_SCALE
            ),
            "product": product[product_index] > 0,
            "regulatory": (
                control["regulatory"][_DEVELOPMENTAL_STAGE_INDEX]
                == parameters["mature_stage_code"]
                and treatment["regulatory"][_DEVELOPMENTAL_STAGE_INDEX]
                == parameters["juvenile_stage_code"]
                and treatment["regulatory"][
                    _REGULATOR_NODE_OFFSET + regulator
                ]
                > control["regulatory"][
                    _REGULATOR_NODE_OFFSET + regulator
                ]
            ),
            "phenotype": any(delta != 0 for delta in phenotype),
            "runtime_consequence": any(delta != 0 for delta in runtime),
            "cost": cost[1] > 0,
        }
    elif predicate == "opposing_multitrait_effects_v1":
        phenotype_nonzero = [delta for delta in phenotype if delta != 0]
        runtime_nonzero = [delta for delta in runtime[:8] if delta != 0]
        checks = {
            "genotype": not genotype_equal,
            "context": context_equal,
            "product": any(delta != 0 for delta in product[:PRODUCT_COUNT]),
            "regulatory": (
                treatment["regulatory"][_ACTIVE_REGULATOR_COUNT_INDEX]
                >= control["regulatory"][_ACTIVE_REGULATOR_COUNT_INDEX]
            ),
            "phenotype": (
                len(phenotype_nonzero)
                >= parameters["minimum_changed_phenotypes"]
                and any(delta > 0 for delta in phenotype_nonzero)
                and any(delta < 0 for delta in phenotype_nonzero)
            ),
            "runtime_consequence": (
                any(delta > 0 for delta in runtime_nonzero)
                and any(delta < 0 for delta in runtime_nonzero)
            ),
            "cost": cost[0] > 0,
        }
    else:  # pragma: no cover - preregistration is module-owned
        raise ValueError(f"unknown frozen predicate: {predicate!r}")
    return {name: bool(checks[name]) for name in REQUIRED_DIMENSIONS}


def evaluate_family_support_v1(
    pair: MatchedPairEvidenceV1,
) -> dict[str, Any]:
    """Compute the frozen family predicate from validated canonical receipts."""

    fixture = fixture_by_family_v1(pair.control.family_id)
    control = next(
        receipt
        for receipt in pair.control.receipts
        if receipt.backend == "reference" and receipt.repeat_index == 0
    )
    treatment = next(
        receipt
        for receipt in pair.treatment.receipts
        if receipt.backend == "reference" and receipt.repeat_index == 0
    )
    checks = _family_checks(
        fixture,
        control.dimensions,
        treatment.dimensions,
    )
    value = {
        "schema_version": SUPPORT_RECEIPT_SCHEMA_VERSION,
        "family_id": fixture["family_id"],
        "pair_id": fixture["pair_id"],
        "fixture_sha256": fixture["fixture_sha256"],
        "control_case_sha256": pair.control.case_sha256(),
        "treatment_case_sha256": pair.treatment.case_sha256(),
        "checkpoint_receipt_sha256": pair.checkpoint.receipt_sha256(),
        "predicate_id": fixture["predicate_id"],
        "dimension_checks": checks,
        "supported": all(checks.values()),
    }
    return {**value, "support_receipt_sha256": content_sha256(value)}


def _pair_state(
    pair: MatchedPairEvidenceV1,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(pair, MatchedPairEvidenceV1):
        return None, ["invalid_pair_type"]
    try:
        fixture = fixture_by_family_v1(pair.control.family_id)
    except ValueError:
        return None, ["unknown_family"]
    defects: list[str] = []
    if pair.fixture_sha256 != fixture["fixture_sha256"]:
        defects.append("fixture_contract_drift")
    control, control_defects = _case_state(
        pair.control,
        expected_arm="control",
        fixture=fixture,
    )
    treatment, treatment_defects = _case_state(
        pair.treatment,
        expected_arm="treatment",
        fixture=fixture,
    )
    defects.extend(control_defects)
    defects.extend(treatment_defects)
    if control is not None and treatment is not None:
        if control.authority != treatment.authority:
            defects.append("cross_arm_authority_mismatch")
        defects.extend(
            _checkpoint_defects(pair, authority=control.authority)
        )
    else:
        defects.extend(_checkpoint_defects(pair, authority=None))
    if defects:
        return None, defects
    try:
        return evaluate_family_support_v1(pair), []
    except (KeyError, StopIteration, TypeError, ValueError) as exc:
        return None, [f"support_evaluation_failure:{type(exc).__name__}"]


def evaluate_c1_v1(
    pairs: Sequence[MatchedPairEvidenceV1],
) -> dict[str, Any]:
    """Evaluate seven exact matched pairs with D > C > B > A precedence."""

    if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence):
        pairs = ()
    rows = list(pairs)
    integrity_defects: list[str] = []
    supports: list[dict[str, Any]] = []
    for index, pair in enumerate(rows):
        support, defects = _pair_state(pair)
        integrity_defects.extend(
            f"pair[{index}]:{defect}" for defect in defects
        )
        if support is not None:
            supports.append(support)
    if integrity_defects:
        return _decision(DECISION_D, rows, supports, integrity_defects)

    authorities = [
        pair.control.receipts[0].authority
        for pair in rows
        if isinstance(pair, MatchedPairEvidenceV1)
    ]
    if authorities and any(
        authority != authorities[0] for authority in authorities[1:]
    ):
        return _decision(
            DECISION_D,
            rows,
            supports,
            ["cross_family_authority_mismatch"],
        )

    family_counts = {
        family: sum(
            pair.control.family_id == family
            for pair in rows
            if isinstance(pair, MatchedPairEvidenceV1)
        )
        for family in MANDATORY_FAMILIES
    }
    if (
        len(rows) != len(MANDATORY_FAMILIES)
        or any(count != 1 for count in family_counts.values())
    ):
        reasons = [
            f"{family}={family_counts[family]}"
            for family in MANDATORY_FAMILIES
            if family_counts[family] != 1
        ]
        return _decision(
            DECISION_C,
            rows,
            supports,
            ["missing_or_ambiguous_family_pair:" + ",".join(reasons)],
        )
    if any(support["supported"] is not True for support in supports):
        negative = sorted(
            support["family_id"]
            for support in supports
            if support["supported"] is not True
        )
        return _decision(
            DECISION_B,
            rows,
            supports,
            ["negative_family_predicates:" + ",".join(negative)],
        )
    return _decision(DECISION_A, rows, supports, [])


def _decision(
    decision: str,
    pairs: Sequence[Any],
    supports: Sequence[Mapping[str, Any]],
    reasons: Sequence[str],
) -> dict[str, Any]:
    pair_rows = []
    for pair in pairs:
        if not isinstance(pair, MatchedPairEvidenceV1):
            continue
        try:
            pair_sha256 = pair.pair_sha256()
        except (KeyError, TypeError, ValueError):
            pair_sha256 = None
        pair_rows.append(
            {
                "family_id": pair.control.family_id,
                "pair_id": pair.control.pair_id,
                "pair_sha256": pair_sha256,
            }
        )
    support_rows = [
        {
            "family_id": support["family_id"],
            "supported": support["supported"],
            "support_receipt_sha256": support[
                "support_receipt_sha256"
            ],
        }
        for support in supports
    ]
    payload = {
        "schema_version": "genome_v2_p10_c1_decision_v2",
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "decision": decision,
        "reasons": list(reasons),
        "pairs": sorted(
            pair_rows,
            key=lambda row: (str(row["family_id"]), str(row["pair_id"])),
        ),
        "family_support": sorted(
            support_rows,
            key=lambda row: str(row["family_id"]),
        ),
    }
    return {**payload, "decision_sha256": content_sha256(payload)}


__all__ = [
    "AuthorityBindingV1",
    "BACKEND_RECEIPT_SCHEMA_VERSION",
    "BackendCaseReceiptV1",
    "CASE_SCHEMA_VERSION",
    "CHECKPOINT_RECEIPT_SCHEMA_VERSION",
    "CheckpointReceiptV1",
    "DECISION_A",
    "DECISION_B",
    "DECISION_C",
    "DECISION_D",
    "DIMENSION_LENGTHS",
    "MANDATORY_FAMILIES",
    "MatchedCaseRecordV1",
    "MatchedPairEvidenceV1",
    "PAIR_SCHEMA_VERSION",
    "PREREGISTRATION_SHA256",
    "PROVIDER_ID",
    "REQUIRED_BACKENDS",
    "REQUIRED_DIMENSIONS",
    "REQUIRED_REPEAT_INDICES",
    "RUNTIME_PROFILE_ID",
    "SCHEMA_VERSION",
    "SUPPORT_RECEIPT_SCHEMA_VERSION",
    "evaluate_c1_v1",
    "evaluate_family_support_v1",
    "fixture_by_family_v1",
    "fixture_input_dimensions_v1",
    "preregistration_payload_v1",
]
