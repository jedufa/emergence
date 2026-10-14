"""Deterministic executor for the preregistered Genome V2 P1.0 C1 gate."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Final, Mapping, Sequence

from engine.authoritative_state.schema import content_sha256

from .genome_upgrade_v2 import (
    PARAMETER_COUNT,
    PRODUCT_COUNT,
    GenomeUpgradeV2V1,
    express_reference_v1,
    project_runtime_traits_v1,
)
from .genome_upgrade_v2_causal_gate_v1 import (
    MANDATORY_FAMILIES,
    PREREGISTRATION_SHA256,
    REQUIRED_BACKENDS,
    REQUIRED_REPEAT_INDICES,
    AuthorityBindingV1,
    BackendCaseReceiptV1,
    CheckpointReceiptV1,
    MatchedCaseRecordV1,
    MatchedPairEvidenceV1,
    evaluate_c1_v1,
    fixture_by_family_v1,
    fixture_input_dimensions_v1,
)


EXECUTOR_SCHEMA_VERSION: Final[str] = "genome_v2_p10_c1_executor_v1"
_ARMS: Final[tuple[str, str]] = ("control", "treatment")
_STAGE_CODES: Final[dict[str, int]] = {
    "mature": 0,
    "juvenile": 1,
    "senescent": 2,
}


class GenomeV2C1ExecutorError(ValueError):
    """Raised when raw C1 evidence cannot form the preregistered matrix."""


@dataclass(frozen=True, slots=True)
class BackendCaseOutputV1:
    """One backend execution result before it is bound into a case receipt."""

    family_id: str
    arm: str
    backend: str
    repeat_index: int
    dimensions: Mapping[str, Sequence[int]]
    execution_receipt_sha256: str
    no_fallback: bool
    no_jit: bool


@dataclass(frozen=True, slots=True)
class C1ExecutionResultV1:
    """Content-addressed evidence and decision produced by one C1 execution."""

    pairs: tuple[MatchedPairEvidenceV1, ...]
    decision: Mapping[str, Any]

    def payload(self) -> dict[str, Any]:
        value = {
            "schema_version": EXECUTOR_SCHEMA_VERSION,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "pair_sha256s": [
                pair.pair_sha256()
                for pair in sorted(
                    self.pairs,
                    key=lambda item: item.control.family_id,
                )
            ],
            "decision": dict(self.decision),
        }
        return {**value, "execution_sha256": content_sha256(value)}


def _genome_from_vector(genotype: Sequence[int]) -> GenomeUpgradeV2V1:
    allele_width = 2 * PRODUCT_COUNT * PARAMETER_COUNT
    alleles = tuple(
        tuple(
            tuple(
                genotype[
                    (homolog * PRODUCT_COUNT + product) * PARAMETER_COUNT :
                    (homolog * PRODUCT_COUNT + product + 1)
                    * PARAMETER_COUNT
                ]
            )
            for product in range(PRODUCT_COUNT)
        )
        for homolog in range(2)
    )
    modifiers = (
        tuple(genotype[allele_width : allele_width + PRODUCT_COUNT]),
        tuple(genotype[allele_width + PRODUCT_COUNT :]),
    )
    return GenomeUpgradeV2V1(alleles=alleles, modifiers=modifiers)


def reference_dimensions_for_fixture_v1(
    family_id: str,
    arm: str,
) -> dict[str, tuple[int, ...]]:
    """Evaluate one frozen fixture arm with the authoritative reference path."""

    genotype, context = fixture_input_dimensions_v1(family_id, arm)
    expression = express_reference_v1(
        _genome_from_vector(genotype),
        context,
    )
    runtime = project_runtime_traits_v1(expression)
    try:
        stage_code = _STAGE_CODES[expression.developmental_stage]
    except KeyError as exc:  # pragma: no cover - mechanic contract owns values
        raise GenomeV2C1ExecutorError(
            "reference expression returned an unknown developmental stage"
        ) from exc
    return {
        "genotype": genotype,
        "context": context,
        "product": expression.products_q10
        + (
            expression.active_product_count,
            expression.saturated_product_count,
        ),
        "regulatory": expression.regulator_nodes_q10
        + expression.epistatic_delta_q10
        + (
            expression.regulatory_cost_q10,
            expression.active_regulator_count,
            expression.active_regulatory_edge_count,
            stage_code,
        ),
        "phenotype": expression.phenotypes_q10,
        "runtime_consequence": tuple(
            int(getattr(runtime, field.name)) for field in fields(runtime)
        ),
        "cost": (
            expression.expression_cost_q10,
            expression.regulatory_cost_q10,
        ),
    }


def _output_matrix(
    backend_outputs: Sequence[BackendCaseOutputV1],
) -> dict[tuple[str, str, str, int], BackendCaseOutputV1]:
    expected = {
        (family, arm, backend, repeat)
        for family in MANDATORY_FAMILIES
        for arm in _ARMS
        for backend in REQUIRED_BACKENDS
        for repeat in REQUIRED_REPEAT_INDICES
    }
    matrix: dict[tuple[str, str, str, int], BackendCaseOutputV1] = {}
    for output in backend_outputs:
        if not isinstance(output, BackendCaseOutputV1):
            raise GenomeV2C1ExecutorError(
                "backend output has an invalid type"
            )
        key = (
            output.family_id,
            output.arm,
            output.backend,
            output.repeat_index,
        )
        if key in matrix:
            raise GenomeV2C1ExecutorError(
                f"duplicate backend output for {key!r}"
            )
        matrix[key] = output
    missing = sorted(expected - set(matrix))
    unexpected = sorted(set(matrix) - expected)
    if missing or unexpected:
        raise GenomeV2C1ExecutorError(
            "backend output matrix does not match preregistration: "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    return matrix


def _authority_map(
    authorities: Mapping[str, AuthorityBindingV1],
) -> dict[str, AuthorityBindingV1]:
    if set(authorities) != set(MANDATORY_FAMILIES):
        raise GenomeV2C1ExecutorError(
            "authority bindings do not cover exactly seven C1 families"
        )
    values = dict(authorities)
    if any(
        not isinstance(authority, AuthorityBindingV1)
        for authority in values.values()
    ):
        raise GenomeV2C1ExecutorError("authority binding has an invalid type")
    return values


def _checkpoint_map(
    checkpoints: Mapping[str, CheckpointReceiptV1],
) -> dict[str, CheckpointReceiptV1]:
    if set(checkpoints) != set(MANDATORY_FAMILIES):
        raise GenomeV2C1ExecutorError(
            "checkpoint receipts do not cover exactly seven C1 families"
        )
    values = dict(checkpoints)
    if any(
        not isinstance(checkpoint, CheckpointReceiptV1)
        for checkpoint in values.values()
    ):
        raise GenomeV2C1ExecutorError("checkpoint receipt has an invalid type")
    return values


def execute_c1_v1(
    *,
    authorities: Mapping[str, AuthorityBindingV1],
    backend_outputs: Sequence[BackendCaseOutputV1],
    checkpoint_receipts: Mapping[str, CheckpointReceiptV1],
) -> C1ExecutionResultV1:
    """Bind an exact backend matrix into receipts and evaluate C1.

    Structural executor defects raise ``GenomeV2C1ExecutorError``. Evidence
    identity, parity, restore, fallback, JIT, and predicate defects are passed
    to the gate and fail closed as C1-D or C1-B.
    """

    authority_by_family = _authority_map(authorities)
    output_by_key = _output_matrix(backend_outputs)
    checkpoint_by_family = _checkpoint_map(checkpoint_receipts)
    pairs: list[MatchedPairEvidenceV1] = []
    for family in MANDATORY_FAMILIES:
        fixture = fixture_by_family_v1(family)
        authority = authority_by_family[family]
        cases: dict[str, MatchedCaseRecordV1] = {}
        for arm in _ARMS:
            case_id = f"{fixture['pair_id']}-{arm}"
            receipts = tuple(
                BackendCaseReceiptV1(
                    family_id=family,
                    pair_id=fixture["pair_id"],
                    case_id=case_id,
                    arm=arm,
                    seed=fixture["seed"],
                    backend=backend,
                    repeat_index=repeat,
                    authority=authority,
                    dimensions=output_by_key[
                        (family, arm, backend, repeat)
                    ].dimensions,
                    execution_receipt_sha256=output_by_key[
                        (family, arm, backend, repeat)
                    ].execution_receipt_sha256,
                    no_fallback=output_by_key[
                        (family, arm, backend, repeat)
                    ].no_fallback,
                    no_jit=output_by_key[
                        (family, arm, backend, repeat)
                    ].no_jit,
                )
                for backend in REQUIRED_BACKENDS
                for repeat in REQUIRED_REPEAT_INDICES
            )
            cases[arm] = MatchedCaseRecordV1(
                family_id=family,
                pair_id=fixture["pair_id"],
                case_id=case_id,
                arm=arm,
                seed=fixture["seed"],
                receipts=receipts,
            )
        pairs.append(
            MatchedPairEvidenceV1(
                control=cases["control"],
                treatment=cases["treatment"],
                checkpoint=checkpoint_by_family[family],
                fixture_sha256=fixture["fixture_sha256"],
            )
        )
    pair_tuple = tuple(pairs)
    return C1ExecutionResultV1(
        pairs=pair_tuple,
        decision=evaluate_c1_v1(pair_tuple),
    )


__all__ = [
    "BackendCaseOutputV1",
    "C1ExecutionResultV1",
    "EXECUTOR_SCHEMA_VERSION",
    "GenomeV2C1ExecutorError",
    "execute_c1_v1",
    "reference_dimensions_for_fixture_v1",
]
