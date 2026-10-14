from __future__ import annotations

from dataclasses import replace

import pytest

from engine.authoritative_state.schema import content_sha256
from engine.evolution.genome_upgrade_v2_c1_executor_v1 import (
    EXECUTOR_SCHEMA_VERSION,
    BackendCaseOutputV1,
    GenomeV2C1ExecutorError,
    execute_c1_v1,
    reference_dimensions_for_fixture_v1,
)
from engine.evolution.genome_upgrade_v2_causal_gate_v1 import (
    DECISION_A,
    DECISION_D,
    DIMENSION_LENGTHS,
    MANDATORY_FAMILIES,
    REQUIRED_BACKENDS,
    REQUIRED_DIMENSIONS,
    REQUIRED_REPEAT_INDICES,
    AuthorityBindingV1,
    CheckpointReceiptV1,
    fixture_by_family_v1,
    fixture_input_dimensions_v1,
)


HASH = "a" * 64
COMMIT = "b" * 40


def _authority() -> AuthorityBindingV1:
    return AuthorityBindingV1(
        source_commit=COMMIT,
        package_id_sha256=HASH,
        package_receipt_sha256=HASH,
        artifact_sha256=HASH,
        execution_route_sha256=HASH,
        checkpoint_sha256=HASH,
    )


def _outputs() -> list[BackendCaseOutputV1]:
    outputs = []
    for family in MANDATORY_FAMILIES:
        for arm in ("control", "treatment"):
            dimensions = reference_dimensions_for_fixture_v1(family, arm)
            for backend in REQUIRED_BACKENDS:
                for repeat in REQUIRED_REPEAT_INDICES:
                    outputs.append(
                        BackendCaseOutputV1(
                            family_id=family,
                            arm=arm,
                            backend=backend,
                            repeat_index=repeat,
                            dimensions=dimensions,
                            execution_receipt_sha256=content_sha256(
                                [family, arm, backend, repeat]
                            ),
                            no_fallback=True,
                            no_jit=True,
                        )
                    )
    return outputs


def _checkpoints() -> dict[str, CheckpointReceiptV1]:
    result = {}
    for family in MANDATORY_FAMILIES:
        fixture = fixture_by_family_v1(family)
        result[family] = CheckpointReceiptV1(
            family_id=family,
            pair_id=fixture["pair_id"],
            control_case_id=f"{fixture['pair_id']}-control",
            treatment_case_id=f"{fixture['pair_id']}-treatment",
            authority=_authority(),
            restored_checkpoint_sha256s=(HASH, HASH),
            execution_receipt_sha256=content_sha256(
                ["checkpoint", family]
            ),
            exact_restore=True,
        )
    return result


def _execute(outputs: list[BackendCaseOutputV1] | None = None):
    return execute_c1_v1(
        authorities={family: _authority() for family in MANDATORY_FAMILIES},
        backend_outputs=_outputs() if outputs is None else outputs,
        checkpoint_receipts=_checkpoints(),
    )


@pytest.mark.parametrize("family", MANDATORY_FAMILIES)
@pytest.mark.parametrize("arm", ["control", "treatment"])
def test_reference_serialization_matches_frozen_inputs_and_shapes(
    family: str,
    arm: str,
) -> None:
    dimensions = reference_dimensions_for_fixture_v1(family, arm)
    genotype, context = fixture_input_dimensions_v1(family, arm)
    assert tuple(dimensions) == REQUIRED_DIMENSIONS
    assert dimensions["genotype"] == genotype
    assert dimensions["context"] == context
    assert {
        name: len(values) for name, values in dimensions.items()
    } == DIMENSION_LENGTHS
    assert all(
        isinstance(value, int) and not isinstance(value, bool)
        for values in dimensions.values()
        for value in values
    )


def test_exact_seven_family_three_backend_x2_matrix_decides_a() -> None:
    result = _execute()
    payload = result.payload()
    assert result.decision["decision"] == DECISION_A
    assert len(result.pairs) == 7
    assert payload["schema_version"] == EXECUTOR_SCHEMA_VERSION
    assert payload["execution_sha256"] == content_sha256(
        {
            key: value
            for key, value in payload.items()
            if key != "execution_sha256"
        }
    )
    assert all(
        len(pair.control.receipts) == 6
        and len(pair.treatment.receipts) == 6
        for pair in result.pairs
    )


def test_executor_rejects_incomplete_or_duplicate_raw_matrix() -> None:
    outputs = _outputs()
    with pytest.raises(GenomeV2C1ExecutorError, match="matrix"):
        _execute(outputs[:-1])
    with pytest.raises(GenomeV2C1ExecutorError, match="duplicate"):
        _execute([*outputs, outputs[0]])


def test_executor_routes_backend_integrity_defect_to_d() -> None:
    outputs = _outputs()
    outputs[0] = replace(outputs[0], no_jit=False)
    assert _execute(outputs).decision["decision"] == DECISION_D
