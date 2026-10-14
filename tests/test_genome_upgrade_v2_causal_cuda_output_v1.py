from __future__ import annotations

from dataclasses import astuple

import pytest
import torch

from engine.evolution.genome_upgrade_v2 import (
    GenomeUpgradeV2V1,
    express_reference_v1,
    project_runtime_traits_v1,
)
from engine.evolution.genome_upgrade_v2_causal_cuda_output_v1 import (
    RECORD_CAPACITY,
    RECORD_WIDTH,
    SCHEMA_VERSION,
    UNAVAILABLE_VALUE,
    GenomeV2CausalCudaOutputError,
    decode_genome_v2_causal_cuda_outputs_v1,
)


def _outputs(
    *,
    producer: str = "birth",
    context_valid: bool = True,
    dropped: int = 0,
    parent_id: int = 3,
    mate_parent_id: int = 5,
    parent_slot: int = 1,
    mate_parent_slot: int = 2,
) -> dict[str, torch.Tensor]:
    prefix = f"{producer}_genome_v2_causal"
    rows = torch.full(
        (1, RECORD_CAPACITY, RECORD_WIDTH),
        UNAVAILABLE_VALUE,
        dtype=torch.int64,
    )
    valid = torch.zeros((1, RECORD_CAPACITY), dtype=torch.bool)
    count = torch.tensor([1], dtype=torch.int64)
    dropped_count = torch.tensor([dropped], dtype=torch.int64)
    genome = GenomeUpgradeV2V1.neutral()
    context = (0,) * 8
    expression = express_reference_v1(genome, context)
    runtime = project_runtime_traits_v1(expression)
    row = rows[0, 0]
    row[0] = 0
    row[1] = 1 if producer == "birth" else 2
    row[2] = 0
    row[3] = 2
    row[4] = 31
    row[5] = 9
    row[6] = 1
    row[7] = 17
    row[8] = parent_id
    row[9] = 4
    row[10] = 1
    row[11] = 2
    column = 12
    for homolog in genome.alleles:
        for product in homolog:
            for parameter in product:
                row[column] = parameter
                column += 1
    for homolog in genome.modifiers:
        for modifier in homolog:
            row[column] = modifier
            column += 1
    assert column == 132
    row[132:144] = torch.tensor(expression.products_q10)
    row[144] = 0
    row[145:155] = torch.tensor(expression.phenotypes_q10)
    row[155] = int(context_valid)
    if context_valid:
        row[156:164] = torch.tensor(context)
    row[164] = expression.expression_cost_q10
    row[165] = 1
    row[166] = expression.regulatory_cost_q10
    row[167:179] = torch.tensor(astuple(runtime))
    row[179] = mate_parent_id
    row[180] = parent_slot
    row[181] = mate_parent_slot
    valid[0, 0] = True
    return {
        f"{prefix}_record_rows": rows,
        f"{prefix}_record_valid": valid,
        f"{prefix}_record_count": count,
        f"{prefix}_record_dropped_count": dropped_count,
        f"{prefix}_capture_enabled": torch.tensor([True]),
    }


def _decode(
    outputs: dict[str, torch.Tensor],
    *,
    producer: str = "birth",
):
    return decode_genome_v2_causal_cuda_outputs_v1(
        outputs,
        producer=producer,  # type: ignore[arg-type]
        cell_ids=("cell-a",),
        capability_contract_sha256="a" * 64,
        package_id_sha256="b" * 64,
        artifact_sha256="c" * 64,
        execution_route_sha256="d" * 64,
    )


@pytest.mark.parametrize("producer", ("birth", "turnover"))
def test_decodes_and_independently_verifies_committed_record(
    producer: str,
) -> None:
    result = _decode(_outputs(producer=producer), producer=producer)

    assert SCHEMA_VERSION == "genome_v2_causal_cuda_output_v3"
    assert RECORD_WIDTH == 182
    assert result.evaluable
    assert result.captured_count == 1
    assert result.dropped_count == 0
    row = result.rows[0]
    assert row.integrity_verified
    assert row.identity.event_kind == (
        "ordinary_birth" if producer == "birth" else "turnover_refill_child"
    )
    assert row.identity.parent_id == 3
    assert row.identity.mate_parent_id == 5
    assert row.identity.parent_slot == 1
    assert row.identity.mate_parent_slot == 2
    assert row.recomputed_expression is not None
    assert row.recomputed_expression.regulator_nodes_q10


def test_dropped_rows_and_unavailable_context_are_non_evaluable_metadata() -> None:
    outputs = _outputs(context_valid=False, dropped=3)
    outputs["birth_genome_v2_causal_record_rows"][0, 0, 165] = 0
    outputs["birth_genome_v2_causal_record_rows"][0, 0, 166] = (
        UNAVAILABLE_VALUE
    )

    result = _decode(outputs)

    assert not result.evaluable
    assert result.dropped_count == 3
    assert result.cells[0].non_evaluable_reasons == (
        "causal_rows_dropped",
        "frozen_event_context_unavailable",
    )
    assert result.rows[0].recomputed_expression is None


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        (
            "record_count",
            torch.tensor([RECORD_CAPACITY + 1]),
            "outside its fixed capacity",
        ),
        ("record_dropped_count", torch.tensor([-1]), "negative"),
        (
            "record_valid",
            torch.zeros((1, RECORD_CAPACITY), dtype=torch.bool),
            "exact count prefix",
        ),
    ),
)
def test_rejects_invalid_count_prefix_and_dropped_contract(
    field: str,
    replacement: torch.Tensor,
    message: str,
) -> None:
    outputs = _outputs()
    outputs[f"birth_genome_v2_causal_{field}"] = replacement

    with pytest.raises(GenomeV2CausalCudaOutputError, match=message):
        _decode(outputs)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    (
        (0, 1, "ordinal"),
        (1, 2, "event kind"),
        (2, 1, "cell ordinal"),
        (6, 0, "no valid sequence"),
        (144, 1, "cannot carry claimed regulator"),
        (179, 3, "causal event identity is invalid"),
        (180, -1, "causal event identity is invalid"),
        (181, -1, "causal event identity is invalid"),
    ),
)
def test_rejects_malformed_or_contradictory_identity(
    column: int,
    value: int,
    message: str,
) -> None:
    outputs = _outputs()
    outputs["birth_genome_v2_causal_record_rows"][0, 0, column] = value

    with pytest.raises(GenomeV2CausalCudaOutputError, match=message):
        _decode(outputs)


@pytest.mark.parametrize(
    ("column", "message"),
    (
        (132, "products contradict"),
        (145, "phenotypes contradict"),
        (164, "expression cost contradicts"),
        (167, "runtime projection contradicts"),
        (166, "regulatory cost contradicts"),
    ),
)
def test_rejects_committed_expression_contradictions(
    column: int,
    message: str,
) -> None:
    outputs = _outputs()
    outputs["birth_genome_v2_causal_record_rows"][0, 0, column] += 1

    with pytest.raises(GenomeV2CausalCudaOutputError, match=message):
        _decode(outputs)


def test_rejects_wrong_shape_dtype_and_noncanonical_cell_ids() -> None:
    outputs = _outputs()
    outputs["birth_genome_v2_causal_record_rows"] = outputs[
        "birth_genome_v2_causal_record_rows"
    ].to(torch.int32)
    with pytest.raises(GenomeV2CausalCudaOutputError, match="wrong dtype"):
        _decode(outputs)

    with pytest.raises(GenomeV2CausalCudaOutputError, match="cell_ids"):
        decode_genome_v2_causal_cuda_outputs_v1(
            _outputs(),
            producer="birth",
            cell_ids=(" cell-a",),
            capability_contract_sha256="a" * 64,
            package_id_sha256="b" * 64,
            artifact_sha256="c" * 64,
            execution_route_sha256="d" * 64,
        )


def test_preserves_high_int64_parent_identities() -> None:
    result = _decode(
        _outputs(
            parent_id=2**40 + 3,
            mate_parent_id=2**40 + 5,
        )
    )

    assert result.rows[0].identity.parent_id == 2**40 + 3
    assert result.rows[0].identity.mate_parent_id == 2**40 + 5


def test_rejects_unavailable_mate_parent_identity() -> None:
    with pytest.raises(
        GenomeV2CausalCudaOutputError,
        match="causal event identity is invalid",
    ):
        _decode(_outputs(mate_parent_id=UNAVAILABLE_VALUE))


def test_rejects_ambiguous_duplicate_natural_identity() -> None:
    outputs = _outputs()
    prefix = "birth_genome_v2_causal"
    outputs[f"{prefix}_record_rows"][0, 1].copy_(
        outputs[f"{prefix}_record_rows"][0, 0]
    )
    outputs[f"{prefix}_record_rows"][0, 1, 0] = 1
    outputs[f"{prefix}_record_valid"][0, 1] = True
    outputs[f"{prefix}_record_count"][0] = 2

    with pytest.raises(GenomeV2CausalCudaOutputError, match="ambiguous"):
        _decode(outputs)


def test_disabled_or_empty_capture_is_non_evaluable_without_crashing() -> None:
    outputs = _outputs()
    prefix = "birth_genome_v2_causal"
    outputs[f"{prefix}_record_rows"].fill_(UNAVAILABLE_VALUE)
    outputs[f"{prefix}_record_valid"].zero_()
    outputs[f"{prefix}_record_count"].zero_()
    outputs[f"{prefix}_capture_enabled"].zero_()

    disabled = _decode(outputs)
    assert not disabled.evaluable
    assert disabled.cells[0].non_evaluable_reasons == (
        "causal_capture_disabled",
    )

    outputs[f"{prefix}_capture_enabled"].fill_(True)
    empty = _decode(outputs)
    assert not empty.evaluable
    assert empty.cells[0].non_evaluable_reasons == (
        "no_causal_rows_captured",
    )


def test_rejects_claimed_records_when_capture_is_disabled() -> None:
    outputs = _outputs()
    outputs["birth_genome_v2_causal_capture_enabled"].zero_()

    with pytest.raises(GenomeV2CausalCudaOutputError, match="disabled"):
        _decode(outputs)


def test_capture_enabled_requires_exact_bool_cell_vector() -> None:
    outputs = _outputs()
    outputs["birth_genome_v2_causal_capture_enabled"] = torch.ones(
        (1, 1), dtype=torch.bool
    )
    with pytest.raises(GenomeV2CausalCudaOutputError, match="wrong fixed shape"):
        _decode(outputs)

    outputs = _outputs()
    outputs["birth_genome_v2_causal_capture_enabled"] = torch.ones(
        (1,), dtype=torch.int64
    )
    with pytest.raises(GenomeV2CausalCudaOutputError, match="wrong dtype"):
        _decode(outputs)


def test_disabled_capture_rejects_valid_rows_even_with_zero_counts() -> None:
    outputs = _outputs()
    prefix = "birth_genome_v2_causal"
    outputs[f"{prefix}_record_count"].zero_()
    outputs[f"{prefix}_record_dropped_count"].zero_()
    outputs[f"{prefix}_capture_enabled"].zero_()

    with pytest.raises(GenomeV2CausalCudaOutputError, match="exact count prefix"):
        _decode(outputs)


def test_physical_cell_ordinal_validates_sliced_b1_evidence() -> None:
    outputs = _outputs()
    outputs["birth_genome_v2_causal_record_rows"][0, 0, 2] = 41

    result = decode_genome_v2_causal_cuda_outputs_v1(
        outputs,
        producer="birth",
        cell_ids=("cell-41",),
        physical_cell_ordinals=(41,),
        capability_contract_sha256="a" * 64,
        package_id_sha256="b" * 64,
        artifact_sha256="c" * 64,
        execution_route_sha256="d" * 64,
    )

    assert result.cells[0].cell_ordinal == 41
    assert result.rows[0].identity.cell_id == "cell-41"
    with pytest.raises(GenomeV2CausalCudaOutputError, match="cell ordinal"):
        _decode(outputs)


@pytest.mark.parametrize(
    "ordinals",
    ((-1,), (1, 2), (True,)),
)
def test_rejects_invalid_physical_cell_ordinal_mapping(
    ordinals: tuple[int, ...],
) -> None:
    with pytest.raises(
        GenomeV2CausalCudaOutputError,
        match="physical_cell_ordinals",
    ):
        decode_genome_v2_causal_cuda_outputs_v1(
            _outputs(),
            producer="birth",
            cell_ids=("cell-a",),
            physical_cell_ordinals=ordinals,
            capability_contract_sha256="a" * 64,
            package_id_sha256="b" * 64,
            artifact_sha256="c" * 64,
            execution_route_sha256="d" * 64,
        )
