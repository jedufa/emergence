"""Decode and verify bounded CUDA Genome V2 causal output records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, Final, Literal

import torch

from .genome_upgrade_v2 import (
    CONTEXT_COUNT,
    PHENOTYPE_COUNT,
    PRODUCT_COUNT,
    GenomeUpgradeV2ExpressionV1,
    GenomeUpgradeV2RuntimeProjectionV1,
    GenomeUpgradeV2V1,
    express_reference_v1,
    project_runtime_traits_v1,
)
from .genome_upgrade_v2_causal_transaction_projection_v1 import (
    GenomeV2CausalTransactionIdentityV1,
)


SCHEMA_VERSION: Final[str] = "genome_v2_causal_cuda_output_v3"
RECORD_CAPACITY: Final[int] = 1024
RECORD_WIDTH: Final[int] = 182
UNAVAILABLE_VALUE: Final[int] = -(2**63)
ALLELE_COUNT: Final[int] = 2 * PRODUCT_COUNT * 4
MODIFIER_COUNT: Final[int] = 2 * PRODUCT_COUNT
RUNTIME_PROJECTION_COUNT: Final[int] = 12

_EVENT_KINDS: Final[dict[str, tuple[int, str]]] = {
    "birth": (1, "ordinary_birth"),
    "turnover": (2, "turnover_refill_child"),
}
_COLUMN_ORDINAL = 0
_COLUMN_EVENT_KIND = 1
_COLUMN_CELL = 2
_COLUMN_GENERATION = 3
_COLUMN_TICK = 4
_COLUMN_EVENT_SEQUENCE = 5
_COLUMN_EVENT_SEQUENCE_VALID = 6
_COLUMN_CHILD_ID = 7
_COLUMN_PARENT_ID = 8
_COLUMN_DESTINATION_SLOT = 9
_COLUMN_FOUNDER_ID = 10
_COLUMN_LINEAGE_DEPTH = 11
_COLUMN_ALLELES = 12
_COLUMN_MODIFIERS = _COLUMN_ALLELES + ALLELE_COUNT
_COLUMN_PRODUCTS = _COLUMN_MODIFIERS + MODIFIER_COUNT
_COLUMN_REGULATOR_VALUES_VALID = _COLUMN_PRODUCTS + PRODUCT_COUNT
_COLUMN_PHENOTYPES = _COLUMN_REGULATOR_VALUES_VALID + 1
_COLUMN_CONTEXT_VALID = _COLUMN_PHENOTYPES + PHENOTYPE_COUNT
_COLUMN_CONTEXT = _COLUMN_CONTEXT_VALID + 1
_COLUMN_EXPRESSION_COST = _COLUMN_CONTEXT + CONTEXT_COUNT
_COLUMN_REGULATORY_COST_VALID = _COLUMN_EXPRESSION_COST + 1
_COLUMN_REGULATORY_COST = _COLUMN_REGULATORY_COST_VALID + 1
_COLUMN_RUNTIME = _COLUMN_REGULATORY_COST + 1
_COLUMN_MATE_PARENT_ID = _COLUMN_RUNTIME + RUNTIME_PROJECTION_COUNT
_COLUMN_PARENT_SLOT = _COLUMN_MATE_PARENT_ID + 1
_COLUMN_MATE_PARENT_SLOT = _COLUMN_PARENT_SLOT + 1
assert _COLUMN_MATE_PARENT_SLOT + 1 == RECORD_WIDTH


class GenomeV2CausalCudaOutputError(ValueError):
    """CUDA causal output is malformed, ambiguous, or contradictory."""


def _fail(message: str) -> GenomeV2CausalCudaOutputError:
    return GenomeV2CausalCudaOutputError(message)


@dataclass(frozen=True)
class GenomeV2CudaCausalDecodedRowV1:
    identity: GenomeV2CausalTransactionIdentityV1
    record_ordinal: int
    producer: str
    genome: GenomeUpgradeV2V1
    committed_products_q10: tuple[int, ...]
    committed_phenotypes_q10: tuple[int, ...]
    committed_expression_cost_q10: int
    committed_runtime_projection: GenomeUpgradeV2RuntimeProjectionV1
    context_q10: tuple[int, ...] | None
    recomputed_expression: GenomeUpgradeV2ExpressionV1 | None
    integrity_verified: bool
    non_evaluable_reason: str | None


@dataclass(frozen=True)
class GenomeV2CudaCausalCellDecodeV1:
    cell_ordinal: int
    cell_id: str
    captured_count: int
    dropped_count: int
    rows: tuple[GenomeV2CudaCausalDecodedRowV1, ...]
    evaluable: bool
    non_evaluable_reasons: tuple[str, ...]


@dataclass(frozen=True)
class GenomeV2CudaCausalDecodeResultV1:
    schema_version: str
    producer: str
    cells: tuple[GenomeV2CudaCausalCellDecodeV1, ...]

    @property
    def rows(self) -> tuple[GenomeV2CudaCausalDecodedRowV1, ...]:
        return tuple(row for cell in self.cells for row in cell.rows)

    @property
    def evaluable(self) -> bool:
        return bool(self.cells) and all(cell.evaluable for cell in self.cells)

    @property
    def captured_count(self) -> int:
        return sum(cell.captured_count for cell in self.cells)

    @property
    def dropped_count(self) -> int:
        return sum(cell.dropped_count for cell in self.cells)


def _tensor(
    outputs: Mapping[str, Any],
    name: str,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    value = outputs.get(name)
    if not isinstance(value, torch.Tensor):
        raise _fail(f"{name} must be a torch tensor")
    if value.dtype != dtype:
        raise _fail(f"{name} has the wrong dtype")
    if tuple(value.shape) != shape:
        raise _fail(f"{name} has the wrong fixed shape")
    if not value.is_contiguous():
        raise _fail(f"{name} must be contiguous")
    return value.detach().cpu()


def _exact_cell_ids(value: Sequence[str], *, cells: int) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or len(value) != cells:
        raise _fail("cell_ids must contain exactly one ID per cell")
    result = tuple(value)
    if any(
        type(cell_id) is not str
        or not cell_id
        or cell_id.strip() != cell_id
        for cell_id in result
    ):
        raise _fail("cell_ids contain a non-canonical ID")
    if len(set(result)) != len(result):
        raise _fail("cell_ids are ambiguous")
    return result


def _physical_cell_ordinals(
    value: Sequence[int] | None,
    *,
    cells: int,
) -> tuple[int, ...]:
    if value is None:
        return tuple(range(cells))
    if isinstance(value, (str, bytes)) or len(value) != cells:
        raise _fail(
            "physical_cell_ordinals must contain exactly one ordinal per cell"
        )
    result = tuple(value)
    if any(type(ordinal) is not int or ordinal < 0 for ordinal in result):
        raise _fail("physical_cell_ordinals contain an invalid ordinal")
    if len(set(result)) != len(result):
        raise _fail("physical_cell_ordinals are ambiguous")
    return result


def _runtime_projection(values: tuple[int, ...]) -> GenomeUpgradeV2RuntimeProjectionV1:
    names = tuple(field.name for field in fields(GenomeUpgradeV2RuntimeProjectionV1))
    if len(names) != RUNTIME_PROJECTION_COUNT:
        raise _fail("runtime projection authority width changed")
    return GenomeUpgradeV2RuntimeProjectionV1(**dict(zip(names, values)))


def _genome(row: torch.Tensor) -> GenomeUpgradeV2V1:
    allele_values = tuple(
        int(value) for value in row[_COLUMN_ALLELES:_COLUMN_MODIFIERS].tolist()
    )
    modifier_values = tuple(
        int(value) for value in row[_COLUMN_MODIFIERS:_COLUMN_PRODUCTS].tolist()
    )
    alleles = tuple(
        tuple(
            tuple(
                allele_values[(homolog * PRODUCT_COUNT + product) * 4 + parameter]
                for parameter in range(4)
            )
            for product in range(PRODUCT_COUNT)
        )
        for homolog in range(2)
    )
    modifiers = tuple(
        tuple(
            modifier_values[homolog * PRODUCT_COUNT + product]
            for product in range(PRODUCT_COUNT)
        )
        for homolog in range(2)
    )
    try:
        return GenomeUpgradeV2V1(alleles=alleles, modifiers=modifiers)
    except (TypeError, ValueError) as exc:
        raise _fail("packed genotype violates the Genome V2 contract") from exc


def _decode_row(
    row: torch.Tensor,
    *,
    producer: Literal["birth", "turnover"],
    cell_ordinal: int,
    cell_id: str,
    ordinal: int,
    capability_contract_sha256: str,
    package_id_sha256: str,
    artifact_sha256: str,
    execution_route_sha256: str,
) -> GenomeV2CudaCausalDecodedRowV1:
    event_code, event_kind = _EVENT_KINDS[producer]
    values = tuple(int(value) for value in row.tolist())
    if values[_COLUMN_ORDINAL] != ordinal:
        raise _fail("causal record ordinal contradicts its valid prefix")
    if values[_COLUMN_EVENT_KIND] != event_code:
        raise _fail("causal event kind contradicts its producer")
    if values[_COLUMN_CELL] != cell_ordinal:
        raise _fail("causal cell ordinal contradicts its tensor row")
    if values[_COLUMN_EVENT_SEQUENCE_VALID] != 1:
        raise _fail("causal event identity has no valid sequence")
    if values[_COLUMN_REGULATOR_VALUES_VALID] != 0:
        raise _fail("packed ABI cannot carry claimed regulator node values")
    context_valid = values[_COLUMN_CONTEXT_VALID]
    if context_valid not in {0, 1}:
        raise _fail("context validity flag is invalid")
    regulatory_cost_valid = values[_COLUMN_REGULATORY_COST_VALID]
    if regulatory_cost_valid not in {0, 1}:
        raise _fail("regulatory-cost validity flag is invalid")

    try:
        identity = GenomeV2CausalTransactionIdentityV1.create(
            event_kind=event_kind,
            event_sequence=values[_COLUMN_EVENT_SEQUENCE],
            child_id=values[_COLUMN_CHILD_ID],
            cell_id=cell_id,
            generation=values[_COLUMN_GENERATION],
            tick=values[_COLUMN_TICK],
            destination_slot=values[_COLUMN_DESTINATION_SLOT],
            parent_slot=values[_COLUMN_PARENT_SLOT],
            mate_parent_slot=values[_COLUMN_MATE_PARENT_SLOT],
            parent_id=values[_COLUMN_PARENT_ID],
            mate_parent_id=values[_COLUMN_MATE_PARENT_ID],
            lineage_founder_id=values[_COLUMN_FOUNDER_ID],
            lineage_depth=values[_COLUMN_LINEAGE_DEPTH],
            capability_contract_sha256=capability_contract_sha256,
            package_id_sha256=package_id_sha256,
            artifact_sha256=artifact_sha256,
            execution_route_sha256=execution_route_sha256,
        )
    except ValueError as exc:
        raise _fail("causal event identity is invalid") from exc

    genome = _genome(row)
    products = tuple(
        values[_COLUMN_PRODUCTS + index] for index in range(PRODUCT_COUNT)
    )
    phenotypes = tuple(
        values[_COLUMN_PHENOTYPES + index] for index in range(PHENOTYPE_COUNT)
    )
    expression_cost = values[_COLUMN_EXPRESSION_COST]
    runtime_values = tuple(
        values[_COLUMN_RUNTIME + index]
        for index in range(RUNTIME_PROJECTION_COUNT)
    )
    committed_runtime = _runtime_projection(runtime_values)

    if context_valid == 0:
        if any(
            values[_COLUMN_CONTEXT + index] != UNAVAILABLE_VALUE
            for index in range(CONTEXT_COUNT)
        ):
            raise _fail("unavailable context contains claimed values")
        context = None
        expression = None
        integrity_verified = False
        reason = "frozen_event_context_unavailable"
    else:
        context = tuple(
            values[_COLUMN_CONTEXT + index] for index in range(CONTEXT_COUNT)
        )
        try:
            expression = express_reference_v1(genome, context)
        except ValueError as exc:
            raise _fail("frozen event context violates the expression contract") from exc
        if products != expression.products_q10:
            raise _fail("committed products contradict reference expression")
        if phenotypes != expression.phenotypes_q10:
            raise _fail("committed phenotypes contradict reference expression")
        if expression_cost != expression.expression_cost_q10:
            raise _fail("committed expression cost contradicts reference expression")
        projected = project_runtime_traits_v1(expression)
        if committed_runtime != projected:
            raise _fail("committed runtime projection contradicts reference expression")
        if (
            regulatory_cost_valid == 1
            and values[_COLUMN_REGULATORY_COST]
            != expression.regulatory_cost_q10
        ):
            raise _fail("committed regulatory cost contradicts reference expression")
        integrity_verified = True
        reason = None

    if (
        regulatory_cost_valid == 0
        and values[_COLUMN_REGULATORY_COST] != UNAVAILABLE_VALUE
    ):
        raise _fail("unavailable regulatory cost contains a claimed value")
    return GenomeV2CudaCausalDecodedRowV1(
        identity=identity,
        record_ordinal=ordinal,
        producer=producer,
        genome=genome,
        committed_products_q10=products,
        committed_phenotypes_q10=phenotypes,
        committed_expression_cost_q10=expression_cost,
        committed_runtime_projection=committed_runtime,
        context_q10=context,
        recomputed_expression=expression,
        integrity_verified=integrity_verified,
        non_evaluable_reason=reason,
    )


def decode_genome_v2_causal_cuda_outputs_v1(
    outputs: Mapping[str, Any],
    *,
    producer: Literal["birth", "turnover"],
    cell_ids: Sequence[str],
    physical_cell_ordinals: Sequence[int] | None = None,
    capability_contract_sha256: str,
    package_id_sha256: str,
    artifact_sha256: str,
    execution_route_sha256: str,
) -> GenomeV2CudaCausalDecodeResultV1:
    """Decode one bounded producer output group and verify every evaluable row."""
    if producer not in _EVENT_KINDS:
        raise _fail("producer must be birth or turnover")
    if not isinstance(outputs, Mapping):
        raise _fail("outputs must be a mapping")
    prefix = f"{producer}_genome_v2_causal"
    rows_value = outputs.get(f"{prefix}_record_rows")
    if not isinstance(rows_value, torch.Tensor) or rows_value.ndim != 3:
        raise _fail(f"{prefix}_record_rows must be a rank-3 torch tensor")
    cells = int(rows_value.shape[0])
    ids = _exact_cell_ids(cell_ids, cells=cells)
    physical_ordinals = _physical_cell_ordinals(
        physical_cell_ordinals,
        cells=cells,
    )
    rows = _tensor(
        outputs,
        f"{prefix}_record_rows",
        dtype=torch.int64,
        shape=(cells, RECORD_CAPACITY, RECORD_WIDTH),
    )
    valid = _tensor(
        outputs,
        f"{prefix}_record_valid",
        dtype=torch.bool,
        shape=(cells, RECORD_CAPACITY),
    )
    count = _tensor(
        outputs,
        f"{prefix}_record_count",
        dtype=torch.int64,
        shape=(cells,),
    )
    dropped = _tensor(
        outputs,
        f"{prefix}_record_dropped_count",
        dtype=torch.int64,
        shape=(cells,),
    )
    capture_enabled = _tensor(
        outputs,
        f"{prefix}_capture_enabled",
        dtype=torch.bool,
        shape=(cells,),
    )

    decoded_cells: list[GenomeV2CudaCausalCellDecodeV1] = []
    natural_identities: set[tuple[Any, ...]] = set()
    for cell in range(cells):
        captured = int(count[cell])
        omitted = int(dropped[cell])
        if captured < 0 or captured > RECORD_CAPACITY:
            raise _fail("causal captured count is outside its fixed capacity")
        if omitted < 0:
            raise _fail("causal dropped count is negative")
        enabled = bool(capture_enabled[cell])
        if not enabled and (captured != 0 or omitted != 0):
            raise _fail("disabled causal capture contains claimed records")
        expected_valid = torch.arange(RECORD_CAPACITY) < captured
        if not torch.equal(valid[cell], expected_valid):
            raise _fail("causal valid mask is not the exact count prefix")
        decoded_rows: list[GenomeV2CudaCausalDecodedRowV1] = []
        reasons: list[str] = []
        if not enabled:
            reasons.append("causal_capture_disabled")
        elif captured == 0:
            reasons.append("no_causal_rows_captured")
        if omitted:
            reasons.append("causal_rows_dropped")
        for ordinal in range(captured):
            decoded = _decode_row(
                rows[cell, ordinal],
                producer=producer,
                cell_ordinal=physical_ordinals[cell],
                cell_id=ids[cell],
                ordinal=ordinal,
                capability_contract_sha256=capability_contract_sha256,
                package_id_sha256=package_id_sha256,
                artifact_sha256=artifact_sha256,
                execution_route_sha256=execution_route_sha256,
            )
            natural_identity = (
                decoded.identity.event_kind,
                decoded.identity.event_sequence,
                decoded.identity.child_id,
                decoded.identity.cell_id,
                decoded.identity.generation,
                decoded.identity.tick,
                decoded.identity.destination_slot,
            )
            if natural_identity in natural_identities:
                raise _fail("causal output contains an ambiguous event identity")
            natural_identities.add(natural_identity)
            if decoded.non_evaluable_reason is not None:
                reasons.append(decoded.non_evaluable_reason)
            decoded_rows.append(decoded)
        unique_reasons = tuple(dict.fromkeys(reasons))
        decoded_cells.append(
            GenomeV2CudaCausalCellDecodeV1(
                cell_ordinal=physical_ordinals[cell],
                cell_id=ids[cell],
                captured_count=captured,
                dropped_count=omitted,
                rows=tuple(decoded_rows),
                evaluable=not unique_reasons,
                non_evaluable_reasons=unique_reasons,
            )
        )
    return GenomeV2CudaCausalDecodeResultV1(
        schema_version=SCHEMA_VERSION,
        producer=producer,
        cells=tuple(decoded_cells),
    )


__all__ = [
    "GenomeV2CausalCudaOutputError",
    "GenomeV2CudaCausalCellDecodeV1",
    "GenomeV2CudaCausalDecodeResultV1",
    "GenomeV2CudaCausalDecodedRowV1",
    "RECORD_CAPACITY",
    "RECORD_WIDTH",
    "SCHEMA_VERSION",
    "UNAVAILABLE_VALUE",
    "decode_genome_v2_causal_cuda_outputs_v1",
]
