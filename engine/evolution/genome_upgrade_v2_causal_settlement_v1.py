"""Bounded event-time to terminal projection for Genome V2 CUDA records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any, Final

import torch

from engine.authoritative_state.schema import content_sha256

from .genome_upgrade_v2 import (
    CONTEXT_COUNT,
    GenomeUpgradeV2ExpressionV1,
    GenomeUpgradeV2V1,
)
from .genome_upgrade_v2_causal_cuda_output_v1 import (
    RECORD_CAPACITY,
    RECORD_WIDTH,
    GenomeV2CudaCausalDecodedRowV1,
    decode_genome_v2_causal_cuda_outputs_v1,
)
from .genome_upgrade_v2_causal_transaction_projection_v1 import (
    GenomeV2CausalTransactionRecordV1,
    join_causal_transaction_records_v1,
)
from .genome_upgrade_v2_package_v1 import (
    validate_checkpoint_authority_binding_v1,
)
from engine.perf.cuda_genome_upgrade_v2_state_bank import (
    CudaGenomeUpgradeV2StateBank,
)


SCHEMA_VERSION: Final[str] = "genome_v2_causal_settlement_projection_v2"
EVENT_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "genome_v2_causal_event_receipt_v2"
)
TERMINAL_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "genome_v2_causal_terminal_receipt_v2"
)
JOIN_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "genome_v2_causal_join_receipt_v2"
)
MAX_PRODUCER_RECORDS: Final[int] = RECORD_CAPACITY
MAX_JOINED_RECORDS: Final[int] = 2 * RECORD_CAPACITY
NEUTRAL_CONTEXT_Q10: Final[tuple[int, ...]] = (0,) * CONTEXT_COUNT

def contract_payload_v1() -> dict[str, Any]:
    """Return the canonical package-bindable settlement contract."""
    return {
        "schema_version": SCHEMA_VERSION,
        "event_receipt_schema_version": EVENT_RECEIPT_SCHEMA_VERSION,
        "terminal_receipt_schema_version": TERMINAL_RECEIPT_SCHEMA_VERSION,
        "join_receipt_schema_version": JOIN_RECEIPT_SCHEMA_VERSION,
        "producer_record_capacity": MAX_PRODUCER_RECORDS,
        "producer_record_width": RECORD_WIDTH,
        "joined_record_capacity": MAX_JOINED_RECORDS,
        "projection_fields": [
            "schema_version",
            "contract_sha256",
            "cell_id",
            "physical_cell_ordinal",
            "capture_status",
            "checkpoint_authority_binding",
            "captured_counts",
            "dropped_counts",
            "record_counts",
            "event_receipts",
            "terminal_receipts",
            "join_receipts",
            "evaluable",
            "reasons",
            "projection_sha256",
        ],
        "transaction_receipt_fields": [
            "schema_version",
            "transaction_record",
            "receipt_sha256",
        ],
        "join_receipt_fields": [
            "schema_version",
            "transaction_id",
            "event_receipt_sha256",
            "terminal_receipt_sha256",
            "receipt_sha256",
        ],
        "capture_statuses": [
            "capture_disabled",
            "capture_enabled_no_records",
            "capture_enabled_not_evaluable",
            "capture_enabled_joined",
        ],
        "cross_producer_natural_identity": [
            "cell_id",
            "event_sequence",
            "child_id",
            "generation",
            "tick",
            "destination_slot",
        ],
        "evaluable_requirements": [
            "exact_checkpoint_authority_binding",
            "frozen_neutral_event_context",
            "zero_dropped_records",
            "verified_event_integrity",
            "two_distinct_parent_ids",
            "one_exact_live_terminal_slot_per_child_id",
            "complete_nonempty_strict_join",
        ],
        "default_off_authority_binding": None,
        "terminal_slot_authority": "live_slot_to_agent_id_intersect_alive_mask",
        "terminal_context_role": (
            "terminal_observation_not_retained_applied_context"
        ),
    }


CONTRACT_PAYLOAD: Final[dict[str, Any]] = contract_payload_v1()
CONTRACT_SHA256: Final[str] = content_sha256(CONTRACT_PAYLOAD)

_PRODUCERS: Final[tuple[str, ...]] = ("birth", "turnover")


class GenomeV2CausalSettlementError(ValueError):
    """Causal settlement inputs are structurally malformed or contradictory."""


def _fail(message: str) -> GenomeV2CausalSettlementError:
    return GenomeV2CausalSettlementError(message)


def _tensor(
    value: Any,
    *,
    name: str,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    device: torch.device | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise _fail(f"{name} must be a torch tensor")
    if value.dtype != dtype or tuple(value.shape) != shape:
        raise _fail(f"{name} has the wrong fixed tensor contract")
    if not value.is_contiguous():
        raise _fail(f"{name} must be contiguous")
    if device is not None and value.device != device:
        raise _fail(f"{name} is on a different device")
    return value


def _capture_enabled(
    outputs: Mapping[str, Any],
    *,
    producer: str,
) -> bool:
    prefix = f"{producer}_genome_v2_causal"
    enabled = _tensor(
        outputs.get(f"{prefix}_capture_enabled"),
        name=f"{prefix}_capture_enabled",
        dtype=torch.bool,
        shape=(1,),
    )
    return bool(enabled.detach().cpu().item())


def _validate_disabled_outputs(
    outputs: Mapping[str, Any],
    *,
    producer: str,
) -> None:
    prefix = f"{producer}_genome_v2_causal"
    rows = _tensor(
        outputs.get(f"{prefix}_record_rows"),
        name=f"{prefix}_record_rows",
        dtype=torch.int64,
        shape=(1, RECORD_CAPACITY, RECORD_WIDTH),
    )
    valid = _tensor(
        outputs.get(f"{prefix}_record_valid"),
        name=f"{prefix}_record_valid",
        dtype=torch.bool,
        shape=(1, RECORD_CAPACITY),
    )
    count = _tensor(
        outputs.get(f"{prefix}_record_count"),
        name=f"{prefix}_record_count",
        dtype=torch.int64,
        shape=(1,),
    )
    dropped = _tensor(
        outputs.get(f"{prefix}_record_dropped_count"),
        name=f"{prefix}_record_dropped_count",
        dtype=torch.int64,
        shape=(1,),
    )
    if valid.any() or int(count.item()) != 0 or int(dropped.item()) != 0:
        raise _fail("disabled causal capture contains claimed records")
    # Touch the fixed row allocation so the disabled path validates the full ABI.
    if rows.numel() != RECORD_CAPACITY * RECORD_WIDTH:
        raise _fail("disabled causal capture row allocation differs")


def _inputs(
    *,
    persistent_state: Mapping[str, Any] | None,
    causal_outputs: Mapping[str, Any] | None,
    slot_to_agent_id: torch.Tensor | None,
) -> tuple[Mapping[str, Any], torch.Tensor]:
    if persistent_state is not None:
        if causal_outputs is not None or slot_to_agent_id is not None:
            raise _fail(
                "persistent_state and explicit causal tensors are mutually exclusive"
            )
        if not isinstance(persistent_state, Mapping):
            raise _fail("persistent_state must be a mapping")
        output_mapping = persistent_state
        resolved_slots = persistent_state.get("slot_to_agent_id")
    else:
        if not isinstance(causal_outputs, Mapping):
            raise _fail("explicit causal_outputs must be a mapping")
        output_mapping = causal_outputs
        resolved_slots = slot_to_agent_id
    if not isinstance(resolved_slots, torch.Tensor):
        raise _fail("slot_to_agent_id is absent")
    return output_mapping, resolved_slots


def _genotype_summary(genome: GenomeUpgradeV2V1) -> dict[str, Any]:
    return {
        "alleles_q10": genome.alleles,
        "modifiers_q10": genome.modifiers,
    }


def _expression_summaries(
    expression: GenomeUpgradeV2ExpressionV1,
    *,
    context_q10: tuple[int, ...],
    context_role: str,
) -> dict[str, dict[str, Any]]:
    return {
        "product_summary": {
            "products_q10": expression.products_q10,
            "active_product_count": expression.active_product_count,
            "saturated_product_count": expression.saturated_product_count,
        },
        "regulatory_summary": {
            "source": "reference_recomputation",
            "regulator_nodes_q10": expression.regulator_nodes_q10,
            "epistatic_delta_q10": expression.epistatic_delta_q10,
            "active_regulator_count": expression.active_regulator_count,
            "active_regulatory_edge_count": (
                expression.active_regulatory_edge_count
            ),
            "developmental_stage": expression.developmental_stage,
        },
        "context_summary": {
            "available": True,
            "role": context_role,
            "context_q10": context_q10,
        },
        "phenotype_summary": {
            "phenotypes_q10": expression.phenotypes_q10,
        },
        "cost_summary": {
            "expression_cost_q10": expression.expression_cost_q10,
            "regulatory_cost_q10": expression.regulatory_cost_q10,
        },
    }


def _event_record(
    row: GenomeV2CudaCausalDecodedRowV1,
) -> GenomeV2CausalTransactionRecordV1 | None:
    expression = row.recomputed_expression
    if not row.integrity_verified or expression is None or row.context_q10 is None:
        return None
    if row.context_q10 != NEUTRAL_CONTEXT_Q10:
        raise _fail("causal event context is not the frozen neutral context")
    summaries = _expression_summaries(
        expression,
        context_q10=row.context_q10,
        context_role="frozen_event_applied_context",
    )
    return GenomeV2CausalTransactionRecordV1.create(
        record_phase="event_time",
        identity=row.identity,
        genotype_summary=_genotype_summary(row.genome),
        **summaries,
        applied_consequence={
            "source": "committed_cuda_runtime_projection",
            **asdict(row.committed_runtime_projection),
        },
        resulting_state={
            "available": False,
            "reason": "event_time_live_state_not_in_bounded_cuda_record",
            "committed_destination_slot": row.identity.destination_slot,
        },
    )


def _terminal_record(
    row: GenomeV2CudaCausalDecodedRowV1,
    *,
    bank: CudaGenomeUpgradeV2StateBank,
    terminal_context_q10: torch.Tensor,
    slot: int | None,
    retained_agent_id: int | None = None,
    retention_relation: str = "direct_child_survival",
) -> GenomeV2CausalTransactionRecordV1:
    if retention_relation not in {
        "direct_child_survival",
        "event_ledger_descendant_retention",
        "terminal_loss",
    }:
        raise _fail("retention_relation is not recognized")
    retained = row.identity.child_id if retained_agent_id is None else retained_agent_id
    if slot is None:
        if retention_relation != "terminal_loss":
            raise _fail("retained terminal records require a slot")
        genome = row.genome
        context = NEUTRAL_CONTEXT_Q10
        committed_products = tuple(row.committed_products_q10)
        committed_phenotypes = tuple(row.committed_phenotypes_q10)
        product_summary = {
            "source": "event_time_record",
            "products_q10": committed_products,
            "retention": "not_live_at_terminal",
        }
        regulatory_summary = {
            "available": False,
            "reason": "terminal_lineage_not_retained",
        }
        phenotype_summary = {
            "source": "event_time_record",
            "phenotypes_q10": committed_phenotypes,
            "retention": "not_live_at_terminal",
        }
        cost_summary = {
            "expression_cost_q10": row.committed_expression_cost_q10,
            "regulatory_cost_available": False,
            "reason": "terminal_lineage_not_retained",
        }
    else:
        genome = bank.genotype(0, slot)
        context = tuple(
            int(value)
            for value in terminal_context_q10[0, slot].detach().cpu().tolist()
        )
        committed_products = tuple(
            int(value)
            for value in bank.products_q10[0, :, slot].detach().cpu().tolist()
        )
        committed_phenotypes = tuple(
            int(value)
            for value in bank.phenotypes_q10[0, :, slot].detach().cpu().tolist()
        )
        product_summary = {
            "source": "terminal_committed_bank",
            "products_q10": committed_products,
            "active_product_count": int(
                bank.active_product_count[0, slot].item()
            ),
            "saturated_product_count": int(
                bank.saturated_product_count[0, slot].item()
            ),
        }
        regulatory_summary = {
            "available": False,
            "reason": "terminal_applied_context_not_retained",
        }
        phenotype_summary = {
            "source": "terminal_committed_bank",
            "phenotypes_q10": committed_phenotypes,
        }
        cost_summary = {
            "expression_cost_q10": int(
                bank.expression_cost_q10[0, slot].item()
            ),
            "regulatory_cost_available": False,
            "reason": "terminal_applied_context_not_retained",
        }
    return GenomeV2CausalTransactionRecordV1.create(
        record_phase="terminal",
        identity=row.identity,
        genotype_summary=_genotype_summary(genome),
        product_summary=product_summary,
        regulatory_summary=regulatory_summary,
        context_summary={
            "available": True,
            "role": "terminal_observation_context",
            "context_q10": context,
            "expression_binding": "not_retained_applied_context",
        },
        phenotype_summary=phenotype_summary,
        cost_summary=cost_summary,
        applied_consequence={
            "available": False,
            "reason": (
                "terminal_observation_context_is_not_the_retained_applied_context"
            ),
            "event_transaction_id": row.identity.transaction_id,
        },
        resulting_state={
            "survived_to_terminal": slot is not None,
            "retention_relation": retention_relation,
            "terminal_slot": slot,
            "agent_id": retained,
            "event_child_id": row.identity.child_id,
            "located_by": (
                "slot_to_agent_id_intersect_alive_mask"
                if retention_relation == "direct_child_survival"
                else (
                    "causal_event_parent_graph_to_live_descendant"
                    if retention_relation == "event_ledger_descendant_retention"
                    else "causal_event_parent_graph_no_live_descendant"
                )
            ),
        },
    )


def _receipt(record: GenomeV2CausalTransactionRecordV1, *, schema: str) -> dict[str, Any]:
    payload = {
        "schema_version": schema,
        "transaction_record": record.as_dict(),
    }
    return {**payload, "receipt_sha256": content_sha256(payload)}


def _base_projection(
    *,
    cell_id: str,
    physical_cell_ordinal: int,
    capture_status: str,
    authority_binding: Mapping[str, str] | None,
    captured_counts: Mapping[str, int],
    dropped_counts: Mapping[str, int],
    event_receipts: list[dict[str, Any]],
    terminal_receipts: list[dict[str, Any]],
    join_receipts: list[dict[str, Any]],
    evaluable: bool,
    reasons: list[str],
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "cell_id": cell_id,
        "physical_cell_ordinal": physical_cell_ordinal,
        "capture_status": capture_status,
        "checkpoint_authority_binding": (
            dict(authority_binding) if authority_binding is not None else None
        ),
        "captured_counts": dict(captured_counts),
        "dropped_counts": dict(dropped_counts),
        "record_counts": {
            "event": len(event_receipts),
            "terminal": len(terminal_receipts),
            "joined": len(join_receipts),
        },
        "event_receipts": event_receipts,
        "terminal_receipts": terminal_receipts,
        "join_receipts": join_receipts,
        "evaluable": evaluable,
        "reasons": list(dict.fromkeys(reasons)),
    }
    return {**payload, "projection_sha256": content_sha256(payload)}


def _live_slot_for_agent_id(
    *,
    alive: torch.Tensor,
    slots: torch.Tensor,
    agent_id: int,
) -> int | None:
    matches = torch.nonzero(
        alive[0] & slots[0].eq(agent_id),
        as_tuple=False,
    ).flatten()
    if int(matches.numel()) == 0:
        return None
    if int(matches.numel()) > 1:
        raise _fail(f"terminal_child_ambiguous:{agent_id}")
    return int(matches.item())


def _terminal_location_for_event_child(
    row: GenomeV2CudaCausalDecodedRowV1,
    *,
    rows: tuple[GenomeV2CudaCausalDecodedRowV1, ...],
    alive: torch.Tensor,
    slots: torch.Tensor,
) -> tuple[int | None, int | None, str]:
    direct = _live_slot_for_agent_id(
        alive=alive,
        slots=slots,
        agent_id=row.identity.child_id,
    )
    if direct is not None:
        return direct, row.identity.child_id, "direct_child_survival"

    children_by_parent: dict[int, set[int]] = {}
    for candidate in rows:
        if candidate.identity.event_sequence <= row.identity.event_sequence:
            continue
        children_by_parent.setdefault(candidate.identity.parent_id, set()).add(
            candidate.identity.child_id
        )
        children_by_parent.setdefault(candidate.identity.mate_parent_id, set()).add(
            candidate.identity.child_id
        )

    visited = {row.identity.child_id}
    frontier = [row.identity.child_id]
    while frontier:
        current = frontier.pop(0)
        for child_id in sorted(children_by_parent.get(current, ())):
            if child_id in visited:
                continue
            visited.add(child_id)
            slot = _live_slot_for_agent_id(
                alive=alive,
                slots=slots,
                agent_id=child_id,
            )
            if slot is not None:
                return slot, child_id, "event_ledger_descendant_retention"
            frontier.append(child_id)
    return None, None, "terminal_loss"


def project_genome_v2_causal_settlement_v1(
    *,
    bank: CudaGenomeUpgradeV2StateBank,
    alive_mask: torch.Tensor,
    terminal_context_q10: torch.Tensor,
    cell_id: str,
    physical_cell_ordinal: int,
    checkpoint_authority_binding: Any = None,
    persistent_state: Mapping[str, Any] | None = None,
    causal_outputs: Mapping[str, Any] | None = None,
    slot_to_agent_id: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Build one content-addressed bounded causal settlement projection."""
    if (
        getattr(bank, "cells", None) != 1
        or type(getattr(bank, "capacity", None)) is not int
    ):
        raise _fail("bank must be a one-cell Genome V2 state bank")
    if type(cell_id) is not str or not cell_id or cell_id.strip() != cell_id:
        raise _fail("cell_id must be one stable canonical string")
    if type(physical_cell_ordinal) is not int or physical_cell_ordinal < 0:
        raise _fail("physical_cell_ordinal must be a non-negative integer")
    alive = _tensor(
        alive_mask,
        name="alive_mask",
        dtype=torch.bool,
        shape=(1, bank.capacity),
        device=bank.device,
    )
    terminal_context = _tensor(
        terminal_context_q10,
        name="terminal_context_q10",
        dtype=torch.int64,
        shape=(1, bank.capacity, CONTEXT_COUNT),
        device=bank.device,
    )
    outputs, raw_slots = _inputs(
        persistent_state=persistent_state,
        causal_outputs=causal_outputs,
        slot_to_agent_id=slot_to_agent_id,
    )
    slots = _tensor(
        raw_slots,
        name="slot_to_agent_id",
        dtype=torch.int64,
        shape=(1, bank.capacity),
        device=bank.device,
    )

    enabled = {
        producer: _capture_enabled(outputs, producer=producer)
        for producer in _PRODUCERS
    }
    if len(set(enabled.values())) != 1:
        raise _fail("birth and turnover causal capture modes contradict")
    if not enabled["birth"]:
        for producer in _PRODUCERS:
            _validate_disabled_outputs(outputs, producer=producer)
        return _base_projection(
            cell_id=cell_id,
            physical_cell_ordinal=physical_cell_ordinal,
            capture_status="capture_disabled",
            authority_binding=None,
            captured_counts={"birth": 0, "turnover": 0, "total": 0},
            dropped_counts={"birth": 0, "turnover": 0, "total": 0},
            event_receipts=[],
            terminal_receipts=[],
            join_receipts=[],
            evaluable=False,
            reasons=["causal_capture_disabled"],
        )

    try:
        binding = validate_checkpoint_authority_binding_v1(
            checkpoint_authority_binding
        )
    except ValueError as exc:
        raise _fail("checkpoint authority binding is invalid") from exc
    digest_arguments = {
        "capability_contract_sha256": binding["capability_contract_sha256"],
        "package_id_sha256": binding["package_id_sha256"],
        "artifact_sha256": binding["artifact_sha256"],
        "execution_route_sha256": binding["execution_route_sha256"],
    }
    decoded = {
        producer: decode_genome_v2_causal_cuda_outputs_v1(
            outputs,
            producer=producer,  # type: ignore[arg-type]
            cell_ids=(cell_id,),
            physical_cell_ordinals=(physical_cell_ordinal,),
            **digest_arguments,
        )
        for producer in _PRODUCERS
    }
    all_rows = tuple(
        row for producer in _PRODUCERS for row in decoded[producer].rows
    )
    natural_identities: set[tuple[Any, ...]] = set()
    for row in all_rows:
        identity = row.identity
        natural = (
            identity.cell_id,
            identity.event_sequence,
            identity.child_id,
            identity.generation,
            identity.tick,
            identity.destination_slot,
        )
        if natural in natural_identities:
            raise _fail("birth and turnover contain a duplicate natural identity")
        natural_identities.add(natural)

    captured_counts = {
        producer: decoded[producer].captured_count for producer in _PRODUCERS
    }
    captured_counts["total"] = sum(captured_counts.values())
    dropped_counts = {
        producer: decoded[producer].dropped_count for producer in _PRODUCERS
    }
    dropped_counts["total"] = sum(dropped_counts.values())
    reasons: list[str] = []
    if captured_counts["total"] == 0:
        reasons.append("no_causal_rows_captured")
    if dropped_counts["total"]:
        reasons.append("causal_rows_dropped")

    event_records: list[GenomeV2CausalTransactionRecordV1] = []
    terminal_records: list[GenomeV2CausalTransactionRecordV1] = []
    retained_terminal_count = 0
    descendant_terminal_count = 0
    loss_terminal_count = 0
    for row in all_rows:
        event_record = _event_record(row)
        if event_record is None:
            reasons.append(
                row.non_evaluable_reason or "event_integrity_not_verified"
            )
            continue
        event_records.append(event_record)
        try:
            terminal_slot, retained_agent_id, relation = (
                _terminal_location_for_event_child(
                    row,
                    rows=all_rows,
                    alive=alive,
                    slots=slots,
                )
            )
        except GenomeV2CausalSettlementError as exc:
            reasons.append(f"{exc}:{row.identity.transaction_id}")
            continue
        if terminal_slot is None:
            loss_terminal_count += 1
        else:
            retained_terminal_count += 1
            if relation == "event_ledger_descendant_retention":
                descendant_terminal_count += 1
        terminal_records.append(
            _terminal_record(
                row,
                bank=bank,
                terminal_context_q10=terminal_context,
                slot=terminal_slot,
                retained_agent_id=retained_agent_id,
                retention_relation=relation,
            )
        )

    joins = ()
    complete = (
        captured_counts["total"] > 0
        and not reasons
        and len(event_records) == captured_counts["total"]
        and len(terminal_records) == captured_counts["total"]
    )
    if complete and retained_terminal_count == 0:
        reasons.append("no_terminal_retention")
        complete = False
    if complete:
        joins = join_causal_transaction_records_v1(
            event_records,
            terminal_records,
        )
    event_receipts = [
        _receipt(record, schema=EVENT_RECEIPT_SCHEMA_VERSION)
        for record in event_records
    ]
    terminal_receipts = [
        _receipt(record, schema=TERMINAL_RECEIPT_SCHEMA_VERSION)
        for record in terminal_records
    ]
    event_by_id = {
        record.identity.transaction_id: receipt
        for record, receipt in zip(event_records, event_receipts, strict=True)
    }
    terminal_by_id = {
        record.identity.transaction_id: receipt
        for record, receipt in zip(
            terminal_records, terminal_receipts, strict=True
        )
    }
    join_receipts: list[dict[str, Any]] = []
    for joined in joins:
        payload = {
            "schema_version": JOIN_RECEIPT_SCHEMA_VERSION,
            "transaction_id": joined.transaction_id,
            "event_receipt_sha256": event_by_id[joined.transaction_id][
                "receipt_sha256"
            ],
            "terminal_receipt_sha256": terminal_by_id[joined.transaction_id][
                "receipt_sha256"
            ],
            "retention_relation": joined.terminal.resulting_state[
                "retention_relation"
            ],
            "terminal_agent_id": joined.terminal.resulting_state["agent_id"],
        }
        join_receipts.append(
            {**payload, "receipt_sha256": content_sha256(payload)}
        )
    projection = _base_projection(
        cell_id=cell_id,
        physical_cell_ordinal=physical_cell_ordinal,
        capture_status=(
            "capture_enabled_joined"
            if complete
            else (
                "capture_enabled_no_records"
                if captured_counts["total"] == 0
                else "capture_enabled_not_evaluable"
            )
        ),
        authority_binding=binding,
        captured_counts=captured_counts,
        dropped_counts=dropped_counts,
        event_receipts=event_receipts,
        terminal_receipts=terminal_receipts,
        join_receipts=join_receipts,
        evaluable=complete,
        reasons=reasons,
    )
    projection["retention_counts"] = {
        "direct_child_survival": retained_terminal_count - descendant_terminal_count,
        "event_ledger_descendant_retention": descendant_terminal_count,
        "terminal_loss": loss_terminal_count,
        "retained_total": retained_terminal_count,
    }
    payload = dict(projection)
    payload.pop("projection_sha256")
    projection["projection_sha256"] = content_sha256(payload)
    return projection


__all__ = [
    "CONTRACT_PAYLOAD",
    "CONTRACT_SHA256",
    "EVENT_RECEIPT_SCHEMA_VERSION",
    "GenomeV2CausalSettlementError",
    "JOIN_RECEIPT_SCHEMA_VERSION",
    "MAX_JOINED_RECORDS",
    "MAX_PRODUCER_RECORDS",
    "SCHEMA_VERSION",
    "TERMINAL_RECEIPT_SCHEMA_VERSION",
    "contract_payload_v1",
    "project_genome_v2_causal_settlement_v1",
]
