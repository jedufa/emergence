from __future__ import annotations

from dataclasses import astuple

import pytest
import torch

from engine.authoritative_state.schema import content_sha256
from engine.evolution.genome_upgrade_v2 import (
    GenomeUpgradeV2V1,
    express_reference_v1,
    project_runtime_traits_v1,
)
from engine.evolution.genome_upgrade_v2_causal_cuda_output_v1 import (
    RECORD_CAPACITY,
    RECORD_WIDTH,
    UNAVAILABLE_VALUE,
)
from engine.evolution.genome_upgrade_v2_causal_settlement_v1 import (
    CONTRACT_PAYLOAD,
    CONTRACT_SHA256,
    SCHEMA_VERSION,
    GenomeV2CausalSettlementError,
    project_genome_v2_causal_settlement_v1,
)
from engine.perf.cuda_genome_upgrade_v2_state_bank import (
    CudaGenomeUpgradeV2StateBank,
)


def _authority_binding() -> dict[str, str]:
    binding = {
        "schema_version": "genome_v2_checkpoint_authority_binding_v1",
        "provider_id": "eiel-production-provider-v1",
        "runtime_profile_id": "eiel-production-runtime-v1",
        "global_genome_profile_id": "genome_upgrade_v2",
        "source_commit": "1" * 40,
    }
    for index, name in enumerate(
        (
            "executable_package_core_sha256",
            "package_id_sha256",
            "package_receipt_sha256",
            "artifact_sha256",
            "manifest_sha256",
            "state_schema_sha256",
            "mechanic_contract_sha256",
            "capability_contract_sha256",
            "cancellation_abi_sha256",
            "observability_schema_sha256",
            "capacity_layout_contract_sha256",
            "execution_route_sha256",
        ),
        start=2,
    ):
        binding[name] = f"{index:x}"[-1] * 64
    return binding


def _producer_outputs(
    producer: str,
    *,
    enabled: bool,
    child_id: int = 17,
    event_sequence: int = 9,
    destination_slot: int = 1,
    physical_cell_ordinal: int = 5,
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
    count = torch.zeros((1,), dtype=torch.int64)
    if enabled:
        genome = GenomeUpgradeV2V1.neutral()
        expression = express_reference_v1(genome, (0,) * 8)
        runtime = project_runtime_traits_v1(expression)
        row = rows[0, 0]
        row[0] = 0
        row[1] = 1 if producer == "birth" else 2
        row[2] = physical_cell_ordinal
        row[3] = 2
        row[4] = 31
        row[5] = event_sequence
        row[6] = 1
        row[7] = child_id
        row[8] = parent_id
        row[9] = destination_slot
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
        row[132:144] = torch.tensor(expression.products_q10)
        row[144] = 0
        row[145:155] = torch.tensor(expression.phenotypes_q10)
        row[155] = 1
        row[156:164] = 0
        row[164] = expression.expression_cost_q10
        row[165] = 1
        row[166] = expression.regulatory_cost_q10
        row[167:179] = torch.tensor(astuple(runtime))
        row[179] = mate_parent_id
        row[180] = parent_slot
        row[181] = mate_parent_slot
        valid[0, 0] = True
        count[0] = 1
    return {
        f"{prefix}_record_rows": rows,
        f"{prefix}_record_valid": valid,
        f"{prefix}_record_count": count,
        f"{prefix}_record_dropped_count": torch.tensor(
            [dropped], dtype=torch.int64
        ),
        f"{prefix}_capture_enabled": torch.tensor(
            [enabled], dtype=torch.bool
        ),
    }


def _outputs(
    *,
    capture_enabled: bool = True,
    birth_child: int = 17,
    birth_sequence: int = 9,
    birth_destination: int = 1,
    turnover_record: bool = False,
    turnover_child: int = 18,
    turnover_sequence: int = 10,
    turnover_destination: int = 2,
    birth_mate_parent: int = 5,
    turnover_mate_parent: int = 7,
    dropped: int = 0,
) -> dict[str, torch.Tensor]:
    return {
        **_producer_outputs(
            "birth",
            enabled=capture_enabled,
            child_id=birth_child,
            event_sequence=birth_sequence,
            destination_slot=birth_destination,
            mate_parent_id=birth_mate_parent,
            dropped=dropped,
        ),
        **_producer_outputs(
            "turnover",
            enabled=capture_enabled and turnover_record,
            child_id=turnover_child,
            event_sequence=turnover_sequence,
            destination_slot=turnover_destination,
            mate_parent_id=turnover_mate_parent,
        ),
    }


def _enabled_outputs(**kwargs: object) -> dict[str, torch.Tensor]:
    outputs = _outputs(**kwargs)
    if not bool(outputs["turnover_genome_v2_causal_capture_enabled"].item()):
        outputs["turnover_genome_v2_causal_capture_enabled"].fill_(True)
    return outputs


def _terminal(
    *,
    child_slots: dict[int, int],
    capacity: int = 6,
) -> tuple[
    CudaGenomeUpgradeV2StateBank,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    bank = CudaGenomeUpgradeV2StateBank.allocate(
        cells=1,
        capacity=capacity,
        device="cpu",
    )
    context = torch.zeros((1, capacity, 8), dtype=torch.int64)
    bank.express(context)
    alive = torch.zeros((1, capacity), dtype=torch.bool)
    slot_ids = torch.full((1, capacity), -1, dtype=torch.int64)
    for child_id, slot in child_slots.items():
        alive[0, slot] = True
        slot_ids[0, slot] = child_id
    return bank, alive, context, slot_ids


def _project(
    outputs: dict[str, torch.Tensor],
    *,
    child_slots: dict[int, int],
    binding: object | None = None,
) -> dict[str, object]:
    bank, alive, context, slots = _terminal(child_slots=child_slots)
    return project_genome_v2_causal_settlement_v1(
        causal_outputs=outputs,
        slot_to_agent_id=slots,
        bank=bank,
        alive_mask=alive,
        terminal_context_q10=context,
        cell_id="cell-a",
        physical_cell_ordinal=5,
        checkpoint_authority_binding=(
            _authority_binding() if binding is None else binding
        ),
    )


def test_contract_payload_is_canonical_and_content_addressed() -> None:
    assert SCHEMA_VERSION == "genome_v2_causal_settlement_projection_v2"
    assert CONTRACT_SHA256 == content_sha256(CONTRACT_PAYLOAD)
    assert CONTRACT_PAYLOAD["joined_record_capacity"] == 2048
    assert CONTRACT_PAYLOAD["producer_record_width"] == 182


def test_disabled_capture_is_validated_without_authority_binding() -> None:
    outputs = _outputs(capture_enabled=False)
    bank, alive, context, slots = _terminal(child_slots={})
    persistent = {**outputs, "slot_to_agent_id": slots}

    projection = project_genome_v2_causal_settlement_v1(
        persistent_state=persistent,
        bank=bank,
        alive_mask=alive,
        terminal_context_q10=context,
        cell_id="cell-a",
        physical_cell_ordinal=5,
    )

    assert projection["capture_status"] == "capture_disabled"
    assert projection["checkpoint_authority_binding"] is None
    assert projection["captured_counts"]["total"] == 0
    assert not projection["evaluable"]
    assert projection["reasons"] == ["causal_capture_disabled"]


def test_enabled_zero_capture_is_bounded_and_does_not_invoke_join() -> None:
    outputs = _outputs(capture_enabled=False)
    outputs["birth_genome_v2_causal_capture_enabled"].fill_(True)
    outputs["turnover_genome_v2_causal_capture_enabled"].fill_(True)

    projection = _project(outputs, child_slots={})

    assert projection["capture_status"] == "capture_enabled_no_records"
    assert projection["event_receipts"] == []
    assert projection["terminal_receipts"] == []
    assert projection["join_receipts"] == []
    assert projection["reasons"] == ["no_causal_rows_captured"]


def test_successful_join_uses_child_id_after_child_moved_slots() -> None:
    projection = _project(_enabled_outputs(), child_slots={17: 4})

    assert projection["capture_status"] == "capture_enabled_joined"
    assert projection["evaluable"]
    assert projection["reasons"] == []
    assert len(projection["join_receipts"]) == 1
    assert projection["retention_counts"] == {
        "direct_child_survival": 1,
        "event_ledger_descendant_retention": 0,
        "terminal_loss": 0,
        "retained_total": 1,
    }
    event = projection["event_receipts"][0]["transaction_record"]
    assert event["identity"]["parent_id"] == 3
    assert event["identity"]["mate_parent_id"] == 5
    assert event["context_summary"]["role"] == "frozen_event_applied_context"
    assert event["context_summary"]["context_q10"] == [0] * 8
    assert event["regulatory_summary"]["source"] == "reference_recomputation"
    assert "regulatory_cost_q10" in event["cost_summary"]
    terminal = projection["terminal_receipts"][0]["transaction_record"]
    assert terminal["identity"]["mate_parent_id"] == 5
    assert terminal["resulting_state"]["terminal_slot"] == 4
    assert terminal["identity"]["destination_slot"] == 1
    assert terminal["context_summary"]["role"] == "terminal_observation_context"
    assert terminal["applied_consequence"]["available"] is False


def test_recycled_original_slot_is_never_used_as_terminal_authority() -> None:
    bank, alive, context, slots = _terminal(child_slots={17: 4, 99: 1})
    projection = project_genome_v2_causal_settlement_v1(
        causal_outputs=_enabled_outputs(),
        slot_to_agent_id=slots,
        bank=bank,
        alive_mask=alive,
        terminal_context_q10=context,
        cell_id="cell-a",
        physical_cell_ordinal=5,
        checkpoint_authority_binding=_authority_binding(),
    )

    state = projection["terminal_receipts"][0]["transaction_record"][
        "resulting_state"
    ]
    assert state["agent_id"] == 17
    assert state["event_child_id"] == 17
    assert state["retention_relation"] == "direct_child_survival"
    assert state["terminal_slot"] == 4


def test_terminal_observation_context_is_not_treated_as_applied_context() -> None:
    bank, alive, context, slots = _terminal(child_slots={17: 4})
    context[0, 4, 0] = 1024

    projection = project_genome_v2_causal_settlement_v1(
        causal_outputs=_enabled_outputs(),
        slot_to_agent_id=slots,
        bank=bank,
        alive_mask=alive,
        terminal_context_q10=context,
        cell_id="cell-a",
        physical_cell_ordinal=5,
        checkpoint_authority_binding=_authority_binding(),
    )

    terminal = projection["terminal_receipts"][0]["transaction_record"]
    assert projection["evaluable"]
    assert terminal["context_summary"]["expression_binding"] == (
        "not_retained_applied_context"
    )
    assert terminal["regulatory_summary"]["available"] is False


@pytest.mark.parametrize("slots", ({}, {18: 1, 117: 4}))
def test_missing_terminal_child_produces_loss_receipt(
    slots: dict[int, int],
) -> None:
    projection = _project(_enabled_outputs(), child_slots=slots)

    assert not projection["evaluable"]
    assert projection["join_receipts"] == []
    assert projection["record_counts"] == {"event": 1, "terminal": 1, "joined": 0}
    assert projection["retention_counts"]["terminal_loss"] == 1
    assert projection["reasons"] == ["no_terminal_retention"]


def test_event_graph_descendant_retention_is_joined_without_reporting() -> None:
    outputs = _enabled_outputs(
        turnover_record=True,
        birth_child=17,
        birth_sequence=9,
        turnover_child=81,
        turnover_sequence=10,
        turnover_destination=2,
        turnover_mate_parent=7,
    )
    outputs["turnover_genome_v2_causal_record_rows"][0, 0, 8] = 17
    projection = _project(outputs, child_slots={81: 4})

    assert projection["evaluable"]
    assert projection["record_counts"] == {"event": 2, "terminal": 2, "joined": 2}
    assert projection["retention_counts"]["event_ledger_descendant_retention"] == 1
    descendant_join = [
        receipt
        for receipt in projection["join_receipts"]
        if receipt["retention_relation"] == "event_ledger_descendant_retention"
    ][0]
    assert descendant_join["terminal_agent_id"] == 81


def test_duplicate_live_child_id_is_not_evaluable() -> None:
    bank, alive, context, slots = _terminal(child_slots={17: 1})
    alive[0, 4] = True
    slots[0, 4] = 17

    projection = project_genome_v2_causal_settlement_v1(
        causal_outputs=_enabled_outputs(),
        slot_to_agent_id=slots,
        bank=bank,
        alive_mask=alive,
        terminal_context_q10=context,
        cell_id="cell-a",
        physical_cell_ordinal=5,
        checkpoint_authority_binding=_authority_binding(),
    )

    assert not projection["evaluable"]
    assert projection["reasons"][0].startswith("terminal_child_ambiguous:")


def test_dropped_rows_make_projection_not_evaluable() -> None:
    projection = _project(
        _enabled_outputs(dropped=2),
        child_slots={17: 4},
    )

    assert not projection["evaluable"]
    assert projection["dropped_counts"]["total"] == 2
    assert projection["reasons"] == ["causal_rows_dropped"]
    assert projection["join_receipts"] == []


def test_enabled_capture_rejects_malformed_authority_binding() -> None:
    with pytest.raises(
        GenomeV2CausalSettlementError,
        match="authority binding",
    ):
        _project(_enabled_outputs(), child_slots={17: 4}, binding={})


def test_rejects_cross_producer_duplicate_natural_identity() -> None:
    outputs = _enabled_outputs(
        turnover_record=True,
        turnover_child=17,
        turnover_sequence=9,
        turnover_destination=1,
    )

    with pytest.raises(
        GenomeV2CausalSettlementError,
        match="duplicate natural identity",
    ):
        _project(outputs, child_slots={17: 4})


def test_projection_hash_is_stable_and_covers_payload() -> None:
    first = _project(_enabled_outputs(), child_slots={17: 4})
    second = _project(_enabled_outputs(), child_slots={17: 4})

    assert first == second
    expected = dict(first)
    digest = expected.pop("projection_sha256")
    assert digest == content_sha256(expected)
