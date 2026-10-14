from __future__ import annotations

import numpy as np
import pytest

from engine.perf.hapb_v2.content_publication_v1 import (
    HapbV2ContentPublicationLedgerV1,
    HapbV2PublicationError,
)
from engine.perf.hapb_v2.exactness_contract_v1 import POLICY_WIDTH_V1


def _row(scale: float) -> np.ndarray:
    return np.full((POLICY_WIDTH_V1,), np.float32(scale), dtype=np.float32)


def test_exact_clone_reuses_zero_weight_publication() -> None:
    ledger = HapbV2ContentPublicationLedgerV1(cells=1, slots_per_cell=4)
    parent = _row(0.1)
    first = ledger.publish_new_unique_content_v1(cell=0, slot=0, row=parent)
    assert first["h2d_row_published"] is True
    clone = ledger.alias_exact_clone_v1(
        cell=0,
        parent_slot=0,
        child_slot=1,
        parent_row=parent,
        child_row=parent.copy(),
    )
    assert clone["kind"] == "exact_clone_alias"
    assert clone["h2d_row_published"] is False
    assert int(ledger.device_map.content_ids[0, 1]) == int(first["content_id"])


def test_new_unique_overflow_content_gets_one_publication() -> None:
    ledger = HapbV2ContentPublicationLedgerV1(cells=1, slots_per_cell=4)
    first = ledger.publish_new_unique_content_v1(cell=0, slot=0, row=_row(0.2))
    mutated = ledger.bump_mutated_content_v1(
        cell=0, slot=2, new_row=_row(0.7)
    )
    assert mutated["kind"] == "mutated_unique_publish"
    assert mutated["h2d_row_published"] is True
    assert mutated["content_id"] != first["content_id"]
    reuse = ledger.publish_new_unique_content_v1(cell=0, slot=3, row=_row(0.7))
    assert reuse["kind"] == "reuse"
    assert reuse["h2d_row_published"] is False


def test_non_exact_clone_fails_closed() -> None:
    ledger = HapbV2ContentPublicationLedgerV1(cells=1, slots_per_cell=2)
    parent = _row(0.3)
    ledger.publish_new_unique_content_v1(cell=0, slot=0, row=parent)
    with pytest.raises(HapbV2PublicationError, match="exact parent/child"):
        ledger.alias_exact_clone_v1(
            cell=0,
            parent_slot=0,
            child_slot=1,
            parent_row=parent,
            child_row=_row(0.31),
        )


def test_device_map_can_be_reconstructed_from_host_events() -> None:
    ledger = HapbV2ContentPublicationLedgerV1(cells=1, slots_per_cell=3)
    row = _row(0.4)
    ledger.publish_new_unique_content_v1(cell=0, slot=0, row=row)
    ledger.alias_exact_clone_v1(
        cell=0,
        parent_slot=0,
        child_slot=1,
        parent_row=row,
        child_row=row,
    )
    ledger.device_map.invalidate_all_v1()
    ledger.reconstruct_device_from_host_v1()
    assert int(ledger.device_map.content_ids[0, 0]) == 1
    assert int(ledger.device_map.content_ids[0, 1]) == 1
