from __future__ import annotations

import copy
import json

from tools.scripts.eiel_hapb_frag_oracle_compare_v1 import (
    campaign_run_to_h4_compare_receipt_v1,
    compare_h4_receipts_v1,
)


def _digest(number: int) -> str:
    return f"{number:064x}"


def _receipt() -> dict:
    cell_ids = [f"cell-{index:02d}" for index in range(48)]
    projections = {
        name: {cell_id: _digest(offset + index) for index, cell_id in enumerate(cell_ids)}
        for name, offset in (
            ("reference_terminal", 100),
            ("midpoint", 200),
            ("reference_midpoint", 300),
            ("resumed_terminal", 100),
        )
    }
    return {
        "h4_exactness_contract": {
            "schema_version": "eiel_hapb_h4_exactness_contract_v1",
            "cell_count": 48,
            "cell_ids": cell_ids,
            "projections": projections,
            "identity": {
                "host_harness_source_commit": "a" * 40,
                "artifact_source_commit": "b" * 40,
                "artifact_sha256": _digest(401),
                "manifest_sha256": _digest(402),
                "execution_route_identity": "tick_native_execution_route_v1",
                "tick_native_control_sha256": _digest(403),
            },
            "observations": {
                "policy_traffic_sha256": _digest(404),
                "rng_trace_sha256": _digest(405),
                "action_cache_sha256": _digest(406),
            },
            "checks": {
                "cell_count_exact": True,
                "midpoint_reference_exact": True,
                "checkpoint_restore_exact": True,
                "fresh_owner_restore": True,
                "continuation_exact": True,
                "compatibility_alias_deletion_proof": True,
                "action_cache_observed": True,
            },
        },
        "reference_terminal_projection_digests": [
            projections["reference_terminal"][cell_id] for cell_id in cell_ids
        ],
        "resumed_terminal_projection_digests": [
            projections["resumed_terminal"][cell_id] for cell_id in cell_ids
        ],
    }


def test_exact_48_cell_contract_compares_equal() -> None:
    result = compare_h4_receipts_v1(_receipt(), _receipt())
    assert result["digest_count"] == 48
    assert result["terminal_projection_digests_equal"] is True
    assert result["bit_equal"] is True


def test_cell_bound_projection_mismatch_is_not_a_pass() -> None:
    candidate = _receipt()
    candidate["h4_exactness_contract"]["projections"]["resumed_terminal"]["cell-17"] = _digest(999)
    result = compare_h4_receipts_v1(_receipt(), candidate)
    assert result["terminal_projection_digests_equal"] is False
    assert result["bit_equal"] is False


def test_missing_action_cache_is_fail_closed() -> None:
    candidate = _receipt()
    candidate["h4_exactness_contract"]["observations"]["action_cache_sha256"] = None
    result = compare_h4_receipts_v1(_receipt(), candidate)
    assert result["bit_equal"] is False
    assert result["qualification_claim"] is False


def test_missing_contract_is_fail_closed() -> None:
    result = compare_h4_receipts_v1({}, _receipt())
    assert result["bit_equal"] is False
    assert "missing" in result["error"]


def test_run_projection_reader_preserves_strict_receipt(tmp_path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "hapb_continuation_receipt.json").write_text(
        json.dumps(_receipt()), encoding="utf-8"
    )
    loaded = campaign_run_to_h4_compare_receipt_v1(run_root)
    assert loaded["h4_exactness_contract"]["cell_count"] == 48
