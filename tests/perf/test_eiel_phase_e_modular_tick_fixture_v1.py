from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from perf.eiel_phase_e_modular_tick_fixture_v1 import (
    ACTION_WIDTH,
    AGENT_COUNT,
    ARM_NAMES,
    EXPECTED_RECEIPT_SHA256,
    LANE_OWNER,
    NEUTRAL_AGENT_COUNT,
    OBSERVATION_WIDTH,
    STRUCTURED_AGENT_COUNT,
    build_phase_e_modular_tick_fixture_receipt,
    validate_phase_e_modular_tick_fixture_receipt,
)


@pytest.fixture(scope="module")
def receipt() -> dict:
    value = build_phase_e_modular_tick_fixture_receipt()
    validate_phase_e_modular_tick_fixture_receipt(value)
    return value


def _f32_bits(value: float) -> int:
    return int(np.float32(value).view(np.uint32))


def test_population_and_three_tick_contract_is_exact(receipt: dict) -> None:
    population = receipt["population_contract"]
    assert population == {
        "agent_count": AGENT_COUNT,
        "structured_agent_count": STRUCTURED_AGENT_COUNT,
        "neutral_agent_count": NEUTRAL_AGENT_COUNT,
        "flat_cells": 1,
        "observation_width": OBSERVATION_WIDTH,
        "action_width": ACTION_WIDTH,
        "tick_count": 3,
        "tick_ages": [0, 20, 40],
    }
    assert population["structured_agent_count"] == 4
    assert population["neutral_agent_count"] == 60


def test_known_initial_values_use_exact_float32_bits(receipt: dict) -> None:
    known = receipt["known_initial"]
    assert known["integrator_state_bits"] == _f32_bits(0.1)
    assert known["integrator_output_bits"] == _f32_bits(0.05)
    assert known["dormant_state_bits"] == _f32_bits(0.75)
    assert known["dormant_output_bits"] == _f32_bits(0.5)


def test_exactly_eight_tick_arms_are_green_and_bounded(receipt: dict) -> None:
    assert tuple(receipt["arms"]) == ARM_NAMES
    assert len(receipt["arms"]) == 8
    for arm in receipt["arms"].values():
        assert arm["ok"] is True
        assert arm["first_mismatch"] is None
        assert arm["lane_owner"] == LANE_OWNER
        assert arm["hidden_fallback_count"] == 0
        assert arm["structural_rng"] == {"before": 0, "after": 0, "delta": 0}
        assert len(arm["raw_sha256"]) == 64
        assert len(arm["float32_bits_sha256"]) == 64
        assert len(arm["tensor_manifest_sha256"]) == 64
        assert len(arm["shape_digest_16_48"]) == 64


def test_tensor_receipts_freeze_dtype_shape_byte_count_and_digest(receipt: dict) -> None:
    required = {"dtype", "shape", "nbytes", "bytes_sha256"}
    for arm in receipt["arms"].values():
        assert arm["tensor_manifest"]
        for tensor in arm["tensor_manifest"].values():
            assert set(tensor) == required
            assert tensor["nbytes"] >= 0
            assert len(tensor["bytes_sha256"]) == 64
    assert receipt["fixed_inputs"]["observations"]["shape"] == [3, 64, 27]
    assert receipt["fixed_inputs"]["base_logits"]["shape"] == [3, 64, 7]
    assert receipt["fixed_inputs"]["policy_tape"]["shape"] == [3, 64, 7]
    assert receipt["fixed_inputs"]["action_masks"]["shape"] == [3, 64, 7]


def test_identity_slot_shape_and_rng_receipts_are_explicit(receipt: dict) -> None:
    assert set(receipt["biological_identity_map"]) == {"0", "1", "2", "3"}
    assert set(receipt["local_slot_map"]) == {"0", "1", "2", "3"}
    assert receipt["shape_contract"]["b2_caps"]["max_modules_per_agent"] == 16
    assert receipt["shape_contract"]["b2_caps"]["max_connections_per_agent"] == 48
    assert receipt["shape_contract"]["shape"]["max_module_slots"] == 16
    assert receipt["shape_contract"]["shape"]["max_connection_slots"] == 48
    assert receipt["structural_rng"] == {"before": 0, "after": 0, "delta": 0}
    assert receipt["hidden_fallback_count"] == 0


def test_causal_reset_lesion_dependency_and_reactivation_checks_are_green(receipt: dict) -> None:
    assert receipt["arms"]["integrator"]["checks"]["reset_changes_state"] is True
    assert receipt["arms"]["input_routing"]["checks"]["lesion_zeroes_route"] is True
    assert receipt["arms"]["output_routing"]["checks"]["binding_lesion_zero"] is True
    assert receipt["arms"]["dependency"]["checks"]["lesion_blocks_dependency"] is True
    assert receipt["arms"]["dormancy"]["checks"]["reactivation_changes_state"] is True
    assert receipt["arms"]["structural_cost"]["checks"]["reporting_only"] is True


def test_repeat_self_digest_and_false_claims_are_enforced(receipt: dict) -> None:
    assert receipt["receipt_sha256"] == EXPECTED_RECEIPT_SHA256
    assert receipt["same_process_repeat"]["ok"] is True
    assert receipt["same_process_repeat"]["first_sha256"] == receipt["same_process_repeat"]["second_sha256"]
    assert receipt["claims"] == {
        "cuda_parity": False,
        "campaign_validated": False,
        "phase_e_closed": False,
        "accelerated_runtime_validated": False,
    }
    validate_phase_e_modular_tick_fixture_receipt(receipt)
    tampered = deepcopy(receipt)
    tampered["claims"]["phase_e_closed"] = True
    with pytest.raises(ValueError, match="DIGEST_MISMATCH"):
        validate_phase_e_modular_tick_fixture_receipt(tampered)
