from __future__ import annotations

from pathlib import Path

import numpy as np

from engine.perf.hapb_v2.exactness_contract_v1 import (
    ACTION_ATTACK_V1,
    CONTRACT_ID_V1,
    HIDDEN_SIZE_V1,
    OBSERVATION_SIZE_V1,
    PACK_OFFSETS_V1,
    POLICY_WIDTH_V1,
    SOURCE_FILE_V1,
    SOURCE_HELPER_V1,
    evaluate_fixed_mlp_policy_logits_from_pack_exact_v1,
    exactness_contract_document_v1,
    sequential_dot_plus_bias_v1,
)


ROOT = Path(__file__).resolve().parents[1]


def test_contract_matches_v1_helper_and_pack_width() -> None:
    source = (ROOT / SOURCE_FILE_V1).read_text()
    assert f"void {SOURCE_HELPER_V1}(" in source
    assert "if (threadIdx.x != 0 || row >= rows) return;" in source
    assert "hapb_compact_policy_logits_kernel_v1<<<" in source
    assert "static_cast<unsigned int>(rows), 1, 0," in source
    document = exactness_contract_document_v1()
    assert document["contract_id"] == CONTRACT_ID_V1
    assert document["policy_width"] == POLICY_WIDTH_V1
    assert document["pack_offsets"]["w1"] == 0
    assert document["gemm_status"]["implemented"] is False


def test_per_neuron_accumulation_is_left_to_right_not_a_tree() -> None:
    weights = np.asarray([1.0, 1e-8, 1.0], dtype=np.float32)
    inputs = np.asarray([1e8, 1.0, -1e8], dtype=np.float32)
    sequential = sequential_dot_plus_bias_v1(weights, inputs, np.float32(0.0))
    tree_like = np.float32(
        (weights[0] * inputs[0] + weights[2] * inputs[2]) + weights[1] * inputs[1]
    )
    assert sequential != tree_like
    assert float(sequential) == 0.0 or np.isfinite(sequential)


def test_cpu_reference_respects_attack_bias_before_temperature() -> None:
    pack = np.zeros((POLICY_WIDTH_V1,), dtype=np.float32)
    observation = np.zeros((OBSERVATION_SIZE_V1,), dtype=np.float32)
    drives = np.zeros((1, 4), dtype=np.float32)
    attack = np.asarray([2.0], dtype=np.float32)
    temperature = np.asarray([0.5], dtype=np.float32)
    gain = np.asarray([0.0], dtype=np.float32)
    logits = evaluate_fixed_mlp_policy_logits_from_pack_exact_v1(
        pack,
        observation,
        slot=0,
        drives=drives,
        attack_bias=attack,
        action_temperature=temperature,
        drive_logit_gain=gain,
    )
    assert logits.shape == (7,)
    bias_before_divide = np.float32(attack[0] / temperature[0])
    bias_after_divide = np.float32(attack[0])
    assert logits[ACTION_ATTACK_V1] == bias_before_divide
    assert logits[ACTION_ATTACK_V1] != bias_after_divide
    assert np.all(logits[:ACTION_ATTACK_V1] == np.float32(0.0))


def test_hidden_layer_uses_tanh_and_bias_first() -> None:
    pack = np.zeros((POLICY_WIDTH_V1,), dtype=np.float32)
    pack[PACK_OFFSETS_V1["b1"]] = np.float32(0.5)
    pack[PACK_OFFSETS_V1["w2"]] = np.float32(1.0)
    pack[PACK_OFFSETS_V1["wout"]] = np.float32(1.0)
    observation = np.zeros((OBSERVATION_SIZE_V1,), dtype=np.float32)
    drives = np.zeros((1, 4), dtype=np.float32)
    logits = evaluate_fixed_mlp_policy_logits_from_pack_exact_v1(
        pack,
        observation,
        slot=0,
        drives=drives,
        attack_bias=np.zeros((1,), dtype=np.float32),
        action_temperature=np.ones((1,), dtype=np.float32),
        drive_logit_gain=np.zeros((1,), dtype=np.float32),
    )
    identity_if_no_tanh = np.float32(0.5)
    assert logits.shape == (7,)
    assert logits[0] != identity_if_no_tanh
    assert logits[0] != np.float32(0.0)
    assert np.all(np.isfinite(logits))
