"""Frozen V1 per-neuron arithmetic contract for any later HAPB V2 kernel.

This module is the CPU-side authority for:

    same weights + same observation => bit-exact logits

It is a line-faithful translation of
``fixed_mlp_policy_logits_from_pack_exact_v1`` in
``engine/perf/cuda_architecture_b_flattened_wdl_controller_v2_kernels.cu``.
The N×64 candidate may parallelize *across* neurons of one layer, but it
must preserve this sequential left-to-right accumulation *inside* each
neuron.  Grouped GEMM is not an exact path under this contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Final, Mapping

import numpy as np


CONTRACT_SCHEMA_V1: Final[str] = "eiel_hapb_v2_policy_exactness_contract_v1"
CONTRACT_ID_V1: Final[str] = "hapb_v1_fixed_mlp_per_neuron_sequential_f32_v1"
SOURCE_HELPER_V1: Final[str] = "fixed_mlp_policy_logits_from_pack_exact_v1"
SOURCE_FILE_V1: Final[str] = (
    "engine/perf/cuda_architecture_b_flattened_wdl_controller_v2_kernels.cu"
)

OBSERVATION_SIZE_V1: Final[int] = 83
HIDDEN_SIZE_V1: Final[int] = 64
ACTION_COUNT_V1: Final[int] = 7
DRIVE_WIDTH_MIN_V1: Final[int] = 4
POLICY_WIDTH_V1: Final[int] = 9_991
DTYPE_V1: Final[str] = "float32"
TEMPERATURE_FLOOR_V1: Final[float] = 0.05
GAIN_FLOOR_V1: Final[float] = 0.0
ACTION_GATHER_V1: Final[int] = 4
ACTION_REST_V1: Final[int] = 5
ACTION_ATTACK_V1: Final[int] = 6

W1_WEIGHTS_V1: Final[int] = HIDDEN_SIZE_V1 * OBSERVATION_SIZE_V1
W1_BIASES_V1: Final[int] = HIDDEN_SIZE_V1
W2_WEIGHTS_V1: Final[int] = HIDDEN_SIZE_V1 * HIDDEN_SIZE_V1
W2_BIASES_V1: Final[int] = HIDDEN_SIZE_V1
WOUT_WEIGHTS_V1: Final[int] = ACTION_COUNT_V1 * HIDDEN_SIZE_V1
WOUT_BIASES_V1: Final[int] = ACTION_COUNT_V1

assert (
    W1_WEIGHTS_V1
    + W1_BIASES_V1
    + W2_WEIGHTS_V1
    + W2_BIASES_V1
    + WOUT_WEIGHTS_V1
    + WOUT_BIASES_V1
    == POLICY_WIDTH_V1
)

PACK_OFFSETS_V1: Final[dict[str, int]] = {
    "w1": 0,
    "b1": W1_WEIGHTS_V1,
    "w2": W1_WEIGHTS_V1 + W1_BIASES_V1,
    "b2": W1_WEIGHTS_V1 + W1_BIASES_V1 + W2_WEIGHTS_V1,
    "wout": W1_WEIGHTS_V1 + W1_BIASES_V1 + W2_WEIGHTS_V1 + W2_BIASES_V1,
    "bout": (
        W1_WEIGHTS_V1
        + W1_BIASES_V1
        + W2_WEIGHTS_V1
        + W2_BIASES_V1
        + WOUT_WEIGHTS_V1
    ),
}


class HapbV2ExactnessContractError(ValueError):
    """The frozen V1 arithmetic contract was violated."""


@dataclass(frozen=True)
class DriveWeightsV1:
    w_hunger_gather: float
    w_pain_rest: float
    w_fatigue_rest: float
    w_pain_attack: float
    w_fear_attack: float


DEFAULT_DRIVE_WEIGHTS_V1: Final[DriveWeightsV1] = DriveWeightsV1(
    w_hunger_gather=1.2,
    w_pain_rest=1.2,
    w_fatigue_rest=1.2,
    w_pain_attack=1.2,
    w_fear_attack=1.2,
)


def pack_offsets_v1(
    *,
    observation_size: int = OBSERVATION_SIZE_V1,
    hidden_size: int = HIDDEN_SIZE_V1,
    action_count: int = ACTION_COUNT_V1,
) -> dict[str, int]:
    if (
        observation_size != OBSERVATION_SIZE_V1
        or hidden_size != HIDDEN_SIZE_V1
        or action_count != ACTION_COUNT_V1
    ):
        raise HapbV2ExactnessContractError(
            "HAPB V2 exactness contract is frozen at obs=83 hidden=64 actions=7"
        )
    return dict(PACK_OFFSETS_V1)


def _require_f32_1d(name: str, value: np.ndarray, size: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float32 or array.ndim != 1 or int(array.shape[0]) != size:
        raise HapbV2ExactnessContractError(
            f"{name} must be a contiguous float32 vector of length {size}"
        )
    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array, dtype=np.float32)
    return array


def sequential_dot_plus_bias_v1(
    weights: np.ndarray,
    inputs: np.ndarray,
    bias: np.float32,
) -> np.float32:
    """One neuron: bias, then left-to-right ``acc += w[i] * x[i]``.

    This is multiply-then-add in program order.  It is not a reduction tree,
    not ``fmaf``, and not a GEMM tile accumulation.
    """

    if (
        weights.dtype != np.float32
        or inputs.dtype != np.float32
        or weights.ndim != 1
        or inputs.ndim != 1
        or int(weights.shape[0]) != int(inputs.shape[0])
    ):
        raise HapbV2ExactnessContractError(
            "per-neuron accumulation requires matching float32 vectors"
        )
    acc = np.float32(bias)
    for index in range(int(weights.shape[0])):
        acc = np.float32(acc + np.float32(weights[index] * inputs[index]))
    return acc


def tanh_f32_v1(value: np.float32) -> np.float32:
    """Host-side tanh for the algorithmic CPU contract.

    GPU bit-exactness uses device ``tanhf`` in the V1 helper.  Host libm
    and device ``tanhf`` are not bit-identical; P0-2 compares N×64 to the
    isolated V1 N×1 device helper, not to these host logits.
    """

    return np.float32(math.tanh(float(np.float32(value))))


def evaluate_fixed_mlp_policy_logits_from_pack_exact_v1(
    pack: np.ndarray,
    observation: np.ndarray,
    *,
    slot: int,
    drives: np.ndarray,
    attack_bias: np.ndarray,
    action_temperature: np.ndarray,
    drive_logit_gain: np.ndarray,
    drive_weights: DriveWeightsV1 = DEFAULT_DRIVE_WEIGHTS_V1,
) -> np.ndarray:
    """CPU reference of ``fixed_mlp_policy_logits_from_pack_exact_v1``."""

    pack = _require_f32_1d("policy pack", pack, POLICY_WIDTH_V1)
    observation = _require_f32_1d("observation", observation, OBSERVATION_SIZE_V1)
    drives = np.asarray(drives, dtype=np.float32)
    if drives.ndim != 2 or int(drives.shape[1]) < DRIVE_WIDTH_MIN_V1:
        raise HapbV2ExactnessContractError("drives must be [slots, >=4] float32")
    if type(slot) is not int or slot < 0 or slot >= int(drives.shape[0]):
        raise HapbV2ExactnessContractError("logical slot is outside drive rows")
    attack_bias = np.asarray(attack_bias, dtype=np.float32).reshape(-1)
    action_temperature = np.asarray(action_temperature, dtype=np.float32).reshape(-1)
    drive_logit_gain = np.asarray(drive_logit_gain, dtype=np.float32).reshape(-1)
    if (
        int(attack_bias.shape[0]) != int(drives.shape[0])
        or int(action_temperature.shape[0]) != int(drives.shape[0])
        or int(drive_logit_gain.shape[0]) != int(drives.shape[0])
    ):
        raise HapbV2ExactnessContractError("slot scalar vectors must match drive rows")

    offsets = PACK_OFFSETS_V1
    h1 = np.empty((HIDDEN_SIZE_V1,), dtype=np.float32)
    h2 = np.empty((HIDDEN_SIZE_V1,), dtype=np.float32)
    for hidden in range(HIDDEN_SIZE_V1):
        weights = pack[
            offsets["w1"] + hidden * OBSERVATION_SIZE_V1
            : offsets["w1"] + (hidden + 1) * OBSERVATION_SIZE_V1
        ]
        bias = np.float32(pack[offsets["b1"] + hidden])
        h1[hidden] = tanh_f32_v1(
            sequential_dot_plus_bias_v1(weights, observation, bias)
        )
    for hidden in range(HIDDEN_SIZE_V1):
        weights = pack[
            offsets["w2"] + hidden * HIDDEN_SIZE_V1
            : offsets["w2"] + (hidden + 1) * HIDDEN_SIZE_V1
        ]
        bias = np.float32(pack[offsets["b2"] + hidden])
        h2[hidden] = tanh_f32_v1(sequential_dot_plus_bias_v1(weights, h1, bias))

    raw_temperature = np.float32(action_temperature[slot])
    temperature = (
        np.float32(TEMPERATURE_FLOOR_V1)
        if raw_temperature < np.float32(TEMPERATURE_FLOOR_V1)
        else raw_temperature
    )
    raw_gain = np.float32(drive_logit_gain[slot])
    gain = (
        np.float32(GAIN_FLOOR_V1)
        if raw_gain < np.float32(GAIN_FLOOR_V1)
        else raw_gain
    )
    slot_drives = drives[slot]
    logits = np.empty((ACTION_COUNT_V1,), dtype=np.float32)
    hunger = np.float32(drive_weights.w_hunger_gather)
    pain_rest = np.float32(drive_weights.w_pain_rest)
    fatigue_rest = np.float32(drive_weights.w_fatigue_rest)
    pain_attack = np.float32(drive_weights.w_pain_attack)
    fear_attack = np.float32(drive_weights.w_fear_attack)
    for action in range(ACTION_COUNT_V1):
        weights = pack[
            offsets["wout"] + action * HIDDEN_SIZE_V1
            : offsets["wout"] + (action + 1) * HIDDEN_SIZE_V1
        ]
        bias = np.float32(pack[offsets["bout"] + action])
        score = sequential_dot_plus_bias_v1(weights, h2, bias)
        if action == ACTION_ATTACK_V1:
            score = np.float32(score + np.float32(attack_bias[slot]))
        score = np.float32(score / temperature)
        if action == ACTION_GATHER_V1:
            score = np.float32(
                score + np.float32(gain * np.float32(hunger * slot_drives[0]))
            )
        elif action == ACTION_REST_V1:
            rest_term = np.float32(
                np.float32(pain_rest * slot_drives[1])
                + np.float32(fatigue_rest * slot_drives[2])
            )
            score = np.float32(score + np.float32(gain * rest_term))
        elif action == ACTION_ATTACK_V1:
            attack_term = np.float32(
                np.float32(pain_attack * slot_drives[1])
                + np.float32(fear_attack * slot_drives[3])
            )
            score = np.float32(score - np.float32(gain * attack_term))
        logits[action] = score
    return logits


def logits_sha256_v1(logits: np.ndarray) -> str:
    array = np.asarray(logits, dtype=np.float32)
    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array, dtype=np.float32)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def exactness_contract_document_v1() -> dict[str, object]:
    """Machine-readable freeze of the V1 per-neuron arithmetic."""

    return {
        "schema_version": CONTRACT_SCHEMA_V1,
        "contract_id": CONTRACT_ID_V1,
        "source_helper": SOURCE_HELPER_V1,
        "source_file": SOURCE_FILE_V1,
        "dtype": DTYPE_V1,
        "observation_size": OBSERVATION_SIZE_V1,
        "hidden_size": HIDDEN_SIZE_V1,
        "action_count": ACTION_COUNT_V1,
        "policy_width": POLICY_WIDTH_V1,
        "pack_offsets": dict(PACK_OFFSETS_V1),
        "activation": "tanhf",
        "accumulation": (
            "per-neuron sequential left-to-right multiply-then-add starting "
            "from the neuron bias; no reduction tree; no explicit fmaf; "
            "no TF32; no FP16"
        ),
        "layer_order": ["h1_obs_affine_tanh", "h2_h1_affine_tanh", "logits"],
        "temperature_floor": TEMPERATURE_FLOOR_V1,
        "gain_floor": GAIN_FLOOR_V1,
        "attack_bias_before_temperature_divide": True,
        "drive_terms_after_temperature_divide": True,
        "action_gather": ACTION_GATHER_V1,
        "action_rest": ACTION_REST_V1,
        "action_attack": ACTION_ATTACK_V1,
        "parallelism_allowed": (
            "across neurons of one layer after the previous layer is complete"
        ),
        "parallelism_forbidden": (
            "inside one neuron; across layers without an explicit barrier; "
            "GEMM/blocked/tree accumulation"
        ),
        "v1_launch": {
            "grid": "rows",
            "block": 1,
            "useful_threads": 1,
            "guard": "threadIdx.x != 0 returns",
            "kernels": [
                "hapb_compact_policy_logits_kernel_v1",
                "hapb_hot_prefix_policy_logits_kernel_v1",
                "hapb_content_alias_policy_logits_kernel_v1",
                "hapb_resident_policy_logits_oracle_kernel_v1",
            ],
        },
        "gemm_status": {
            "implemented": False,
            "reason": (
                "standard grouped/batched GEMM changes per-neuron accumulation "
                "order and may enable TF32; it is not bit-exact under this "
                "contract and remains a future measured candidate only"
            ),
        },
        "no_claims": [
            "not a production route",
            "not a qualification claim",
            "not a V2 success claim",
        ],
    }


def exactness_contract_canonical_json_v1() -> str:
    return json.dumps(exactness_contract_document_v1(), indent=2, sort_keys=True) + "\n"


def exactness_contract_sha256_v1() -> str:
    return hashlib.sha256(
        exactness_contract_canonical_json_v1().encode("utf-8")
    ).hexdigest()


def require_contract_frozen_v1(document: Mapping[str, object]) -> None:
    if document.get("contract_id") != CONTRACT_ID_V1:
        raise HapbV2ExactnessContractError("exactness contract id mismatch")
    if document.get("schema_version") != CONTRACT_SCHEMA_V1:
        raise HapbV2ExactnessContractError("exactness contract schema mismatch")
    if document.get("policy_width") != POLICY_WIDTH_V1:
        raise HapbV2ExactnessContractError("exactness contract pack width mismatch")
