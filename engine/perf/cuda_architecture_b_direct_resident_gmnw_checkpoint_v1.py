"""Strict CPU sidecar checkpoints for integrated resident GMNW state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from math import isfinite
from typing import Any, Final

import torch

from engine.perf.cuda_architecture_b_direct_resident_gmnw_gru_v1 import (
    GMNW_COUNTER_COLUMNS_V1,
    GMNW_MATCHED_MLP_PACKED_WIDTH_V1,
    GMNW_MATCHED_MLP_SCRATCH_WIDTH_V1,
    GMNW_POLICY_ARM_CODES_V1,
    GMNW_SELECTION_COUNT_METRIC_NAMES_V1,
    GMNW_SELECTION_METRIC_NAMES_V1,
    GMNW_SELECTION_METRICS_STATUS_NAME_V1,
    GMNW_SELECTION_POLICY_FAMILIES_V1,
    GMNW_VENV_CONSEQUENCE_PROFILE_CODES_V1,
    GMNW_VENV_CONSEQUENCE_PROFILE_TENSOR_INDEX_V1,
    PreparedDirectResidentGmnwGruV1,
)


GMNW_CHECKPOINT_SIDECAR_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_cuda_architecture_b_direct_resident_gmnw_checkpoint_sidecar_v1"
)
GMNW_CHECKPOINT_SIDECAR_CODEC_VERSION_V1: Final[int] = 1
GMNW_CHECKPOINT_SIDECAR_SCHEMA_VERSION_SEL1_V1: Final[str] = (
    "eiel_cuda_architecture_b_direct_resident_gmnw_checkpoint_sidecar_sel1_v1"
)
GMNW_CHECKPOINT_SIDECAR_CODEC_VERSION_SEL1_V1: Final[int] = 1
GMNW_CHECKPOINT_SIDECAR_SCHEMA_VERSION_SEL1_V2: Final[str] = (
    "eiel_cuda_architecture_b_direct_resident_gmnw_checkpoint_sidecar_sel1_v2"
)
GMNW_CHECKPOINT_SIDECAR_CODEC_VERSION_SEL1_V2: Final[int] = 2
GMNW_MULTICELL_CHECKPOINT_SCHEMA_VERSION_SEL1_V1: Final[str] = (
    "eiel_cuda_architecture_b_direct_resident_gmnw_multicell_checkpoint_sel1_v1"
)
GMNW_MULTICELL_CHECKPOINT_CODEC_VERSION_SEL1_V1: Final[int] = 1
GMNW_MULTICELL_CHECKPOINT_SCHEMA_VERSION_SEL1_V2: Final[str] = (
    "eiel_cuda_architecture_b_direct_resident_gmnw_multicell_checkpoint_sel1_v2"
)
GMNW_MULTICELL_CHECKPOINT_CODEC_VERSION_SEL1_V2: Final[int] = 2
GMNW_CHECKPOINT_TENSOR_NAMES_V1: Final[tuple[str, ...]] = (
    "stream_id",
    "episode_index",
    "cue_identity",
    "cue_phase",
    "phase_tick",
    "delay_remaining",
    "decision_due",
    "correct_action",
    "last_response",
    "correctness",
    "consequence_status",
    "task_rng_state",
    "task_rng_cursor",
    "reproduction_scale",
    "counters",
    "damage_total",
)
GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_V1: Final[tuple[str, ...]] = (
    *GMNW_CHECKPOINT_TENSOR_NAMES_V1,
    "task_status",
    "cue_alphabet",
    "delay_ticks",
    "distractors_enabled",
    "reset_hidden_each_tick",
    "task_seed",
    "policy_arm_code",
    "policy_use_gru",
)
GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_V2: Final[tuple[str, ...]] = (
    *GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_V1,
    "venv_consequence_profile_code",
)
_GMNW_CHECKPOINT_SELECTION_METRIC_NAMES_SEL1_V1: Final[tuple[str, ...]] = (
    *GMNW_SELECTION_METRIC_NAMES_V1,
    GMNW_SELECTION_METRICS_STATUS_NAME_V1,
)
_GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_WITH_METRICS_V1: Final[tuple[str, ...]] = (
    *GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_V1,
    *_GMNW_CHECKPOINT_SELECTION_METRIC_NAMES_SEL1_V1,
)
_GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_WITH_METRICS_V2: Final[tuple[str, ...]] = (
    *GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_V2,
    *_GMNW_CHECKPOINT_SELECTION_METRIC_NAMES_SEL1_V1,
)

_GMNW_CONFIG_NAMES_V1: Final[frozenset[str]] = frozenset(
    {
        "enabled",
        "generation_base",
        "generation_rows_per_cell",
        "cue_ticks",
        "consequence_ticks",
        "grace_generation_count",
        "severe_generation_count",
        "grace_damage_fraction",
        "severe_damage_fraction",
        "lethal_damage_fraction",
        "grace_reproduction_scale",
    }
)
_GMNW_CONFIG_NAMES_SEL1_LEGACY_V1: Final[frozenset[str]] = _GMNW_CONFIG_NAMES_V1 | {
    "control_matrix_enabled",
    "policy_arm_codes",
    "cue_visible_at_decision",
    "gru_founder_count",
    "qualification_trace_enabled",
    "qualification_trace_output_ticks",
    "qualification_decision_tape_enabled",
}
_GMNW_CONFIG_NAMES_SEL1_V1: Final[frozenset[str]] = (
    _GMNW_CONFIG_NAMES_SEL1_LEGACY_V1 | {"selection_metrics_enabled"}
)
_GMNW_CONFIG_NAMES_SEL1_EXPLICIT_MECHANICS_V1: Final[frozenset[str]] = (
    _GMNW_CONFIG_NAMES_SEL1_V1 | {"terminal_low_population_no_reinit"}
)
_GMNW_CONFIG_NAMES_SEL1_LEGACY_V2: Final[frozenset[str]] = (
    _GMNW_CONFIG_NAMES_SEL1_LEGACY_V1 | {"venv_consequence_profile_codes"}
)
_GMNW_CONFIG_NAMES_SEL1_V2: Final[frozenset[str]] = (
    _GMNW_CONFIG_NAMES_SEL1_V1 | {"venv_consequence_profile_codes"}
)
_GMNW_CONFIG_NAMES_SEL1_EXPLICIT_MECHANICS_V2: Final[frozenset[str]] = (
    _GMNW_CONFIG_NAMES_SEL1_EXPLICIT_MECHANICS_V1
    | {"venv_consequence_profile_codes"}
)
_GMNW_TENSOR_DTYPES_V1: Final[tuple[torch.dtype, ...]] = (
    torch.int64,
    torch.int64,
    torch.int8,
    torch.int8,
    torch.int32,
    torch.int32,
    torch.bool,
    torch.int8,
    torch.int8,
    torch.int8,
    torch.int8,
    torch.int64,
    torch.int64,
    torch.float32,
    torch.int64,
    torch.float64,
    torch.int32,
    torch.int32,
    torch.int32,
    torch.bool,
    torch.bool,
    torch.int64,
)
_GMNW_CONTROL_TENSOR_DTYPES_V1: Final[tuple[torch.dtype, ...]] = (
    torch.int8,
    torch.float32,
    torch.float32,
    torch.int32,
)
_GMNW_CHECKPOINT_TENSOR_DTYPES_SEL1_V1: Final[tuple[torch.dtype, ...]] = (
    *_GMNW_TENSOR_DTYPES_V1,
    torch.int8,
    torch.bool,
)
_GMNW_CHECKPOINT_TENSOR_DTYPES_SEL1_V2: Final[tuple[torch.dtype, ...]] = (
    *_GMNW_CHECKPOINT_TENSOR_DTYPES_SEL1_V1,
    torch.int32,
)
_GMNW_CHECKPOINT_SELECTION_METRIC_DTYPES_SEL1_V1: Final[tuple[torch.dtype, ...]] = (
    *(torch.int64 for _ in GMNW_SELECTION_COUNT_METRIC_NAMES_V1),
    torch.float64,
    torch.int32,
)
_SEL1_DELAYED_ARM_CODE_V1: Final[int] = GMNW_POLICY_ARM_CODES_V1[
    ("mixed_selection", False)
]
_SEL1_IMMEDIATE_ARM_CODE_V1: Final[int] = GMNW_POLICY_ARM_CODES_V1[
    ("mixed_selection", True)
]
_POLICY_ARM_CUE_VISIBILITY_BY_CODE_V1: Final[dict[int, bool]] = {
    code: cue_visible for (_, cue_visible), code in GMNW_POLICY_ARM_CODES_V1.items()
}
_SIDECAR_FIELDS_V1: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "codec_version",
        "absolute_generation",
        "gmnw_config",
        "tensor_manifest",
        "manifest_sha256",
        "tensors",
        "digest_sha256",
    }
)
_MANIFEST_ROW_FIELDS_V1: Final[frozenset[str]] = frozenset(
    {"index", "name", "dtype", "shape", "sha256"}
)
_MULTICELL_CHECKPOINT_FIELDS_SEL1_V1: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "codec_version",
        "absolute_generation",
        "cell_count",
        "cell_ids",
        "sidecars",
        "digest_sha256",
    }
)
_CONTROL_PER_CELL_CONFIG_NAMES_V1: Final[tuple[str, ...]] = (
    "policy_arm_codes",
    "cue_visible_at_decision",
    "gru_founder_count",
    "venv_consequence_profile_codes",
)


class DirectResidentGmnwCheckpointV1Error(ValueError):
    """The dedicated GMNW checkpoint sidecar is invalid or incompatible."""


def _fail(path: str, message: str) -> DirectResidentGmnwCheckpointV1Error:
    return DirectResidentGmnwCheckpointV1Error(f"{path}: {message}")


def _canonical_json_bytes(value: Any, *, path: str) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise _fail(path, "must be canonical-JSON compatible") from exc


def _canonical_sha256(value: Any, *, path: str) -> str:
    return sha256(_canonical_json_bytes(value, path=path)).hexdigest()


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().to(device="cpu").contiguous()
    return sha256(cpu.numpy().tobytes(order="C")).hexdigest()


def _required_absolute_generation(value: Any, *, path: str) -> int:
    if type(value) is not int or value < 0:
        raise _fail(path, "must be a nonnegative exact integer")
    return value


def _validated_core_config(
    value: Any, *, path: str, require_exact_fields: bool
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(path, "must be a mapping")
    if any(type(key) is not str for key in value):
        raise _fail(path, "keys must be exact strings")
    config = dict(value)
    if (
        require_exact_fields and set(config) != _GMNW_CONFIG_NAMES_V1
    ) or not _GMNW_CONFIG_NAMES_V1 <= set(config):
        missing = sorted(_GMNW_CONFIG_NAMES_V1 - set(config))
        extra = sorted(set(config) - _GMNW_CONFIG_NAMES_V1)
        raise _fail(path, f"field set mismatch; missing={missing}, extra={extra}")

    if config["enabled"] is not True:
        raise _fail(f"{path}.enabled", "must be exactly true")
    for name, minimum in (
        ("generation_base", 0),
        ("generation_rows_per_cell", 1),
        ("cue_ticks", 1),
        ("consequence_ticks", 1),
        ("grace_generation_count", 1),
        ("severe_generation_count", 1),
    ):
        value_at_name = config[name]
        if type(value_at_name) is not int or value_at_name < minimum:
            raise _fail(f"{path}.{name}", f"must be an exact integer >= {minimum}")
    for name in (
        "grace_damage_fraction",
        "severe_damage_fraction",
        "lethal_damage_fraction",
        "grace_reproduction_scale",
    ):
        value_at_name = config[name]
        if type(value_at_name) is not float or not isfinite(value_at_name):
            raise _fail(f"{path}.{name}", "must be an exact finite float")
        if value_at_name < 0.0:
            raise _fail(f"{path}.{name}", "must be nonnegative")
    if config["grace_reproduction_scale"] > 1.0:
        raise _fail(f"{path}.grace_reproduction_scale", "must not exceed 1.0")

    core = {name: config[name] for name in _GMNW_CONFIG_NAMES_V1}
    encoded = _canonical_json_bytes(core, path=path)
    decoded = json.loads(encoded.decode("ascii"))
    if not isinstance(decoded, dict):
        raise AssertionError("validated GMNW config did not encode as an object")
    return decoded


def _validated_prepared_config(value: Any, *, path: str) -> tuple[dict[str, Any], bool]:
    core = _validated_core_config(value, path=path, require_exact_fields=False)
    config = dict(value)
    if set(config) == _GMNW_CONFIG_NAMES_V1:
        return core, False
    if set(config) not in {
        _GMNW_CONFIG_NAMES_SEL1_LEGACY_V2,
        _GMNW_CONFIG_NAMES_SEL1_V2,
        _GMNW_CONFIG_NAMES_SEL1_EXPLICIT_MECHANICS_V2,
    }:
        missing = sorted(_GMNW_CONFIG_NAMES_SEL1_EXPLICIT_MECHANICS_V2 - set(config))
        extra = sorted(set(config) - _GMNW_CONFIG_NAMES_SEL1_EXPLICIT_MECHANICS_V2)
        raise _fail(path, f"field set mismatch; missing={missing}, extra={extra}")

    if type(config["control_matrix_enabled"]) is not bool:
        raise _fail(f"{path}.control_matrix_enabled", "must be an exact boolean")
    for name in (
        "qualification_trace_enabled",
        "qualification_decision_tape_enabled",
    ):
        if type(config[name]) is not bool:
            raise _fail(f"{path}.{name}", "must be an exact boolean")
    if "selection_metrics_enabled" in config and type(
        config["selection_metrics_enabled"]
    ) is not bool:
        raise _fail(
            f"{path}.selection_metrics_enabled", "must be an exact boolean"
        )
    if "terminal_low_population_no_reinit" in config and type(
        config["terminal_low_population_no_reinit"]
    ) is not bool:
        raise _fail(
            f"{path}.terminal_low_population_no_reinit",
            "must be an exact boolean",
        )
    trace_ticks = config["qualification_trace_output_ticks"]
    if type(trace_ticks) is not int or trace_ticks < 0:
        raise _fail(
            f"{path}.qualification_trace_output_ticks",
            "must be a nonnegative exact integer",
        )
    if config["qualification_trace_enabled"] is (trace_ticks == 0):
        raise _fail(
            f"{path}.qualification_trace_output_ticks",
            "must be positive exactly when qualification trace is enabled",
        )

    sequence_names = (
        "policy_arm_codes",
        "cue_visible_at_decision",
        "gru_founder_count",
        "venv_consequence_profile_codes",
    )
    sequences: dict[str, tuple[Any, ...]] = {}
    for name in sequence_names:
        raw = config[name]
        if not isinstance(raw, (list, tuple)) or not raw:
            raise _fail(f"{path}.{name}", "must be a nonempty list or tuple")
        sequences[name] = tuple(raw)
    cell_count = len(sequences["policy_arm_codes"])
    if any(len(values) != cell_count for values in sequences.values()):
        raise _fail(path, "per-cell control configuration lengths disagree")

    policy_codes = sequences["policy_arm_codes"]
    cue_visibility = sequences["cue_visible_at_decision"]
    founder_counts = sequences["gru_founder_count"]
    profile_codes = sequences["venv_consequence_profile_codes"]
    valid_codes = frozenset(_POLICY_ARM_CUE_VISIBILITY_BY_CODE_V1)
    valid_profile_codes = frozenset(GMNW_VENV_CONSEQUENCE_PROFILE_CODES_V1.values())
    for cell, (code, visible, founder_count, profile_code) in enumerate(
        zip(
            policy_codes,
            cue_visibility,
            founder_counts,
            profile_codes,
            strict=True,
        )
    ):
        if type(code) is not int or code not in valid_codes:
            raise _fail(f"{path}.policy_arm_codes[{cell}]", "is not a known arm code")
        if type(visible) is not bool:
            raise _fail(
                f"{path}.cue_visible_at_decision[{cell}]",
                "must be an exact boolean",
            )
        if visible is not _POLICY_ARM_CUE_VISIBILITY_BY_CODE_V1[code]:
            raise _fail(
                f"{path}.cue_visible_at_decision[{cell}]",
                "contradicts the policy arm code",
            )
        is_sel1_arm = code in {
            _SEL1_DELAYED_ARM_CODE_V1,
            _SEL1_IMMEDIATE_ARM_CODE_V1,
        }
        if is_sel1_arm:
            if type(founder_count) is not int or not 0 <= founder_count <= 64:
                raise _fail(
                    f"{path}.gru_founder_count[{cell}]",
                    "must be an exact integer in [0, 64] for SEL1",
                )
        elif founder_count is not None:
            raise _fail(
                f"{path}.gru_founder_count[{cell}]",
                "must be null outside mixed selection",
            )
        if type(profile_code) is not int or profile_code not in valid_profile_codes:
            raise _fail(
                f"{path}.venv_consequence_profile_codes[{cell}]",
                "must be an exact known VENV consequence profile code",
            )

    sel1_codes = {_SEL1_DELAYED_ARM_CODE_V1, _SEL1_IMMEDIATE_ARM_CODE_V1}
    sel1 = any(code in sel1_codes for code in policy_codes)
    if sel1 and any(code not in sel1_codes for code in policy_codes):
        raise _fail(path, "SEL1 sidecars cannot mix selection and control arm codes")
    extended = config["control_matrix_enabled"] or sel1 or any(profile_codes)
    if not config["control_matrix_enabled"] and any(code != 0 for code in policy_codes):
        raise _fail(
            f"{path}.control_matrix_enabled",
            "must be exactly true for non-default policy arms",
        )
    if extended and config["control_matrix_enabled"] is not True:
        raise _fail(
            f"{path}.control_matrix_enabled",
            "must be exactly true for SEL1 or VENV checkpoint authority",
        )
    if not extended:
        return core, False

    encoded = _canonical_json_bytes(config, path=path)
    decoded = json.loads(encoded.decode("ascii"))
    if not isinstance(decoded, dict):
        raise AssertionError("validated control config did not encode as an object")
    return decoded, True


def _validated_serialized_config(
    value: Any, *, path: str, sel1_version: int
) -> dict[str, Any]:
    if sel1_version == 0:
        return _validated_core_config(value, path=path, require_exact_fields=True)
    if sel1_version == 1:
        if not isinstance(value, Mapping):
            raise _fail(path, "must be a mapping")
        config = dict(value)
        if set(config) not in {
            _GMNW_CONFIG_NAMES_SEL1_LEGACY_V1,
            _GMNW_CONFIG_NAMES_SEL1_V1,
            _GMNW_CONFIG_NAMES_SEL1_EXPLICIT_MECHANICS_V1,
        }:
            raise _fail(path, "must be an exact historical SEL1 v1 configuration")
        policy_codes = config.get("policy_arm_codes")
        if not isinstance(policy_codes, (list, tuple)) or not policy_codes:
            raise _fail(f"{path}.policy_arm_codes", "must be a nonempty list or tuple")
        config["venv_consequence_profile_codes"] = [0] * len(policy_codes)
        value = config
    config, observed_sel1 = _validated_prepared_config(value, path=path)
    if not observed_sel1:
        raise _fail(path, "must be the exact extended control configuration")
    return config


def _validated_prepared_tensors(
    prepared: PreparedDirectResidentGmnwGruV1,
) -> tuple[torch.Tensor, ...]:
    tensors = getattr(prepared, "gmnw_tensors", None)
    if not isinstance(tensors, tuple) or len(tensors) not in (23, 27, 30):
        raise _fail(
            "prepared.gmnw_tensors",
            "must be the exact 23-, 27-, or 30-tensor GMNW ABI",
        )
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise _fail("prepared.gmnw_tensors", "all entries must be tensors")

    stream_id = tensors[0]
    if stream_id.ndim != 2 or any(extent <= 0 for extent in stream_id.shape):
        raise _fail("prepared.gmnw_tensors[0]", "must have positive [cells,capacity]")
    cells, capacity = (int(extent) for extent in stream_id.shape)
    config, _ = _validated_prepared_config(
        prepared.gmnw_config, path="prepared.gmnw_config"
    )
    metrics_enabled = config.get("selection_metrics_enabled")
    if metrics_enabled is True and len(tensors) != 30:
        raise _fail(
            "prepared.gmnw_tensors",
            "selection_metrics_enabled requires the 30-tensor resident ABI",
        )
    if metrics_enabled is False and len(tensors) == 30:
        raise _fail(
            "prepared.gmnw_tensors",
            "the 30-tensor resident ABI requires selection_metrics_enabled",
        )
    generations = config["generation_rows_per_cell"]
    rows = (cells, capacity)
    expected_shapes = (
        *(rows for _ in range(14)),
        (cells, generations, GMNW_COUNTER_COLUMNS_V1),
        (cells, generations),
        *((cells,) for _ in range(6)),
    )
    for index, (tensor, dtype, shape) in enumerate(
        zip(tensors[:22], _GMNW_TENSOR_DTYPES_V1, expected_shapes, strict=True)
    ):
        if tensor.dtype != dtype:
            raise _fail(
                f"prepared.gmnw_tensors[{index}].dtype",
                f"expected {_dtype_name(dtype)}, got {_dtype_name(tensor.dtype)}",
            )
        if tuple(tensor.shape) != shape:
            raise _fail(
                f"prepared.gmnw_tensors[{index}].shape",
                f"expected {list(shape)}, got {list(tensor.shape)}",
            )
    profile = tensors[GMNW_VENV_CONSEQUENCE_PROFILE_TENSOR_INDEX_V1]
    if profile.dtype != torch.int32:
        raise _fail("prepared.gmnw_tensors[22].dtype", "expected int32")
    if tuple(profile.shape) != (cells,):
        raise _fail(
            "prepared.gmnw_tensors[22].shape",
            f"expected [{cells}], got {list(profile.shape)}",
        )
    if not profile.is_contiguous():
        raise _fail("prepared.gmnw_tensors[22]", "must be contiguous")
    valid_profile_codes = torch.tensor(
        sorted(GMNW_VENV_CONSEQUENCE_PROFILE_CODES_V1.values()),
        dtype=profile.dtype,
        device=profile.device,
    )
    if not bool(torch.isin(profile, valid_profile_codes).all().item()):
        raise _fail(
            "prepared.gmnw_tensors[22]",
            "values must be known VENV consequence profile codes",
        )
    expected_profile_codes = torch.tensor(
        config.get("venv_consequence_profile_codes", (0,) * cells),
        dtype=profile.dtype,
        device=profile.device,
    )
    if not torch.equal(profile, expected_profile_codes):
        raise _fail(
            "prepared.gmnw_tensors[22]",
            "VENV consequence profile codes contradict prepared.gmnw_config",
        )
    if len(tensors) in (27, 30):
        control_shapes = (
            (cells,),
            (GMNW_MATCHED_MLP_PACKED_WIDTH_V1,),
            (cells, GMNW_MATCHED_MLP_SCRATCH_WIDTH_V1),
            (cells,),
        )
        for index, (tensor, dtype, shape) in enumerate(
            zip(
                tensors[23:27],
                _GMNW_CONTROL_TENSOR_DTYPES_V1,
                control_shapes,
                strict=True,
            ),
            start=23,
        ):
            if tensor.dtype != dtype:
                raise _fail(
                    f"prepared.gmnw_tensors[{index}].dtype",
                    f"expected {_dtype_name(dtype)}, got {_dtype_name(tensor.dtype)}",
                )
            if tuple(tensor.shape) != shape:
                raise _fail(
                    f"prepared.gmnw_tensors[{index}].shape",
                    f"expected {list(shape)}, got {list(tensor.shape)}",
                )
        policy_arm = tensors[23]
        valid_arm_codes = torch.tensor(
            sorted(_POLICY_ARM_CUE_VISIBILITY_BY_CODE_V1),
            dtype=policy_arm.dtype,
            device=policy_arm.device,
        )
        if not bool(torch.isin(policy_arm, valid_arm_codes).all().item()):
            raise _fail(
                "prepared.gmnw_tensors[23]",
                "policy arm values must be within the known control code set",
            )
        if bool(torch.count_nonzero(tensors[26]).item()):
            raise _fail(
                "prepared.gmnw_tensors[26]",
                "control status must be zero",
            )
    if len(tensors) == 30:
        selection_shapes = (
            (
                cells,
                generations,
                len(GMNW_SELECTION_POLICY_FAMILIES_V1),
                len(GMNW_SELECTION_COUNT_METRIC_NAMES_V1),
            ),
            (cells, generations, len(GMNW_SELECTION_POLICY_FAMILIES_V1)),
            (cells,),
        )
        selection_dtypes = (torch.int64, torch.float64, torch.int32)
        for index, (tensor, dtype, shape) in enumerate(
            zip(tensors[27:], selection_dtypes, selection_shapes, strict=True),
            start=27,
        ):
            if tensor.dtype != dtype:
                raise _fail(
                    f"prepared.gmnw_tensors[{index}].dtype",
                    f"expected {_dtype_name(dtype)}, got {_dtype_name(tensor.dtype)}",
                )
            if tuple(tensor.shape) != shape:
                raise _fail(
                    f"prepared.gmnw_tensors[{index}].shape",
                    f"expected {list(shape)}, got {list(tensor.shape)}",
                )
        metrics = getattr(prepared, "selection_metrics", None)
        expected_metric_names = {
            *GMNW_SELECTION_METRIC_NAMES_V1,
            GMNW_SELECTION_METRICS_STATUS_NAME_V1,
        }
        if not isinstance(metrics, Mapping) or set(metrics) != expected_metric_names:
            raise _fail(
                "prepared.selection_metrics",
                "must expose the exact resident 30-tensor metric views",
            )
        for column, name in enumerate(GMNW_SELECTION_COUNT_METRIC_NAMES_V1):
            if not torch.equal(metrics[name], tensors[27][..., column]):
                raise _fail(
                    f"prepared.selection_metrics.{name}",
                    "must match the resident count tensor",
                )
        if not torch.equal(metrics["recurrent_upkeep_charge_total"], tensors[28]):
            raise _fail(
                "prepared.selection_metrics.recurrent_upkeep_charge_total",
                "must match the resident upkeep tensor",
            )
        if not torch.equal(metrics[GMNW_SELECTION_METRICS_STATUS_NAME_V1], tensors[29]):
            raise _fail(
                "prepared.selection_metrics.status",
                "must match the resident status tensor",
            )
    return tensors


def _validated_policy_family_tags(
    prepared: PreparedDirectResidentGmnwGruV1,
    *,
    expected_shape: tuple[int, int],
) -> torch.Tensor:
    source = getattr(getattr(prepared, "patch", None), "source", None)
    live_state = getattr(source, "live_state", None)
    if not isinstance(live_state, Mapping):
        raise _fail("prepared.patch.source.live_state", "must be a mapping")
    family_tags = live_state.get("policy_use_gru")
    if (
        not isinstance(family_tags, torch.Tensor)
        or family_tags.dtype != torch.bool
        or tuple(family_tags.shape) != expected_shape
    ):
        raise _fail(
            "prepared.patch.source.live_state.policy_use_gru",
            "must be bool with the GMNW per-slot shape",
        )
    return family_tags


def _validated_prepared_selection_metrics(
    prepared: PreparedDirectResidentGmnwGruV1,
    *,
    cells: int,
    generations: int,
    generation: int,
    generation_base: int,
    require_zero_future: bool,
) -> tuple[torch.Tensor, ...]:
    metrics = getattr(prepared, "selection_metrics", None)
    if metrics is None:
        return ()
    if not isinstance(metrics, Mapping):
        raise _fail("prepared.selection_metrics", "must be a mapping")
    if not metrics:
        return ()
    if any(type(name) is not str for name in metrics):
        raise _fail("prepared.selection_metrics", "keys must be exact strings")
    expected_names = set(_GMNW_CHECKPOINT_SELECTION_METRIC_NAMES_SEL1_V1)
    if set(metrics) != expected_names:
        missing = sorted(expected_names - set(metrics))
        extra = sorted(set(metrics) - expected_names)
        raise _fail(
            "prepared.selection_metrics",
            f"field set mismatch; missing={missing}, extra={extra}",
        )

    metric_shape = (cells, generations, len(GMNW_SELECTION_POLICY_FAMILIES_V1))
    expected_shapes = (
        *(metric_shape for _ in GMNW_SELECTION_METRIC_NAMES_V1),
        (cells,),
    )
    validated: list[torch.Tensor] = []
    for name, dtype, shape in zip(
        _GMNW_CHECKPOINT_SELECTION_METRIC_NAMES_SEL1_V1,
        _GMNW_CHECKPOINT_SELECTION_METRIC_DTYPES_SEL1_V1,
        expected_shapes,
        strict=True,
    ):
        tensor = metrics[name]
        if not isinstance(tensor, torch.Tensor):
            raise _fail(f"prepared.selection_metrics.{name}", "must be a tensor")
        if tensor.dtype != dtype:
            raise _fail(
                f"prepared.selection_metrics.{name}.dtype",
                f"expected {_dtype_name(dtype)}, got {_dtype_name(tensor.dtype)}",
            )
        if tuple(tensor.shape) != shape:
            raise _fail(
                f"prepared.selection_metrics.{name}.shape",
                f"expected {list(shape)}, got {list(tensor.shape)}",
            )
        validated.append(tensor)

    if bool(torch.count_nonzero(validated[-1]).item()):
        raise _fail("prepared.selection_metrics.status", "must be zero")
    completed_rows = generation - generation_base
    if require_zero_future:
        for name, tensor in zip(
            GMNW_SELECTION_METRIC_NAMES_V1,
            validated[:-1],
            strict=True,
        ):
            if bool(torch.count_nonzero(tensor[:, completed_rows:]).item()):
                raise _fail(
                    f"prepared.selection_metrics.{name}",
                    "future rows after the checkpoint generation must be zero",
                )
    return tuple(validated)


def _validated_generation_for_config(
    absolute_generation: Any, config: Mapping[str, Any], *, path: str
) -> int:
    generation = _required_absolute_generation(absolute_generation, path=path)
    generation_base = config["generation_base"]
    final_generation = generation_base + config["generation_rows_per_cell"]
    if not generation_base <= generation <= final_generation:
        raise _fail(
            path,
            f"must be within [{generation_base}, {final_generation}] for GMNW config",
        )
    return generation


def _validated_occupancy(
    prepared: PreparedDirectResidentGmnwGruV1, stream_id: torch.Tensor
) -> None:
    source = getattr(getattr(prepared, "patch", None), "source", None)
    live_state = getattr(source, "live_state", None)
    if not isinstance(live_state, Mapping):
        raise _fail("prepared.patch.source.live_state", "must be a mapping")
    alive_mask = live_state.get("alive_mask")
    slot_to_agent_id = live_state.get("slot_to_agent_id")
    expected_shape = tuple(stream_id.shape)
    if (
        not isinstance(alive_mask, torch.Tensor)
        or alive_mask.dtype != torch.bool
        or tuple(alive_mask.shape) != expected_shape
    ):
        raise _fail(
            "prepared.patch.source.live_state.alive_mask",
            "must be bool with the GMNW per-slot shape",
        )
    if (
        not isinstance(slot_to_agent_id, torch.Tensor)
        or slot_to_agent_id.dtype != torch.int64
        or tuple(slot_to_agent_id.shape) != expected_shape
    ):
        raise _fail(
            "prepared.patch.source.live_state.slot_to_agent_id",
            "must be int64 with the GMNW per-slot shape",
        )

    alive = alive_mask.detach().to(device="cpu").contiguous()
    agent_ids = slot_to_agent_id.detach().to(device="cpu").contiguous()
    occupied = agent_ids >= 0
    if not torch.equal(alive, occupied):
        raise _fail(
            "prepared.patch.source.live_state",
            "alive_mask and slot_to_agent_id occupancy disagree",
        )
    expected_stream_id = torch.where(alive, agent_ids, torch.full_like(agent_ids, -1))
    observed_stream_id = stream_id.detach().to(device="cpu").contiguous()
    if not torch.equal(observed_stream_id, expected_stream_id):
        raise _fail(
            "tensors.stream_id",
            "does not match prepared live-state occupancy and agent identity",
        )


def _manifest_for_tensors(
    names: tuple[str, ...], tensors: tuple[torch.Tensor, ...]
) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "name": name,
            "dtype": _dtype_name(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": _tensor_sha256(tensor),
        }
        for index, (name, tensor) in enumerate(zip(names, tensors, strict=True))
    ]


def _checkpoint_tensor_contract(
    *,
    cells: int,
    capacity: int,
    generations: int,
    sel1_version: int,
    selection_metrics: bool = False,
) -> tuple[tuple[str, ...], tuple[torch.dtype, ...], tuple[tuple[int, ...], ...]]:
    rows = (cells, capacity)
    base_shapes = (
        *(rows for _ in range(14)),
        (cells, generations, GMNW_COUNTER_COLUMNS_V1),
        (cells, generations),
    )
    if sel1_version == 0:
        return (
            GMNW_CHECKPOINT_TENSOR_NAMES_V1,
            _GMNW_TENSOR_DTYPES_V1[:16],
            base_shapes,
        )
    sel1_shapes_v1 = (*base_shapes, *((cells,) for _ in range(7)), rows)
    if sel1_version == 1:
        names = GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_V1
        dtypes = _GMNW_CHECKPOINT_TENSOR_DTYPES_SEL1_V1
        sel1_shapes = sel1_shapes_v1
    else:
        names = GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_V2
        dtypes = _GMNW_CHECKPOINT_TENSOR_DTYPES_SEL1_V2
        sel1_shapes = (*sel1_shapes_v1, (cells,))
    if not selection_metrics:
        return names, dtypes, sel1_shapes
    metric_shape = (cells, generations, len(GMNW_SELECTION_POLICY_FAMILIES_V1))
    return (
        (
            _GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_WITH_METRICS_V1
            if sel1_version == 1
            else _GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_WITH_METRICS_V2
        ),
        (
            *dtypes,
            *_GMNW_CHECKPOINT_SELECTION_METRIC_DTYPES_SEL1_V1,
        ),
        (
            *sel1_shapes,
            *(metric_shape for _ in GMNW_SELECTION_METRIC_NAMES_V1),
            (cells,),
        ),
    )


def _digest_payload(sidecar: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": sidecar["schema_version"],
        "codec_version": sidecar["codec_version"],
        "absolute_generation": sidecar["absolute_generation"],
        "gmnw_config": sidecar["gmnw_config"],
        "tensor_manifest": sidecar["tensor_manifest"],
        "manifest_sha256": sidecar["manifest_sha256"],
    }


def _multicell_digest_payload(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": checkpoint["schema_version"],
        "codec_version": checkpoint["codec_version"],
        "absolute_generation": checkpoint["absolute_generation"],
        "cell_count": checkpoint["cell_count"],
        "cell_ids": checkpoint["cell_ids"],
        "sidecar_digests": [
            sidecar.get("digest_sha256") if isinstance(sidecar, Mapping) else None
            for sidecar in checkpoint["sidecars"]
        ],
    }


def _built_sidecar(
    *,
    config: Mapping[str, Any],
    generation: int,
    tensors: tuple[torch.Tensor, ...],
    sel1: bool,
    selection_metrics: bool = False,
) -> dict[str, Any]:
    if selection_metrics and not sel1:
        raise AssertionError("selection metrics require an SEL1 sidecar")
    names = (
        _GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_WITH_METRICS_V2
        if selection_metrics
        else (
            GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_V2
            if sel1
            else GMNW_CHECKPOINT_TENSOR_NAMES_V1
        )
    )
    manifest = _manifest_for_tensors(names, tensors)
    sidecar: dict[str, Any] = {
        "schema_version": (
            GMNW_CHECKPOINT_SIDECAR_SCHEMA_VERSION_SEL1_V2
            if sel1
            else GMNW_CHECKPOINT_SIDECAR_SCHEMA_VERSION_V1
        ),
        "codec_version": (
            GMNW_CHECKPOINT_SIDECAR_CODEC_VERSION_SEL1_V2
            if sel1
            else GMNW_CHECKPOINT_SIDECAR_CODEC_VERSION_V1
        ),
        "absolute_generation": generation,
        "gmnw_config": dict(config),
        "tensor_manifest": manifest,
        "manifest_sha256": _canonical_sha256(manifest, path="tensor_manifest"),
        "tensors": dict(zip(names, tensors, strict=True)),
    }
    sidecar["digest_sha256"] = _canonical_sha256(
        _digest_payload(sidecar), path="sidecar"
    )
    return sidecar


def _validated_cell_ids(
    value: Any, *, expected_count: int, path: str, serialized: bool
) -> tuple[str, ...]:
    expected_type = list if serialized else (list, tuple)
    if not isinstance(value, expected_type) or len(value) != expected_count:
        raise _fail(path, f"must contain exactly {expected_count} cell identifiers")
    cell_ids = tuple(value)
    for index, cell_id in enumerate(cell_ids):
        if type(cell_id) is not str or not cell_id:
            raise _fail(f"{path}[{index}]", "must be a nonempty exact string")
    if len(set(cell_ids)) != expected_count:
        raise _fail(path, "must contain unique cell identifiers in canonical order")
    return cell_ids


def _validate_distinct_tensor_storages(
    tensors: Sequence[torch.Tensor], names: Sequence[str], *, path: str
) -> None:
    occupied: dict[tuple[str, int], str] = {}
    for name, tensor in zip(names, tensors, strict=True):
        if not tensor.numel():
            continue
        key = (str(tensor.device), int(tensor.untyped_storage().data_ptr()))
        previous = occupied.get(key)
        if previous is not None:
            raise _fail(path, f"tensor storage alias: {name!r} aliases {previous!r}")
        occupied[key] = name


def _built_multicell_checkpoint(
    *,
    cell_ids: tuple[str, ...],
    generation: int,
    sidecars: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    checkpoint: dict[str, Any] = {
        "schema_version": GMNW_MULTICELL_CHECKPOINT_SCHEMA_VERSION_SEL1_V2,
        "codec_version": GMNW_MULTICELL_CHECKPOINT_CODEC_VERSION_SEL1_V2,
        "absolute_generation": generation,
        "cell_count": len(cell_ids),
        "cell_ids": list(cell_ids),
        "sidecars": list(sidecars),
    }
    checkpoint["digest_sha256"] = _canonical_sha256(
        _multicell_digest_payload(checkpoint), path="multicell_checkpoint"
    )
    return checkpoint


def export_direct_resident_gmnw_checkpoint_v1(
    prepared: PreparedDirectResidentGmnwGruV1,
    *,
    absolute_generation: int,
) -> dict[str, Any]:
    """Export mutable integrated GMNW state as detached contiguous CPU clones."""

    prepared_tensors = _validated_prepared_tensors(prepared)
    config, extended = _validated_prepared_config(
        prepared.gmnw_config, path="prepared.gmnw_config"
    )
    generation = _validated_generation_for_config(
        absolute_generation, config, path="absolute_generation"
    )
    _validated_occupancy(prepared, prepared_tensors[0])

    checkpoint_sources: tuple[torch.Tensor, ...]
    selection_metrics = False
    if extended:
        if len(prepared_tensors) not in (27, 30):
            raise _fail(
                "prepared.gmnw_tensors",
                "control checkpoints require the 27- or 30-tensor resident ABI",
            )
        cells, capacity = (int(extent) for extent in prepared_tensors[0].shape)
        family_tags = _validated_policy_family_tags(
            prepared, expected_shape=(cells, capacity)
        )
        expected_codes = torch.tensor(
            config["policy_arm_codes"],
            dtype=prepared_tensors[23].dtype,
            device=prepared_tensors[23].device,
        )
        if not torch.equal(prepared_tensors[23], expected_codes):
            raise _fail(
                "prepared.gmnw_tensors[23]",
                "policy arm codes contradict prepared.gmnw_config",
            )
        metric_sources = _validated_prepared_selection_metrics(
            prepared,
            cells=cells,
            generations=config["generation_rows_per_cell"],
            generation=generation,
            generation_base=config["generation_base"],
            require_zero_future=True,
        )
        selection_metrics = bool(metric_sources)
        checkpoint_sources = (
            *prepared_tensors[:22],
            prepared_tensors[23],
            family_tags,
            prepared_tensors[22].to(dtype=torch.int32),
            *metric_sources,
        )
    else:
        metrics = getattr(prepared, "selection_metrics", None)
        if metrics is not None and not isinstance(metrics, Mapping):
            raise _fail("prepared.selection_metrics", "must be a mapping")
        if metrics:
            raise _fail(
                "prepared.selection_metrics",
                "selection metrics require an extended control prepared owner",
            )
        checkpoint_sources = prepared_tensors[:16]

    cpu_tensors = tuple(
        tensor.detach().to(device="cpu").contiguous().clone()
        for tensor in checkpoint_sources
    )
    return _built_sidecar(
        config=config,
        generation=generation,
        tensors=cpu_tensors,
        sel1=extended,
        selection_metrics=selection_metrics,
    )


def _validated_serialized_sidecar(
    sidecar: Mapping[str, Any],
) -> tuple[dict[str, Any], int, tuple[torch.Tensor, ...], bool, bool]:
    """Validate a sidecar without trusting or requiring a prepared owner."""

    if not isinstance(sidecar, Mapping):
        raise _fail("sidecar", "must be a mapping")
    if set(sidecar) != _SIDECAR_FIELDS_V1:
        missing = sorted(_SIDECAR_FIELDS_V1 - set(sidecar))
        extra = sorted(set(sidecar) - _SIDECAR_FIELDS_V1)
        raise _fail("sidecar", f"field set mismatch; missing={missing}, extra={extra}")
    schema_version = sidecar["schema_version"]
    if schema_version == GMNW_CHECKPOINT_SIDECAR_SCHEMA_VERSION_V1:
        sel1_version = 0
        expected_codec_version = GMNW_CHECKPOINT_SIDECAR_CODEC_VERSION_V1
    elif schema_version == GMNW_CHECKPOINT_SIDECAR_SCHEMA_VERSION_SEL1_V1:
        sel1_version = 1
        expected_codec_version = GMNW_CHECKPOINT_SIDECAR_CODEC_VERSION_SEL1_V1
    elif schema_version == GMNW_CHECKPOINT_SIDECAR_SCHEMA_VERSION_SEL1_V2:
        sel1_version = 2
        expected_codec_version = GMNW_CHECKPOINT_SIDECAR_CODEC_VERSION_SEL1_V2
    else:
        raise _fail("sidecar.schema_version", "mismatch")
    sel1 = sel1_version != 0
    if (
        type(sidecar["codec_version"]) is not int
        or sidecar["codec_version"] != expected_codec_version
    ):
        raise _fail("sidecar.codec_version", "mismatch")

    config = _validated_serialized_config(
        sidecar["gmnw_config"],
        path="sidecar.gmnw_config",
        sel1_version=sel1_version,
    )
    generation = _validated_generation_for_config(
        sidecar["absolute_generation"],
        config,
        path="sidecar.absolute_generation",
    )
    tensor_map = sidecar["tensors"]
    if not isinstance(tensor_map, Mapping):
        raise _fail("sidecar.tensors", "must be a mapping")
    if any(type(name) is not str for name in tensor_map):
        raise _fail("sidecar.tensors", "keys must be exact strings")
    tensor_names = set(tensor_map)
    selection_metrics = False
    if sel1:
        base_names = (
            GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_V1
            if sel1_version == 1
            else GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_V2
        )
        metric_names = (
            _GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_WITH_METRICS_V1
            if sel1_version == 1
            else _GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_WITH_METRICS_V2
        )
    else:
        base_names = GMNW_CHECKPOINT_TENSOR_NAMES_V1
        metric_names = ()
    if sel1 and tensor_names == set(metric_names):
        selection_metrics = True
        names = metric_names
    elif sel1 and tensor_names == set(base_names):
        names = base_names
    elif not sel1 and tensor_names == set(GMNW_CHECKPOINT_TENSOR_NAMES_V1):
        names = GMNW_CHECKPOINT_TENSOR_NAMES_V1
    else:
        expected_names = set(
            metric_names
            if sel1
            and tensor_names.intersection(_GMNW_CHECKPOINT_SELECTION_METRIC_NAMES_SEL1_V1)
            else (
                base_names
                if sel1
                else GMNW_CHECKPOINT_TENSOR_NAMES_V1
            )
        )
        missing = sorted(expected_names - tensor_names)
        extra = sorted(tensor_names - expected_names)
        raise _fail(
            "sidecar.tensors",
            f"field set mismatch; missing={missing}, extra={extra}",
        )

    manifest = sidecar["tensor_manifest"]
    expected_tensor_count = len(names)
    if not isinstance(manifest, list) or len(manifest) != expected_tensor_count:
        raise _fail(
            "sidecar.tensor_manifest",
            f"must be the exact {expected_tensor_count}-row manifest",
        )
    stream_id = tensor_map["stream_id"]
    if not isinstance(stream_id, torch.Tensor) or stream_id.ndim != 2:
        raise _fail("sidecar.tensors.stream_id", "must be a rank-2 tensor")
    cells, capacity = (int(extent) for extent in stream_id.shape)
    if cells <= 0 or capacity <= 0:
        raise _fail("sidecar.tensors.stream_id", "must have positive extents")
    generations = config["generation_rows_per_cell"]
    names, dtypes, expected_shapes = _checkpoint_tensor_contract(
        cells=cells,
        capacity=capacity,
        generations=generations,
        sel1_version=sel1_version,
        selection_metrics=selection_metrics,
    )

    validated: list[torch.Tensor] = []
    for index, (name, dtype, shape, row) in enumerate(
        zip(
            names,
            dtypes,
            expected_shapes,
            manifest,
            strict=True,
        )
    ):
        if not isinstance(row, Mapping) or set(row) != _MANIFEST_ROW_FIELDS_V1:
            raise _fail(f"sidecar.tensor_manifest[{index}]", "field set mismatch")
        descriptor = {
            "index": index,
            "name": name,
            "dtype": _dtype_name(dtype),
            "shape": list(shape),
        }
        if {key: row.get(key) for key in descriptor} != descriptor:
            raise _fail(
                f"sidecar.tensor_manifest[{index}]",
                "index/name/dtype/shape descriptor mismatch",
            )
        tensor = tensor_map[name]
        if not isinstance(tensor, torch.Tensor):
            raise _fail(f"sidecar.tensors.{name}", "must be a tensor")
        if tensor.device.type != "cpu":
            raise _fail(f"sidecar.tensors.{name}.device", "must be CPU")
        if tensor.dtype != dtype:
            raise _fail(
                f"sidecar.tensors.{name}.dtype",
                f"expected {_dtype_name(dtype)}, got {_dtype_name(tensor.dtype)}",
            )
        if tuple(tensor.shape) != shape:
            raise _fail(
                f"sidecar.tensors.{name}.shape",
                f"expected {list(shape)}, got {list(tensor.shape)}",
            )
        if not tensor.is_contiguous() or tensor.requires_grad:
            raise _fail(
                f"sidecar.tensors.{name}",
                "must be contiguous, detached checkpoint storage",
            )
        digest = row["sha256"]
        if type(digest) is not str or _tensor_sha256(tensor) != digest:
            raise _fail(f"sidecar.tensors.{name}.sha256", "digest mismatch")
        validated.append(tensor)

    if sel1_version == 1:
        metric_offset = len(GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_V1)
        profile_codes = torch.zeros((cells,), dtype=torch.int32, device="cpu")
        validated = [
            *validated[:metric_offset],
            profile_codes,
            *validated[metric_offset:],
        ]

    if sel1:
        policy_arm_codes = validated[22]
        expected_codes = torch.tensor(config["policy_arm_codes"], dtype=torch.int8)
        if not torch.equal(policy_arm_codes, expected_codes):
            raise _fail(
                "sidecar.tensors.policy_arm_code",
                "contradicts sidecar.gmnw_config",
            )
        profile_index = GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_V2.index(
            "venv_consequence_profile_code"
        )
        profile_codes = validated[profile_index]
        expected_profile_codes = torch.tensor(
            config["venv_consequence_profile_codes"], dtype=torch.int32
        )
        if not torch.equal(profile_codes, expected_profile_codes):
            raise _fail(
                "sidecar.tensors.venv_consequence_profile_code",
                "contradicts sidecar.gmnw_config",
            )
        if bool(torch.count_nonzero(validated[16]).item()):
            raise _fail("sidecar.tensors.task_status", "must be zero")
    if selection_metrics:
        metric_offset = len(GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_V2)
        metrics = validated[metric_offset:]
        if bool(torch.count_nonzero(metrics[-1]).item()):
            raise _fail("sidecar.tensors.status", "must be zero")
        completed_rows = generation - config["generation_base"]
        for name, tensor in zip(
            GMNW_SELECTION_METRIC_NAMES_V1,
            metrics[:-1],
            strict=True,
        ):
            if bool(torch.count_nonzero(tensor[:, completed_rows:]).item()):
                raise _fail(
                    f"sidecar.tensors.{name}",
                    "future rows after the checkpoint generation must be zero",
                )

    manifest_digest = sidecar["manifest_sha256"]
    if (
        type(manifest_digest) is not str
        or _canonical_sha256(manifest, path="sidecar.tensor_manifest")
        != manifest_digest
    ):
        raise _fail("sidecar.manifest_sha256", "digest mismatch")
    sidecar_digest = sidecar["digest_sha256"]
    if (
        type(sidecar_digest) is not str
        or _canonical_sha256(_digest_payload(sidecar), path="sidecar") != sidecar_digest
    ):
        raise _fail("sidecar.digest_sha256", "digest mismatch")
    return config, generation, tuple(validated), sel1, selection_metrics


def expand_direct_resident_gmnw_checkpoint_v1(
    sidecar: Mapping[str, Any],
    *,
    generation_rows_per_cell: int,
) -> dict[str, Any]:
    """Extend a validated checkpoint with zero future aggregate rows."""

    config, generation, tensors, sel1, selection_metrics = (
        _validated_serialized_sidecar(sidecar)
    )
    source_rows = config["generation_rows_per_cell"]
    if (
        type(generation_rows_per_cell) is not int
        or generation_rows_per_cell <= source_rows
    ):
        raise _fail(
            "generation_rows_per_cell",
            f"must be an exact integer greater than {source_rows}",
        )

    expanded_tensors = list(
        tensor.detach().to(device="cpu").contiguous().clone() for tensor in tensors
    )
    added_rows = generation_rows_per_cell - source_rows
    for index in (14, 15):
        tensor = expanded_tensors[index]
        padding_shape = (tensor.size(0), added_rows, *tensor.shape[2:])
        padding = torch.zeros(padding_shape, dtype=tensor.dtype, device="cpu")
        expanded_tensors[index] = torch.cat((tensor, padding), dim=1).contiguous()
    if selection_metrics:
        metric_offset = len(GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_V2)
        for index in range(metric_offset, len(expanded_tensors) - 1):
            tensor = expanded_tensors[index]
            padding_shape = (tensor.size(0), added_rows, *tensor.shape[2:])
            padding = torch.zeros(padding_shape, dtype=tensor.dtype, device="cpu")
            expanded_tensors[index] = torch.cat((tensor, padding), dim=1).contiguous()

    expanded_config = dict(config)
    expanded_config["generation_rows_per_cell"] = generation_rows_per_cell
    return _built_sidecar(
        config=expanded_config,
        generation=generation,
        tensors=tuple(expanded_tensors),
        sel1=sel1,
        selection_metrics=selection_metrics,
    )


def _single_cell_control_config(
    config: Mapping[str, Any], *, cell: int
) -> dict[str, Any]:
    single = dict(config)
    for name in _CONTROL_PER_CELL_CONFIG_NAMES_V1:
        single[name] = [config[name][cell]]
    return single


def export_direct_resident_gmnw_multicell_checkpoint_v1(
    prepared: PreparedDirectResidentGmnwGruV1,
    *,
    absolute_generation: int,
    cell_ids: Sequence[str],
) -> dict[str, Any]:
    """Export one strict, storage-private control sidecar per ordered cell."""

    batched = export_direct_resident_gmnw_checkpoint_v1(
        prepared, absolute_generation=absolute_generation
    )
    config, generation, tensors, extended, selection_metrics = (
        _validated_serialized_sidecar(batched)
    )
    if not extended or config["control_matrix_enabled"] is not True:
        raise _fail(
            "prepared.gmnw_config",
            "multicell checkpoints require an enabled control matrix",
        )
    cell_count = int(tensors[0].size(0))
    ordered_ids = _validated_cell_ids(
        cell_ids,
        expected_count=cell_count,
        path="cell_ids",
        serialized=False,
    )

    sidecars: list[dict[str, Any]] = []
    for cell in range(cell_count):
        private_tensors = tuple(
            tensor[cell : cell + 1].contiguous().clone() for tensor in tensors
        )
        sidecars.append(
            _built_sidecar(
                config=_single_cell_control_config(config, cell=cell),
                generation=generation,
                tensors=private_tensors,
                sel1=True,
                selection_metrics=selection_metrics,
            )
        )
    result = _built_multicell_checkpoint(
        cell_ids=ordered_ids,
        generation=generation,
        sidecars=tuple(sidecars),
    )
    _validated_multicell_checkpoint(result, expected_cell_ids=ordered_ids)
    return result


def _validated_multicell_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    expected_cell_ids: Sequence[str] | None = None,
) -> tuple[tuple[str, ...], int, tuple[dict[str, Any], ...]]:
    if not isinstance(checkpoint, Mapping):
        raise _fail("multicell_checkpoint", "must be a mapping")
    if set(checkpoint) != _MULTICELL_CHECKPOINT_FIELDS_SEL1_V1:
        missing = sorted(_MULTICELL_CHECKPOINT_FIELDS_SEL1_V1 - set(checkpoint))
        extra = sorted(set(checkpoint) - _MULTICELL_CHECKPOINT_FIELDS_SEL1_V1)
        raise _fail(
            "multicell_checkpoint",
            f"field set mismatch; missing={missing}, extra={extra}",
        )
    multicell_versions = {
        GMNW_MULTICELL_CHECKPOINT_SCHEMA_VERSION_SEL1_V1: (
            GMNW_MULTICELL_CHECKPOINT_CODEC_VERSION_SEL1_V1
        ),
        GMNW_MULTICELL_CHECKPOINT_SCHEMA_VERSION_SEL1_V2: (
            GMNW_MULTICELL_CHECKPOINT_CODEC_VERSION_SEL1_V2
        ),
    }
    schema_version = checkpoint["schema_version"]
    if schema_version not in multicell_versions:
        raise _fail("multicell_checkpoint.schema_version", "mismatch")
    expected_sidecar_schema = (
        GMNW_CHECKPOINT_SIDECAR_SCHEMA_VERSION_SEL1_V1
        if schema_version == GMNW_MULTICELL_CHECKPOINT_SCHEMA_VERSION_SEL1_V1
        else GMNW_CHECKPOINT_SIDECAR_SCHEMA_VERSION_SEL1_V2
    )
    if (
        type(checkpoint["codec_version"]) is not int
        or checkpoint["codec_version"]
        != multicell_versions[schema_version]
    ):
        raise _fail("multicell_checkpoint.codec_version", "mismatch")
    cell_count = checkpoint["cell_count"]
    if type(cell_count) is not int or not 1 <= cell_count <= 72:
        raise _fail(
            "multicell_checkpoint.cell_count", "must be an exact integer in [1,72]"
        )
    cell_ids = _validated_cell_ids(
        checkpoint["cell_ids"],
        expected_count=cell_count,
        path="multicell_checkpoint.cell_ids",
        serialized=True,
    )
    if expected_cell_ids is not None:
        expected = _validated_cell_ids(
            expected_cell_ids,
            expected_count=cell_count,
            path="cell_ids",
            serialized=False,
        )
        if cell_ids != expected:
            raise _fail(
                "multicell_checkpoint.cell_ids",
                "cell count/order does not exactly match the requested continuation",
            )
    generation = _required_absolute_generation(
        checkpoint["absolute_generation"],
        path="multicell_checkpoint.absolute_generation",
    )
    raw_sidecars = checkpoint["sidecars"]
    if not isinstance(raw_sidecars, list) or len(raw_sidecars) != cell_count:
        raise _fail(
            "multicell_checkpoint.sidecars",
            f"must contain exactly {cell_count} ordered sidecars",
        )

    validated_sidecars: list[dict[str, Any]] = []
    first_shared_config: dict[str, Any] | None = None
    first_layout: tuple[tuple[torch.dtype, tuple[int, ...]], ...] | None = None
    first_selection_metrics: bool | None = None
    all_tensors: list[torch.Tensor] = []
    all_names: list[str] = []
    for cell, raw_sidecar in enumerate(raw_sidecars):
        if (
            not isinstance(raw_sidecar, Mapping)
            or raw_sidecar.get("schema_version") != expected_sidecar_schema
        ):
            raise _fail(
                f"multicell_checkpoint.sidecars[{cell}].schema_version",
                "does not match the outer multicell checkpoint version",
            )
        config, sidecar_generation, tensors, sel1, selection_metrics = (
            _validated_serialized_sidecar(raw_sidecar)
        )
        if not sel1:
            raise _fail(
                f"multicell_checkpoint.sidecars[{cell}].schema_version",
                "must be the extended control sidecar schema",
            )
        if sidecar_generation != generation:
            raise _fail(
                f"multicell_checkpoint.sidecars[{cell}].absolute_generation",
                "does not match the multicell checkpoint",
            )
        if int(tensors[0].size(0)) != 1:
            raise _fail(
                f"multicell_checkpoint.sidecars[{cell}]",
                "must contain exactly one cell",
            )
        if any(len(config[name]) != 1 for name in _CONTROL_PER_CELL_CONFIG_NAMES_V1):
            raise _fail(
                f"multicell_checkpoint.sidecars[{cell}].gmnw_config",
                "must contain exactly one arm/cue/founder/profile metadata row",
            )
        shared_config = {
            name: value
            for name, value in config.items()
            if name not in _CONTROL_PER_CELL_CONFIG_NAMES_V1
        }
        layout = tuple((tensor.dtype, tuple(tensor.shape[1:])) for tensor in tensors)
        if first_shared_config is None:
            first_shared_config = shared_config
            first_layout = layout
            first_selection_metrics = selection_metrics
        else:
            if _canonical_json_bytes(
                shared_config,
                path=f"multicell_checkpoint.sidecars[{cell}].gmnw_config",
            ) != _canonical_json_bytes(
                first_shared_config,
                path="multicell_checkpoint.sidecars[0].gmnw_config",
            ):
                raise _fail(
                    f"multicell_checkpoint.sidecars[{cell}].gmnw_config",
                    "shared metadata/config drift",
                )
            if layout != first_layout:
                raise _fail(
                    f"multicell_checkpoint.sidecars[{cell}].tensors",
                    "dtype/shape schema drift",
                )
            if selection_metrics is not first_selection_metrics:
                raise _fail(
                    f"multicell_checkpoint.sidecars[{cell}].tensors",
                    "selection metrics presence drift",
                )
        all_tensors.extend(tensors)
        names = (
            _GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_WITH_METRICS_V2
            if selection_metrics
            else GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_V2
        )
        all_names.extend(f"cell={cell}:{name}" for name in names)
        validated_sidecars.append(dict(raw_sidecar))

    _validate_distinct_tensor_storages(
        all_tensors,
        all_names,
        path="multicell_checkpoint.sidecars",
    )
    digest = checkpoint["digest_sha256"]
    if (
        type(digest) is not str
        or _canonical_sha256(
            _multicell_digest_payload(checkpoint), path="multicell_checkpoint"
        )
        != digest
    ):
        raise _fail("multicell_checkpoint.digest_sha256", "digest/order mismatch")
    return cell_ids, generation, tuple(validated_sidecars)


def validate_direct_resident_gmnw_multicell_checkpoint_v1(
    checkpoint: Mapping[str, Any],
    *,
    expected_cell_ids: Sequence[str] | None = None,
    expected_absolute_generation: int | None = None,
) -> tuple[tuple[str, ...], int]:
    """Validate a reloaded multicell checkpoint without restoring its state."""

    cell_ids, generation, _sidecars = _validated_multicell_checkpoint(
        checkpoint,
        expected_cell_ids=expected_cell_ids,
    )
    if expected_absolute_generation is not None:
        expected_generation = _required_absolute_generation(
            expected_absolute_generation,
            path="expected_absolute_generation",
        )
        if generation != expected_generation:
            raise _fail(
                "multicell_checkpoint.absolute_generation",
                "does not exactly match the expected durable checkpoint boundary",
            )
    return cell_ids, generation


def direct_resident_gmnw_cell_sidecar_digest_v1(
    checkpoint: Mapping[str, Any], *, cell_id: str
) -> str:
    """Return one validated cell's task-sidecar digest, independent of batch layout."""

    if type(cell_id) is not str or not cell_id:
        raise _fail("cell_id", "must be a nonempty exact string")
    cell_ids, _generation, sidecars = _validated_multicell_checkpoint(checkpoint)
    try:
        index = cell_ids.index(cell_id)
    except ValueError as exc:
        raise _fail("cell_id", "is absent from the multicell checkpoint") from exc
    digest = sidecars[index].get("digest_sha256")
    if type(digest) is not str or len(digest) != 64:
        raise _fail("sidecar.digest_sha256", "must be a validated SHA-256 digest")
    return digest


def _joined_multicell_sidecar(
    sidecars: tuple[dict[str, Any], ...], *, generation: int
) -> dict[str, Any]:
    validated = tuple(_validated_serialized_sidecar(sidecar) for sidecar in sidecars)
    configs = tuple(item[0] for item in validated)
    tensors_by_cell = tuple(item[2] for item in validated)
    selection_metrics = validated[0][4]
    config = dict(configs[0])
    for name in _CONTROL_PER_CELL_CONFIG_NAMES_V1:
        config[name] = [cell_config[name][0] for cell_config in configs]
    tensors = tuple(
        torch.cat([cell_tensors[index] for cell_tensors in tensors_by_cell], dim=0)
        .contiguous()
        .clone()
        for index in range(len(tensors_by_cell[0]))
    )
    return _built_sidecar(
        config=config,
        generation=generation,
        tensors=tensors,
        sel1=True,
        selection_metrics=selection_metrics,
    )


def restore_direct_resident_gmnw_multicell_checkpoint_v1(
    prepared: PreparedDirectResidentGmnwGruV1,
    checkpoint: Mapping[str, Any],
    *,
    absolute_generation: int,
    cell_ids: Sequence[str],
) -> int:
    """Atomically validate and restore an ordered collection of private cells."""

    _cell_ids, generation, sidecars = _validated_multicell_checkpoint(
        checkpoint, expected_cell_ids=cell_ids
    )
    if generation != absolute_generation:
        raise _fail(
            "multicell_checkpoint.absolute_generation",
            f"expected {absolute_generation}, got {generation}",
        )
    return restore_direct_resident_gmnw_checkpoint_v1(
        prepared,
        _joined_multicell_sidecar(sidecars, generation=generation),
        absolute_generation=absolute_generation,
    )


def continue_direct_resident_gmnw_multicell_checkpoint_v1(
    prepared: PreparedDirectResidentGmnwGruV1,
    checkpoint: Mapping[str, Any],
    *,
    absolute_generation: int,
    cell_ids: Sequence[str],
) -> int:
    """Restore private cells into an equal or longer fresh continuation owner."""

    ordered_ids, generation, sidecars = _validated_multicell_checkpoint(
        checkpoint, expected_cell_ids=cell_ids
    )
    target_config, target_sel1 = _validated_prepared_config(
        prepared.gmnw_config, path="prepared.gmnw_config"
    )
    if not target_sel1:
        raise _fail(
            "prepared.gmnw_config",
            "continuation target must use the extended control contract",
        )
    source_config, _, _, _, _ = _validated_serialized_sidecar(sidecars[0])
    source_rows = source_config["generation_rows_per_cell"]
    target_rows = target_config["generation_rows_per_cell"]
    if target_rows < source_rows:
        raise _fail(
            "prepared.gmnw_config.generation_rows_per_cell",
            f"continuation cannot shrink checkpoint rows below {source_rows}",
        )
    if target_rows > source_rows:
        sidecars = tuple(
            expand_direct_resident_gmnw_checkpoint_v1(
                sidecar, generation_rows_per_cell=target_rows
            )
            for sidecar in sidecars
        )
    expanded = _built_multicell_checkpoint(
        cell_ids=ordered_ids,
        generation=generation,
        sidecars=sidecars,
    )
    return restore_direct_resident_gmnw_multicell_checkpoint_v1(
        prepared,
        expanded,
        absolute_generation=absolute_generation,
        cell_ids=ordered_ids,
    )


def _validated_sidecar(
    prepared: PreparedDirectResidentGmnwGruV1,
    sidecar: Mapping[str, Any],
    *,
    absolute_generation: int,
) -> tuple[tuple[torch.Tensor, ...], bool, bool]:
    target_config, target_sel1 = _validated_prepared_config(
        prepared.gmnw_config, path="prepared.gmnw_config"
    )
    sidecar_config, sidecar_generation, restored, sidecar_sel1, selection_metrics = (
        _validated_serialized_sidecar(sidecar)
    )
    if target_sel1 and not sidecar_sel1:
        raise _fail(
            "sidecar.schema_version",
            "legacy v1 sidecars are incomplete for an extended control target",
        )
    if sidecar_sel1 and not target_sel1:
        raise _fail("sidecar.schema_version", "SEL1 sidecar requires an SEL1 target")
    comparable_target_config = (
        target_config
        if sidecar_sel1
        else _validated_core_config(
            prepared.gmnw_config,
            path="prepared.gmnw_config",
            require_exact_fields=False,
        )
    )
    if _canonical_json_bytes(
        sidecar_config, path="sidecar.gmnw_config"
    ) != _canonical_json_bytes(comparable_target_config, path="prepared.gmnw_config"):
        raise _fail("sidecar.gmnw_config", "does not exactly match prepared config")

    expected_generation = _validated_generation_for_config(
        absolute_generation, target_config, path="absolute_generation"
    )
    if sidecar_generation != expected_generation:
        raise _fail(
            "sidecar.absolute_generation",
            f"expected {expected_generation}, got {sidecar_generation}",
        )

    target_tensors = _validated_prepared_tensors(prepared)
    if sidecar_sel1:
        if len(target_tensors) not in (27, 30):
            raise _fail(
                "prepared.gmnw_tensors",
                "control restore requires the 27- or 30-tensor resident ABI",
            )
        family_target = _validated_policy_family_tags(
            prepared, expected_shape=tuple(target_tensors[0].shape)
        )
        cells = int(target_tensors[0].size(0))
        metric_targets = _validated_prepared_selection_metrics(
            prepared,
            cells=cells,
            generations=target_config["generation_rows_per_cell"],
            generation=expected_generation,
            generation_base=target_config["generation_base"],
            require_zero_future=False,
        )
        if bool(metric_targets) is not selection_metrics:
            raise _fail(
                "prepared.selection_metrics",
                "presence must exactly match the checkpoint",
            )
        checkpoint_targets = (
            *target_tensors[:22],
            target_tensors[23],
            family_target,
            target_tensors[22],
            *metric_targets,
        )
        names = (
            _GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_WITH_METRICS_V2
            if selection_metrics
            else GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_V2
        )
        _validate_distinct_tensor_storages(
            checkpoint_targets[: len(GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_V2)],
            names[: len(GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_V2)],
            path="prepared SEL1 checkpoint targets",
        )
    else:
        checkpoint_targets = target_tensors[:16]
        names = GMNW_CHECKPOINT_TENSOR_NAMES_V1
    for name, target, tensor in zip(names, checkpoint_targets, restored, strict=True):
        if tensor.dtype != target.dtype or tuple(tensor.shape) != tuple(target.shape):
            raise _fail(
                f"sidecar.tensors.{name}",
                "does not match the prepared tensor dtype and shape",
            )
    _validated_occupancy(prepared, restored[0])
    return restored, sidecar_sel1, selection_metrics


def restore_direct_resident_gmnw_checkpoint_v1(
    prepared: PreparedDirectResidentGmnwGruV1,
    sidecar: Mapping[str, Any],
    *,
    absolute_generation: int,
) -> int:
    """Validate then restore mutable GMNW state into an existing fresh owner."""

    restored, sel1, selection_metrics = _validated_sidecar(
        prepared, sidecar, absolute_generation=absolute_generation
    )
    target_tensors = prepared.gmnw_tensors
    with torch.no_grad():
        if sel1:
            family_target = _validated_policy_family_tags(
                prepared, expected_shape=tuple(target_tensors[0].shape)
            )
            metric_targets = (
                tuple(
                    prepared.selection_metrics[name]
                    for name in _GMNW_CHECKPOINT_SELECTION_METRIC_NAMES_SEL1_V1
                )
                if selection_metrics
                else ()
            )
            targets = (
                *target_tensors[:22],
                target_tensors[23],
                family_target,
                target_tensors[22],
                *metric_targets,
            )
            for target, source in zip(targets, restored, strict=True):
                target.copy_(source, non_blocking=False)
        else:
            for target, source in zip(target_tensors[:16], restored, strict=True):
                target.copy_(source, non_blocking=False)
            target_tensors[16].zero_()
    return absolute_generation


__all__ = [
    "GMNW_CHECKPOINT_SIDECAR_CODEC_VERSION_V1",
    "GMNW_CHECKPOINT_SIDECAR_CODEC_VERSION_SEL1_V1",
    "GMNW_CHECKPOINT_SIDECAR_CODEC_VERSION_SEL1_V2",
    "GMNW_CHECKPOINT_SIDECAR_SCHEMA_VERSION_V1",
    "GMNW_CHECKPOINT_SIDECAR_SCHEMA_VERSION_SEL1_V1",
    "GMNW_CHECKPOINT_SIDECAR_SCHEMA_VERSION_SEL1_V2",
    "GMNW_CHECKPOINT_TENSOR_NAMES_V1",
    "GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_V1",
    "GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_V2",
    "GMNW_MULTICELL_CHECKPOINT_CODEC_VERSION_SEL1_V1",
    "GMNW_MULTICELL_CHECKPOINT_CODEC_VERSION_SEL1_V2",
    "GMNW_MULTICELL_CHECKPOINT_SCHEMA_VERSION_SEL1_V1",
    "GMNW_MULTICELL_CHECKPOINT_SCHEMA_VERSION_SEL1_V2",
    "DirectResidentGmnwCheckpointV1Error",
    "continue_direct_resident_gmnw_multicell_checkpoint_v1",
    "direct_resident_gmnw_cell_sidecar_digest_v1",
    "expand_direct_resident_gmnw_checkpoint_v1",
    "export_direct_resident_gmnw_multicell_checkpoint_v1",
    "export_direct_resident_gmnw_checkpoint_v1",
    "restore_direct_resident_gmnw_multicell_checkpoint_v1",
    "restore_direct_resident_gmnw_checkpoint_v1",
    "validate_direct_resident_gmnw_multicell_checkpoint_v1",
]
