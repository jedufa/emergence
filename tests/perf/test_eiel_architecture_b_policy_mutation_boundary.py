from __future__ import annotations

import copy
import dataclasses
import hashlib
import runpy
from pathlib import Path
from typing import Any

import pytest
import torch

from engine.native_runtime import run_controller as run_controller_module
from engine.native_runtime.compact_summary_authority_adapter import (
    assemble_compact_summary_source,
)
from engine.native_runtime.compact_summary_digests import (
    POLICY_MUTATION_TAPE_SCHEMA_VERSION,
    policy_mutation_tape_digest_from_tensor_sha256,
    policy_mutation_tape_digest_sha256,
)
from engine.native_runtime.run_controller import (
    NATIVE_BOUNDARY_SCHEMA_VERSION,
    NATIVE_CHECKPOINT_PROJECTION_SCHEMA_VERSIONS,
    NATIVE_POLICY_MUTATION_TAPE_BOUNDARY_AUTHORITY_SCHEMA_VERSION,
    NativeRunController,
    NativeRunControllerError,
)
from engine.perf.cuda_architecture_b_torch_rng_state_pack_v1 import (
    pack_torch_cpu_rng_state_v1,
)


ROOT = Path(__file__).resolve().parents[2]
ADAPTER_FIXTURES = runpy.run_path(
    str(ROOT / "tests/test_eiel_compact_summary_authority_adapter.py")
)
RUN_ID = "policy-tape-boundary-run"
CONTROLLER_ID = "policy-tape-boundary-controller"
OWNER_EPOCH = 7
TAPE_FIELDS = (
    "policy_birth_crossover_mask_tape",
    "policy_birth_delta_tape",
    "policy_birth_mutation_mask_tape",
    "policy_birth_post_rng_state_tape",
    "policy_birth_semantic_draws_tape",
)


def _adapter_inputs() -> Any:
    fixture = ADAPTER_FIXTURES["_inputs"]
    namespace = fixture.__globals__
    original_projection = namespace["_projection"]

    def projection_with_runtime_cursor(*args: Any, **kwargs: Any) -> dict[str, Any]:
        value = original_projection(*args, **kwargs)
        value.setdefault("runtime_event_cursor", value["event_cursor"])
        return value

    namespace["_projection"] = projection_with_runtime_cursor
    try:
        return fixture()
    finally:
        namespace["_projection"] = original_projection


def _resident_tape(cursor: int) -> dict[str, torch.Tensor]:
    return {
        "policy_birth_crossover_mask_tape": torch.tensor(
            [[False, True], [True, False], [False, False]], dtype=torch.bool
        ),
        "policy_birth_delta_tape": torch.tensor(
            [[0.25, -0.5], [1.0, 2.0], [3.0, 4.0]], dtype=torch.float32
        ),
        "policy_birth_mutation_mask_tape": torch.tensor(
            [[True, False], [False, True], [True, True]], dtype=torch.bool
        ),
        "policy_birth_post_rng_state_tape": torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            dtype=torch.uint8,
        ),
        "policy_birth_semantic_draws_tape": torch.tensor(
            [2, 4, 6], dtype=torch.int64
        ),
        "policy_birth_tape_cursor": torch.tensor([cursor], dtype=torch.int64),
    }


def _authority(tape: dict[str, torch.Tensor], cursor: int) -> dict[str, Any]:
    prefixes = {
        name: tape[name][:cursor].detach().to("cpu").contiguous()
        for name in TAPE_FIELDS
    }
    return {
        "schema_version": (
            NATIVE_POLICY_MUTATION_TAPE_BOUNDARY_AUTHORITY_SCHEMA_VERSION
        ),
        "source_version": POLICY_MUTATION_TAPE_SCHEMA_VERSION,
        "cursor": cursor,
        "consumed_shapes": {
            name: [int(size) for size in prefix.shape]
            for name, prefix in prefixes.items()
        },
        "tensor_sha256": {
            name: hashlib.sha256(
                prefix.numpy().tobytes(order="C")
            ).hexdigest()
            for name, prefix in prefixes.items()
        },
        "authority_complete": True,
        "export_non_consuming": True,
    }


def _projection(tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "tensors": tensors,
        "tensor_manifest": {
            name: {"bytes": int(tensor.numel() * tensor.element_size())}
            for name, tensor in tensors.items()
        },
    }


def _boundary(offset: int, *, cursor: int) -> dict[str, Any]:
    tape = _resident_tape(cursor)
    pre = _projection({"state": torch.tensor([offset], dtype=torch.int64)})
    post = _projection({"state": torch.tensor([offset + 1], dtype=torch.int64)})
    return {
        "schema_version": NATIVE_BOUNDARY_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "controller_id": CONTROLLER_ID,
        "owner_epoch": OWNER_EPOCH,
        "snapshot_id": (
            f"{CONTROLLER_ID}:owner:{OWNER_EPOCH}:boundary:{offset + 1}"
        ),
        "snapshot_sequence": offset + 1,
        "completed_generation": offset,
        "next_generation": offset + 1,
        "tick": 0,
        "phase": "generation_complete",
        "controller_status_at_clone": (
            "generation_complete_ready_to_resolve_terminal"
        ),
        "generation_result": {"executed": True, "hidden_fallback": False},
        "generation_end_reporting_frame": pre,
        "generation_end_reporting_frame_role": (
            "transaction_local_generation_end_values_folded_after_turnover_commit"
        ),
        "turnover_result": {"executed": True, "hidden_fallback": False},
        "post_turnover_checkpoint_projection": post,
        "policy_mutation_tape": tape,
        "generation_end_reporting_frame_tensor_bytes": 8,
        "post_turnover_projection_tensor_bytes": sum(
            int(tensor.numel() * tensor.element_size())
            for tensor in post["tensors"].values()
        ),
        "policy_mutation_tape_authority": _authority(tape, cursor),
    }


@pytest.fixture
def controller(monkeypatch: pytest.MonkeyPatch) -> NativeRunController:
    monkeypatch.setattr(
        run_controller_module,
        "_projection_digest_manifests",
        lambda _projection: {},
    )
    value = object.__new__(NativeRunController)
    value._start_generation = 0
    value._run_id = RUN_ID
    value._controller_id = CONTROLLER_ID
    value._owner_epoch = OWNER_EPOCH
    return value


def test_g2_boundaries_export_adapter_ready_prefixes_in_generation_order(
    controller: NativeRunController,
) -> None:
    normalized = [
        controller._normalize_boundary(_boundary(offset, cursor=cursor), offset=offset)
        for offset, cursor in enumerate((0, 2))
    ]

    assert [row["completed_generation"] for row in normalized] == [0, 1]
    assert [
        int(row["policy_mutation_tape"]["policy_birth_tape_cursor"].item())
        for row in normalized
    ] == [0, 2]
    for row in normalized:
        authority = row["policy_mutation_tape_authority"]
        tape = row["policy_mutation_tape"]
        assert policy_mutation_tape_digest_sha256(tape) == (
            policy_mutation_tape_digest_from_tensor_sha256(
                authority["cursor"],
                authority["consumed_shapes"],
                authority["tensor_sha256"],
            )
        )
        adapter_inputs = dataclasses.replace(
            _adapter_inputs(),
            policy_mutation_tape=tape,
        )
        source = assemble_compact_summary_source(adapter_inputs)
        assert source.rng["authority_complete"] is True
        assert source.rng["export_non_consuming"] is True


def test_boundary_accepts_packed_post_rng_state_with_canonical_authority(
    controller: NativeRunController,
) -> None:
    cursor = 2
    boundary = _boundary(0, cursor=cursor)
    generator = torch.Generator(device="cpu").manual_seed(20260731)
    full_rows = []
    for offset in range(3):
        torch.rand((11 + offset,), generator=generator, dtype=torch.float64)
        full_rows.append(generator.get_state().clone())
    full_state_rows = torch.stack(full_rows).contiguous()
    packed_state_rows = pack_torch_cpu_rng_state_v1(full_state_rows)
    tape = boundary["policy_mutation_tape"]
    tape["policy_birth_post_rng_state_packed_tape"] = packed_state_rows
    del tape["policy_birth_post_rng_state_tape"]
    authority = boundary["policy_mutation_tape_authority"]
    full_prefix = full_state_rows[:cursor].contiguous()
    authority["consumed_shapes"]["policy_birth_post_rng_state_tape"] = [
        cursor,
        int(full_prefix.size(1)),
    ]
    authority["tensor_sha256"]["policy_birth_post_rng_state_tape"] = (
        hashlib.sha256(full_prefix.numpy().tobytes(order="C")).hexdigest()
    )

    normalized = controller._normalize_boundary(boundary, offset=0)

    assert "policy_birth_post_rng_state_packed_tape" in normalized[
        "policy_mutation_tape"
    ]
    assert "policy_birth_post_rng_state_tape" not in normalized[
        "policy_mutation_tape"
    ]


def test_boundary_accepts_replay_backed_post_rng_state_authority(
    controller: NativeRunController,
) -> None:
    cursor = 2
    boundary = _boundary(0, cursor=cursor)
    generator = torch.Generator(device="cpu").manual_seed(20260731)
    full_rows = []
    for draws in (2, 4, 6):
        torch.rand((draws,), generator=generator, dtype=torch.float32)
        full_rows.append(generator.get_state().clone())
    full_state_rows = torch.stack(full_rows).contiguous()
    tape = boundary["policy_mutation_tape"]
    tape["policy_birth_post_rng_state_packed_tape"] = torch.empty(
        (0, 0), dtype=torch.uint8
    )
    del tape["policy_birth_post_rng_state_tape"]
    authority = boundary["policy_mutation_tape_authority"]
    full_prefix = full_state_rows[:cursor].contiguous()
    authority["consumed_shapes"]["policy_birth_post_rng_state_tape"] = [
        cursor,
        int(full_prefix.size(1)),
    ]
    authority["tensor_sha256"]["policy_birth_post_rng_state_tape"] = (
        hashlib.sha256(full_prefix.numpy().tobytes(order="C")).hexdigest()
    )

    normalized = controller._normalize_boundary(boundary, offset=0)

    assert tuple(
        normalized["policy_mutation_tape"][
            "policy_birth_post_rng_state_packed_tape"
        ].shape
    ) == (0, 0)
    assert "policy_birth_post_rng_state_tape" not in normalized[
        "policy_mutation_tape"
    ]


def test_boundary_rejects_missing_policy_tape_authority_precisely(
    controller: NativeRunController,
) -> None:
    boundary = _boundary(0, cursor=1)
    del boundary["policy_mutation_tape_authority"]

    with pytest.raises(
        NativeRunControllerError,
        match=r"boundaries\[0\]\.policy_mutation_tape_authority required authority is absent",
    ):
        controller._normalize_boundary(boundary, offset=0)


@pytest.mark.parametrize(
    "schema_version",
    sorted(NATIVE_CHECKPOINT_PROJECTION_SCHEMA_VERSIONS),
)
def test_native_checkpoint_boundary_requires_policy_tape_authority(
    controller: NativeRunController,
    schema_version: str,
) -> None:
    boundary = _boundary(0, cursor=1)
    boundary["policy_mutation_tape"] = {}
    boundary["post_turnover_checkpoint_projection"]["schema_version"] = (
        schema_version
    )
    del boundary["policy_mutation_tape_authority"]

    with pytest.raises(
        NativeRunControllerError,
        match=r"boundaries\[0\]\.policy_mutation_tape_authority required authority is absent",
    ):
        controller._normalize_boundary(boundary, offset=0)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda authority: authority["tensor_sha256"].pop(
                "policy_birth_delta_tape"
            ),
            "tensor_sha256 fields mismatch",
        ),
        (
            lambda authority: authority["consumed_shapes"].__setitem__(
                "policy_birth_delta_tape", [2, 2]
            ),
            "consumed_shapes.policy_birth_delta_tape contradicts the consumed prefix",
        ),
        (
            lambda authority: authority["tensor_sha256"].__setitem__(
                "policy_birth_delta_tape", "0" * 64
            ),
            "tensor_sha256.policy_birth_delta_tape contradicts the consumed prefix",
        ),
        (
            lambda authority: authority.__setitem__("export_non_consuming", False),
            "export_non_consuming must be True",
        ),
    ],
)
def test_boundary_rejects_malformed_policy_tape_authority(
    controller: NativeRunController,
    mutate: Any,
    message: str,
) -> None:
    boundary = _boundary(0, cursor=1)
    mutate(boundary["policy_mutation_tape_authority"])

    with pytest.raises(NativeRunControllerError, match=message):
        controller._normalize_boundary(boundary, offset=0)


def test_normalization_is_read_only_for_resident_tape(
    controller: NativeRunController,
) -> None:
    boundary = _boundary(0, cursor=2)
    resident_before = {
        name: tensor.clone()
        for name, tensor in boundary["post_turnover_checkpoint_projection"][
            "tensors"
        ].items()
    }

    controller._normalize_boundary(copy.deepcopy(boundary), offset=0)

    for name, expected in resident_before.items():
        assert torch.equal(
            boundary["post_turnover_checkpoint_projection"]["tensors"][name],
            expected,
        )
