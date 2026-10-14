from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import pytest
import torch

import native_runtime.run_controller as run_controller_module
from native_runtime.architecture_b_crossing_metrics import (
    architecture_b_crossing_payload_metrics,
)
from native_runtime.run_controller import (
    NATIVE_BOUNDARY_SCHEMA_VERSION,
    NATIVE_RESULT_SCHEMA_VERSION,
    NATIVE_SNAPSHOT_VALIDATION_RECEIPT_SCHEMA_VERSION,
    NativeRunController,
)


RUN_ID = "copy-accounting-run"
CONTROLLER_ID = "copy-accounting-controller"
OWNER_EPOCH = 3
START_GENERATION = 4
GENERATION_COUNT = 2
SOURCE_PROJECTION_DIGEST = "a" * 64


def _projection(values: Sequence[int]) -> dict[str, Any]:
    tensor = torch.tensor(values, dtype=torch.int64)
    return {
        "tensors": {"state": tensor},
        "tensor_manifest": {
            "state": {"bytes": tensor.numel() * tensor.element_size()}
        },
    }


def _projection_bytes(projection: Mapping[str, Any]) -> int:
    return sum(
        int(tensor.numel()) * int(tensor.element_size())
        for tensor in projection["tensors"].values()
    )


def _boundary(offset: int) -> dict[str, Any]:
    completed_generation = START_GENERATION + offset
    reporting = _projection([offset, offset + 1])
    post_turnover = _projection([offset + 10, offset + 11, offset + 12])
    reporting_bytes = _projection_bytes(reporting)
    post_turnover_bytes = _projection_bytes(post_turnover)
    return {
        "schema_version": NATIVE_BOUNDARY_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "controller_id": CONTROLLER_ID,
        "owner_epoch": OWNER_EPOCH,
        "snapshot_id": (
            f"{CONTROLLER_ID}:owner:{OWNER_EPOCH}:boundary:{offset + 1}"
        ),
        "snapshot_sequence": offset + 1,
        "completed_generation": completed_generation,
        "next_generation": completed_generation + 1,
        "tick": 0,
        "phase": "generation_complete",
        "controller_status_at_clone": (
            "generation_complete_ready_to_resolve_terminal"
        ),
        "generation_result": {"executed": True, "hidden_fallback": False},
        "generation_end_reporting_frame": reporting,
        "generation_end_reporting_frame_role": (
            "transaction_local_generation_end_values_folded_after_turnover_commit"
        ),
        "turnover_result": {"executed": True, "hidden_fallback": False},
        "post_turnover_checkpoint_projection": post_turnover,
        "generation_end_reporting_frame_tensor_bytes": reporting_bytes,
        "post_turnover_projection_tensor_bytes": post_turnover_bytes,
    }


def _native_result() -> dict[str, Any]:
    boundaries = [_boundary(offset) for offset in range(GENERATION_COUNT)]
    projection_tensor_bytes = sum(
        boundary["generation_end_reporting_frame_tensor_bytes"]
        + boundary["post_turnover_projection_tensor_bytes"]
        for boundary in boundaries
    )
    return {
        "schema_version": NATIVE_RESULT_SCHEMA_VERSION,
        "terminal_status": "awaiting_snapshot_validation",
        "run_id": RUN_ID,
        "controller_id": CONTROLLER_ID,
        "owner_epoch": OWNER_EPOCH,
        "executed": True,
        "native_owner": True,
        "hidden_fallback": False,
        "runtime_jit": False,
        "completed_generations": GENERATION_COUNT,
        "start_generation": START_GENERATION,
        "requested_generations": GENERATION_COUNT,
        "trajectory_commit_frontier": START_GENERATION + GENERATION_COUNT - 1,
        "snapshot_frontier": START_GENERATION - 1,
        "next_generation": START_GENERATION + GENERATION_COUNT,
        "next_tick": 0,
        "controller_invocations": 1,
        "python_generation_dispatches": 0,
        "native_generation_transactions": GENERATION_COUNT,
        "turnover_commits": GENERATION_COUNT,
        "boundary_snapshot_count": GENERATION_COUNT,
        "projection_export_count": GENERATION_COUNT * 2,
        "projection_tensor_bytes": projection_tensor_bytes,
        "final_snapshot_sequence": GENERATION_COUNT,
        "success_marker_allowed": False,
        "boundaries": boundaries,
    }


class _NativeController:
    def __init__(self) -> None:
        self.status = "initialized"
        self.run_started = False
        self.sealed = False
        self.native_result: dict[str, Any] | None = None

    def controller_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "eiel_native_architecture_b_controller_manifest_v1",
            "run_id": RUN_ID,
            "controller_id": CONTROLLER_ID,
            "owner_epoch": OWNER_EPOCH,
            "status": self.status,
            "run_started": self.run_started,
            "sealed": self.sealed,
            "native_owner_private": True,
            "python_callback_entry_points": 0,
        }

    def run_generations(
        self,
        start_generation: int,
        generation_count: int,
        steps_per_generation: int,
        max_births: int,
    ) -> dict[str, Any]:
        assert (start_generation, generation_count) == (
            START_GENERATION,
            GENERATION_COUNT,
        )
        assert steps_per_generation == 1
        assert max_births == 0
        self.status = "awaiting_snapshot_validation"
        self.run_started = True
        self.native_result = _native_result()
        return self.native_result

    def acknowledge_snapshot_validation(
        self, source_projection_digest: str
    ) -> dict[str, Any]:
        assert source_projection_digest == SOURCE_PROJECTION_DIGEST
        self.status = "run_complete"
        self.sealed = True
        return {
            "schema_version": NATIVE_SNAPSHOT_VALIDATION_RECEIPT_SCHEMA_VERSION,
            "run_id": RUN_ID,
            "controller_id": CONTROLLER_ID,
            "owner_epoch": OWNER_EPOCH,
            "acknowledged": True,
            "disposition": "accepted",
            "terminal_status": "run_complete",
            "trajectory_commit_frontier": START_GENERATION + GENERATION_COUNT - 1,
            "snapshot_frontier": START_GENERATION + GENERATION_COUNT - 1,
            "final_snapshot_sequence": GENERATION_COUNT,
            "validated_source_projection_digest": source_projection_digest,
            "success_marker_allowed": True,
        }

    def fail_snapshot_validation(self, message: str) -> dict[str, Any]:
        raise AssertionError(f"unexpected snapshot validation failure: {message}")


@pytest.fixture
def completed_run(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_NativeController, NativeRunController, dict[str, Any]]:
    monkeypatch.setattr(
        run_controller_module,
        "_projection_digest_manifests",
        lambda projection: {
            "source_projection": {"digest": SOURCE_PROJECTION_DIGEST}
        },
    )
    native_controller = _NativeController()
    controller = NativeRunController(
        native_controller,
        start_generation=START_GENERATION,
        default_steps_per_generation=1,
        default_max_births=0,
        safe_default_decision={"supported": True},
        extension_path="/not-loaded-in-this-test.so",
        run_id=RUN_ID,
    )
    result = controller.run_many(GENERATION_COUNT)
    return native_controller, controller, result


@pytest.fixture
def run_result(
    completed_run: tuple[_NativeController, NativeRunController, dict[str, Any]],
) -> dict[str, Any]:
    return completed_run[2]


def _walk_values(value: Any) -> Iterator[Any]:
    yield value
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_values(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            yield from _walk_values(item)


def _assert_final_projection_alias(result: Mapping[str, Any]) -> None:
    final_projection = result["final_checkpoint_projection"]
    last_post_turnover = result["boundaries"][-1][
        "post_turnover_checkpoint_projection"
    ]

    assert final_projection is last_post_turnover
    for name, final_tensor in final_projection["tensors"].items():
        boundary_tensor = last_post_turnover["tensors"][name]
        assert final_tensor is boundary_tensor
        assert final_tensor.untyped_storage().data_ptr() == (
            boundary_tensor.untyped_storage().data_ptr()
        )


def _final_state_tensor(result: Mapping[str, Any]) -> torch.Tensor:
    return result["final_checkpoint_projection"]["tensors"]["state"]


def test_detached_copy_clones_shared_tensor_storage_once() -> None:
    source = torch.arange(12, dtype=torch.int64).reshape(3, 4)
    source_view = source[:, 1:]

    copied = run_controller_module._detached_copy(
        {"tensor": source, "alias": source, "view": source_view}
    )

    assert copied["tensor"] is copied["alias"]
    assert copied["tensor"].untyped_storage().data_ptr() == (
        copied["view"].untyped_storage().data_ptr()
    )
    assert copied["tensor"].untyped_storage().data_ptr() != (
        source.untyped_storage().data_ptr()
    )
    assert copied["view"].shape == source_view.shape
    assert copied["view"].stride() == source_view.stride()
    assert copied["view"].storage_offset() == source_view.storage_offset()


def test_run_many_result_preserves_alias_and_detaches_native_input(
    completed_run: tuple[_NativeController, NativeRunController, dict[str, Any]],
) -> None:
    native_controller, _, result = completed_run
    assert native_controller.native_result is not None
    native_tensor = native_controller.native_result["boundaries"][-1][
        "post_turnover_checkpoint_projection"
    ]["tensors"]["state"]

    _assert_final_projection_alias(result)
    assert torch.equal(_final_state_tensor(result), native_tensor)
    assert _final_state_tensor(result).untyped_storage().data_ptr() != (
        native_tensor.untyped_storage().data_ptr()
    )


def test_cached_final_result_preserves_final_boundary_alias(
    completed_run: tuple[_NativeController, NativeRunController, dict[str, Any]],
) -> None:
    _, controller, immediate = completed_run

    cached = controller.final_result()

    _assert_final_projection_alias(cached)
    assert _final_state_tensor(cached).untyped_storage().data_ptr() != (
        _final_state_tensor(immediate).untyped_storage().data_ptr()
    )
    assert architecture_b_crossing_payload_metrics(cached)[
        "unique_tensor_storage_bytes"
    ] == cached["boundary_tensor_manifest_bytes"]


def test_repeat_final_result_calls_are_detached_from_caller_mutations(
    completed_run: tuple[_NativeController, NativeRunController, dict[str, Any]],
) -> None:
    native_controller, controller, immediate = completed_run
    assert native_controller.native_result is not None
    expected = torch.tensor([11, 12, 13], dtype=torch.int64)

    _final_state_tensor(immediate).fill_(-1)
    native_controller.native_result["boundaries"][-1][
        "post_turnover_checkpoint_projection"
    ]["tensors"]["state"].fill_(-2)
    first_cached = controller.final_result()
    assert torch.equal(_final_state_tensor(first_cached), expected)

    _final_state_tensor(first_cached).fill_(-3)
    first_cached["terminal_status"] = "caller-mutated"
    second_cached = controller.final_result()

    _assert_final_projection_alias(first_cached)
    _assert_final_projection_alias(second_cached)
    assert second_cached["terminal_status"] == "run_complete"
    assert torch.equal(_final_state_tensor(second_cached), expected)
    assert _final_state_tensor(first_cached).untyped_storage().data_ptr() != (
        _final_state_tensor(second_cached).untyped_storage().data_ptr()
    )


def test_unique_crossing_tensor_bytes_equal_boundary_manifest_bytes(
    run_result: dict[str, Any],
) -> None:
    metrics = architecture_b_crossing_payload_metrics(run_result)

    assert metrics["unique_tensor_storage_bytes"] == run_result[
        "boundary_tensor_manifest_bytes"
    ]


def test_snapshot_validation_receipt_contains_no_tensors_or_boundaries(
    run_result: dict[str, Any],
) -> None:
    receipt = run_result["snapshot_validation_receipt"]

    assert "boundaries" not in receipt
    assert not any(isinstance(value, torch.Tensor) for value in _walk_values(receipt))
    assert architecture_b_crossing_payload_metrics(receipt)[
        "unique_tensor_storage_bytes"
    ] == 0
