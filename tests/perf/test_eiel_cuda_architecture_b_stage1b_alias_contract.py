from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import perf.cuda_architecture_b_stage1b as stage1b
from perf.cuda_architecture_b_stage1b import (
    CudaStage1BController,
    CudaStage1BError,
)


def _empty_controller() -> CudaStage1BController:
    cells = (SimpleNamespace(), SimpleNamespace())
    return CudaStage1BController(cells, {}, max_streams=2)  # type: ignore[arg-type]


def test_flat_alias_map_preserves_exact_identical_views() -> None:
    tensor = torch.arange(6)

    aliases = stage1b._alias_map(
        {
            "resident.owner": tensor,
            "resident.alias": tensor.view_as(tensor),
        }
    )

    assert aliases == {
        "resident.alias": "resident.alias",
        "resident.owner": "resident.alias",
    }


def test_flat_alias_map_rejects_distinct_views_sharing_storage() -> None:
    tensor = torch.arange(6)

    with pytest.raises(
        CudaStage1BError,
        match=r"distinct tensor views share storage.*resident\.owner.*resident\.view",
    ):
        stage1b._alias_map(
            {
                "resident.owner": tensor,
                "resident.view": tensor[1:],
            }
        )


def test_generation_prepared_exact_alias_reuses_transient_bank() -> None:
    controller = _empty_controller()
    values = (torch.arange(4), torch.arange(4, 8))
    transient_banks: dict[tuple[tuple[object, ...], ...], torch.Tensor] = {}

    first = controller._stack_native_argument(values, transient_banks)
    repeated = controller._stack_native_argument(
        tuple(value.view_as(value) for value in values),
        transient_banks,
    )

    assert repeated is first
    assert len(transient_banks) == 1


def test_generation_prepared_distinct_view_of_resident_bank_fails_closed() -> None:
    cells = (SimpleNamespace(), SimpleNamespace())
    resident_bank = torch.arange(10).reshape(2, 5)
    controller = CudaStage1BController(
        cells,  # type: ignore[arg-type]
        {"resident.state": resident_bank},
        max_streams=2,
    )
    transient_banks: dict[tuple[tuple[object, ...], ...], torch.Tensor] = {}

    with pytest.raises(
        CudaStage1BError,
        match=r"generation-prepared tensor view.*resident bank 'resident\.state'",
    ):
        controller._stack_native_argument(
            (resident_bank[0, 1:], resident_bank[1, 1:]),
            transient_banks,
        )

    assert transient_banks == {}


def test_generation_prepared_distinct_storage_views_fail_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cells = tuple(
        SimpleNamespace(
            tensors={"alive_mask": torch.ones(1, dtype=torch.bool)},
            birth_state=object(),
            config=object(),
            deterministic=True,
        )
        for _ in range(2)
    )
    controller = CudaStage1BController(cells, {}, max_streams=2)  # type: ignore[arg-type]
    source_tensors = (torch.arange(5), torch.arange(5, 10))
    module = object()
    prepared_calls = iter(
        SimpleNamespace(
            module=module,
            arguments=(source[:-1], source[1:]),
        )
        for source in source_tensors
    )
    monkeypatch.setattr(
        stage1b,
        "_prepare_cuda_generation_call",
        lambda *_args, **_kwargs: next(prepared_calls),
    )

    with pytest.raises(
        CudaStage1BError,
        match=r"generation-prepared transient tensor view.*not an exact identical alias",
    ):
        controller.execute_generation()
