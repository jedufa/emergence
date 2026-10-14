from __future__ import annotations

"""Same-shape, one-context CUDA controller for Architecture B Stage 1B."""

from dataclasses import dataclass, replace
from os import PathLike
from time import perf_counter_ns
from typing import Any, Final, Mapping

import torch

from perf.cuda_architecture_b_generation import (
    CudaGenerationConfig,
    CudaGenerationResult,
    _cuda_generation_result,
    _prepare_cuda_generation_call,
)
from perf.cuda_architecture_b_generation_birth_state import (
    CudaGenerationBirthState,
)


CUDA_STAGE1B_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_stage2_tick_synchronous_same_shape_v1"
)


class CudaStage1BError(RuntimeError):
    """Raised before dispatch when a cell group is not an isolated same-shape bank."""


@dataclass(frozen=True)
class CudaStage1BCell:
    cell_id: str
    tensors: Mapping[str, torch.Tensor]
    birth_state: CudaGenerationBirthState
    config: CudaGenerationConfig
    deterministic: bool


@dataclass(frozen=True)
class CudaStage1BResidentCell:
    cell_id: str
    tensors: dict[str, torch.Tensor]
    birth_state: CudaGenerationBirthState
    config: CudaGenerationConfig
    deterministic: bool


@dataclass(frozen=True)
class CudaStage1BReceipt:
    schema_version: str
    cell_count: int
    stream_count: int
    device: str
    leading_cell_bank_count: int
    all_banks_contiguous: bool
    one_cuda_context: bool
    python_tick_iterations: int
    python_generation_boundary_dispatches: int
    native_generation_calls: int
    native_multicell_calls: int
    physical_native_generation_dispatches: int
    tick_synchronous_cross_cell_controller: bool
    batched_policy_world_execution: bool
    policy_world_batch_mode: str
    host_transfer_between_ticks: bool
    device_synchronizations: int
    generation_wall_time_ns: int
    peak_allocated_bytes: int
    final_allocated_bytes: int
    final_reserved_bytes: int
    total_device_bytes: int
    vram_headroom_fraction: float
    vram_headroom_qualified: bool
    hidden_fallback: bool = False
    cancellation_isolation_claimed: bool = False
    exact_cell_parity_claimed: bool = False
    throughput_qualified: bool = False
    promotion_qualified: bool = False


@dataclass(frozen=True)
class CudaStage1BExecution:
    cells: tuple[CudaStage1BResidentCell, ...]
    results: tuple[CudaGenerationResult, ...]
    receipt: CudaStage1BReceipt


def _tensor_storage_key(tensor: torch.Tensor) -> tuple[str, int]:
    return (str(tensor.device), int(tensor.untyped_storage()._cdata))


def _tensor_view_key(tensor: torch.Tensor) -> tuple[Any, ...]:
    pointer: int | tuple[str, int]
    if tensor.numel() == 0:
        pointer = ("empty", int(tensor.untyped_storage()._cdata))
    else:
        pointer = int(tensor.data_ptr())
    return (
        _tensor_storage_key(tensor),
        pointer,
        int(tensor.storage_offset()),
        tuple(int(value) for value in tensor.shape),
        tuple(int(value) for value in tensor.stride()),
        tensor.dtype,
        str(tensor.device),
    )


def _flatten_cell(cell: CudaStage1BCell) -> dict[str, torch.Tensor]:
    birth = cell.birth_state
    out = {f"resident.{name}": tensor for name, tensor in cell.tensors.items()}
    for prefix, values in (
        ("birth.identity", birth.identity_state),
        ("birth.scalar_rng", birth.scalar_rng_state),
        ("birth.genome", birth.genome_state),
        ("birth.drive", birth.drive_state),
        ("birth.policy", birth.policy_state),
        ("birth.turnover_policy", birth.turnover_policy_state),
    ):
        out.update({f"{prefix}.{name}": tensor for name, tensor in values.items()})
    out.update(
        {
            "birth.policy_mem_dim": birth.policy_mem_dim,
            "birth.policy_use_gru": birth.policy_use_gru,
            "birth.planner_pcg64_state_words": birth.planner_pcg64_state_words,
            "birth.planner_pcg64_buffer_words": birth.planner_pcg64_buffer_words,
            "birth.policy_np_semantic_draws": birth.policy_np_semantic_draws,
            "birth.runner_pcg64_state_words": birth.runner_pcg64_state_words,
            "birth.runner_pcg64_buffer_words": birth.runner_pcg64_buffer_words,
            "birth.metrics_np_semantic_draws": birth.metrics_np_semantic_draws,
            "birth.event_tape_active_count": birth.event_tape_active_count,
            "birth.event_tape_status": birth.event_tape_status,
        }
    )
    out.update(
        {
            f"birth.event_tape_columns.{index}": tensor
            for index, tensor in enumerate(birth.event_tape_columns)
        }
    )
    return out


def _alias_map(flat: Mapping[str, torch.Tensor]) -> dict[str, str]:
    representatives: dict[tuple[Any, ...], str] = {}
    views_by_storage: dict[tuple[str, int], tuple[tuple[Any, ...], str]] = {}
    result: dict[str, str] = {}
    for path in sorted(flat):
        tensor = flat[path]
        key = _tensor_view_key(tensor)
        storage_key = _tensor_storage_key(tensor)
        previous = views_by_storage.get(storage_key)
        if previous is not None and previous[0] != key:
            raise CudaStage1BError(
                "Stage 1B cannot preserve aliasing: distinct tensor views "
                f"share storage at paths {previous[1]!r} and {path!r}"
            )
        views_by_storage.setdefault(storage_key, (key, path))
        representative = representatives.setdefault(key, path)
        result[path] = representative
    return result


def _mapped_values(
    views: Mapping[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    dotted = f"{prefix}."
    return {
        path[len(dotted) :]: tensor
        for path, tensor in views.items()
        if path.startswith(dotted)
    }


def _rebuild_birth_state(
    source: CudaGenerationBirthState,
    views: Mapping[str, torch.Tensor],
) -> CudaGenerationBirthState:
    event_columns = tuple(
        views[f"birth.event_tape_columns.{index}"]
        for index in range(len(source.event_tape_columns))
    )
    return replace(
        source,
        identity_state=_mapped_values(views, "birth.identity"),
        scalar_rng_state=_mapped_values(views, "birth.scalar_rng"),
        genome_state=_mapped_values(views, "birth.genome"),
        drive_state=_mapped_values(views, "birth.drive"),
        policy_state=_mapped_values(views, "birth.policy"),
        turnover_policy_state=_mapped_values(views, "birth.turnover_policy"),
        policy_mem_dim=views["birth.policy_mem_dim"],
        policy_use_gru=views["birth.policy_use_gru"],
        planner_pcg64_state_words=views["birth.planner_pcg64_state_words"],
        planner_pcg64_buffer_words=views["birth.planner_pcg64_buffer_words"],
        policy_np_semantic_draws=views["birth.policy_np_semantic_draws"],
        runner_pcg64_state_words=views["birth.runner_pcg64_state_words"],
        runner_pcg64_buffer_words=views["birth.runner_pcg64_buffer_words"],
        metrics_np_semantic_draws=views["birth.metrics_np_semantic_draws"],
        event_tape_columns=event_columns,
        event_tape_active_count=views["birth.event_tape_active_count"],
        event_tape_status=views["birth.event_tape_status"],
    )


class CudaStage1BController:
    def __init__(
        self,
        cells: tuple[CudaStage1BResidentCell, ...],
        banks: Mapping[str, torch.Tensor],
        *,
        max_streams: int,
    ) -> None:
        self._cells = cells
        self._banks = dict(banks)
        self._max_streams = min(len(cells), max_streams)
        self._bank_by_view: dict[tuple[Any, ...], torch.Tensor] = {}
        self._resident_bank_by_storage: dict[tuple[str, int], str] = {}
        for path, bank in self._banks.items():
            self._resident_bank_by_storage[_tensor_storage_key(bank)] = path
            for index in range(len(cells)):
                self._bank_by_view[_tensor_view_key(bank[index])] = bank

    def _stack_native_argument(
        self,
        values: tuple[Any, ...],
        transient_banks: dict[tuple[tuple[Any, ...], ...], torch.Tensor],
    ) -> Any:
        first = values[0]
        if isinstance(first, torch.Tensor):
            if not all(isinstance(value, torch.Tensor) for value in values):
                raise CudaStage1BError("native multicell tensor argument types differ")
            bank = self._bank_by_view.get(_tensor_view_key(first))
            if bank is not None and all(
                _tensor_view_key(bank[index]) == _tensor_view_key(value)
                for index, value in enumerate(values)
            ):
                return bank
            if any(
                value.shape != first.shape
                or value.stride() != first.stride()
                or value.dtype != first.dtype
                or value.device != first.device
                or not value.is_contiguous()
                for value in values[1:]
            ):
                raise CudaStage1BError(
                    "native multicell transient tensor shapes differ"
                )
            view_keys = tuple(_tensor_view_key(value) for value in values)
            bank = transient_banks.get(view_keys)
            if bank is not None:
                return bank

            storage_keys = tuple(_tensor_storage_key(value) for value in values)
            for index, storage_key in enumerate(storage_keys):
                resident_path = self._resident_bank_by_storage.get(storage_key)
                if resident_path is not None:
                    raise CudaStage1BError(
                        "Stage 1B cannot preserve aliasing: generation-prepared "
                        f"tensor view at cell={index} shares storage with resident "
                        f"bank {resident_path!r} but is not its exact bank view"
                    )

            storage_owner: dict[tuple[str, int], int] = {}
            for index, storage_key in enumerate(storage_keys):
                previous_index = storage_owner.setdefault(storage_key, index)
                if previous_index != index:
                    raise CudaStage1BError(
                        "Stage 1B cannot preserve aliasing: generation-prepared "
                        "tensor values from cells "
                        f"{previous_index} and {index} share storage"
                    )

            for previous_view_keys in transient_banks:
                previous_storage_keys = {
                    view_key[0] for view_key in previous_view_keys
                }
                for index, storage_key in enumerate(storage_keys):
                    if storage_key in previous_storage_keys:
                        raise CudaStage1BError(
                            "Stage 1B cannot preserve aliasing: generation-prepared "
                            f"transient tensor view at cell={index} shares storage "
                            "with a previously stacked argument but is not an exact "
                            "identical alias"
                        )

            bank = torch.stack(values, dim=0).contiguous()
            transient_banks[view_keys] = bank
            return bank
        if isinstance(first, (list, tuple)):
            if not all(
                isinstance(value, type(first)) and len(value) == len(first)
                for value in values
            ):
                raise CudaStage1BError("native multicell tensor-list shapes differ")
            return [
                self._stack_native_argument(
                    tuple(value[index] for value in values),
                    transient_banks,
                )
                for index in range(len(first))
            ]
        if any(value != first for value in values[1:]):
            raise CudaStage1BError(
                "native multicell scalar generation arguments differ"
            )
        return first

    @classmethod
    def from_cells(
        cls,
        cells: tuple[CudaStage1BCell, ...] | list[CudaStage1BCell],
        *,
        max_streams: int | None = None,
    ) -> "CudaStage1BController":
        source_cells = tuple(cells)
        if len(source_cells) < 2:
            raise CudaStage1BError("Stage 1B requires at least two cells")
        if len({cell.cell_id for cell in source_cells}) != len(source_cells):
            raise CudaStage1BError("Stage 1B cell IDs must be unique")
        if max_streams is None:
            max_streams = len(source_cells)
        if type(max_streams) is not int or max_streams <= 0:
            raise CudaStage1BError("max_streams must be a positive integer")

        first = source_cells[0]
        first_flat = _flatten_cell(first)
        if not first_flat:
            raise CudaStage1BError("Stage 1B cells must expose resident state")
        expected_paths = set(first_flat)
        expected_aliases = _alias_map(first_flat)
        device = first.tensors["alive_mask"].device
        if device.type != "cuda":
            raise CudaStage1BError("Stage 1B cells must be CUDA-resident")

        flattened = [first_flat]
        for cell in source_cells[1:]:
            if cell.config != first.config or cell.deterministic != first.deterministic:
                raise CudaStage1BError(
                    "Stage 1B cells must share one exact generation configuration"
                )
            current = _flatten_cell(cell)
            if set(current) != expected_paths:
                raise CudaStage1BError("Stage 1B cells expose different state surfaces")
            if _alias_map(current) != expected_aliases:
                raise CudaStage1BError("Stage 1B cells expose different alias graphs")
            flattened.append(current)

        source_storage_owners: dict[tuple[str, int], tuple[int, str]] = {}
        representatives = sorted(set(expected_aliases.values()))
        for index, current in enumerate(flattened):
            for representative in representatives:
                storage_key = _tensor_storage_key(current[representative])
                previous = source_storage_owners.setdefault(
                    storage_key,
                    (index, representative),
                )
                if previous[0] != index:
                    raise CudaStage1BError(
                        "Stage 1B source cells share tensor storage at "
                        f"cell={previous[0]} path={previous[1]!r} and "
                        f"cell={index} path={representative!r}"
                    )

        for path in sorted(expected_paths):
            reference = first_flat[path]
            if not reference.is_contiguous():
                raise CudaStage1BError(
                    f"Stage 1B state surface is not contiguous at cell=0 path={path}"
                )
            for index, current in enumerate(flattened[1:], start=1):
                candidate = current[path]
                if (
                    candidate.shape != reference.shape
                    or candidate.stride() != reference.stride()
                    or candidate.dtype != reference.dtype
                    or candidate.device != reference.device
                    or not candidate.is_contiguous()
                ):
                    raise CudaStage1BError(
                        f"Stage 1B state surface differs at cell={index} path={path}"
                    )

        banks: dict[str, torch.Tensor] = {}
        for representative in representatives:
            tensors = [flat[representative] for flat in flattened]
            banks[representative] = torch.stack(tensors, dim=0).contiguous()

        resident_cells: list[CudaStage1BResidentCell] = []
        for index, source in enumerate(source_cells):
            views = {
                path: banks[representative][index]
                for path, representative in expected_aliases.items()
            }
            resident_cells.append(
                CudaStage1BResidentCell(
                    cell_id=source.cell_id,
                    tensors=_mapped_values(views, "resident"),
                    birth_state=_rebuild_birth_state(source.birth_state, views),
                    config=source.config,
                    deterministic=source.deterministic,
                )
            )
        return cls(tuple(resident_cells), banks, max_streams=max_streams)

    @property
    def cells(self) -> tuple[CudaStage1BResidentCell, ...]:
        return self._cells

    @property
    def banks(self) -> Mapping[str, torch.Tensor]:
        return self._banks

    def execute_generation(
        self,
        *,
        build_directory: str | PathLike[str] | None = None,
    ) -> CudaStage1BExecution:
        device = self._cells[0].tensors["alive_mask"].device
        calls = tuple(
            _prepare_cuda_generation_call(
                cell.tensors,
                cell.config,
                deterministic=cell.deterministic,
                birth_state=cell.birth_state,
                build_directory=build_directory,
            )
            for cell in self._cells
        )
        module = calls[0].module
        if any(call.module is not module for call in calls[1:]):
            raise CudaStage1BError("Stage 1B cells resolved different native modules")
        argument_count = len(calls[0].arguments)
        if any(len(call.arguments) != argument_count for call in calls[1:]):
            raise CudaStage1BError("Stage 1B native argument counts differ")
        transient_banks: dict[
            tuple[tuple[Any, ...], ...], torch.Tensor
        ] = {}
        stacked_arguments = tuple(
            self._stack_native_argument(
                tuple(call.arguments[index] for call in calls),
                transient_banks,
            )
            for index in range(argument_count)
        )
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started_ns = perf_counter_ns()
        raw_results = module.eiel_cuda_architecture_b_generation_multicell_v2(
            *stacked_arguments,
            int(self._max_streams),
        )
        torch.cuda.synchronize(device)
        wall_time_ns = perf_counter_ns() - started_ns
        if len(raw_results) != len(self._cells):
            raise CudaStage1BError("native multicell result count differs from cells")
        results = tuple(
            _cuda_generation_result(outputs, cell.config)
            for outputs, cell in zip(raw_results, self._cells, strict=True)
        )
        properties = torch.cuda.get_device_properties(device)
        reserved = int(torch.cuda.memory_reserved(device))
        total = int(properties.total_memory)
        headroom = max(0.0, float(total - reserved) / float(total))
        receipt = CudaStage1BReceipt(
            schema_version=CUDA_STAGE1B_SCHEMA_VERSION,
            cell_count=len(self._cells),
            stream_count=self._max_streams,
            device=str(device),
            leading_cell_bank_count=len(self._banks),
            all_banks_contiguous=all(bank.is_contiguous() for bank in self._banks.values()),
            one_cuda_context=True,
            python_tick_iterations=0,
            python_generation_boundary_dispatches=1,
            native_generation_calls=len(results),
            native_multicell_calls=1,
            physical_native_generation_dispatches=1,
            tick_synchronous_cross_cell_controller=True,
            batched_policy_world_execution=True,
            policy_world_batch_mode=(
                "tick_synchronous_cross_cell_policy_cache_observation_mlp_"
                "lifecycle_v1"
            ),
            host_transfer_between_ticks=False,
            device_synchronizations=2,
            generation_wall_time_ns=wall_time_ns,
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            final_allocated_bytes=int(torch.cuda.memory_allocated(device)),
            final_reserved_bytes=reserved,
            total_device_bytes=total,
            vram_headroom_fraction=headroom,
            vram_headroom_qualified=headroom >= 0.10,
        )
        return CudaStage1BExecution(
            cells=self._cells,
            results=results,
            receipt=receipt,
        )


__all__ = [
    "CUDA_STAGE1B_SCHEMA_VERSION",
    "CudaStage1BCell",
    "CudaStage1BController",
    "CudaStage1BError",
    "CudaStage1BExecution",
    "CudaStage1BReceipt",
    "CudaStage1BResidentCell",
]
