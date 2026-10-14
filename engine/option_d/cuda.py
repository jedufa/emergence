"""Standalone transactional CUDA backend for the frozen Option D contract."""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

from dataclasses import replace
from hashlib import sha256
import os
from pathlib import Path
import threading
from typing import Any, Final

import torch
from torch.utils.cpp_extension import load

from .checkpoint import CheckpointSnapshot, capture_checkpoint as _capture_checkpoint
from .contracts import (
    CpythonRNGState,
    DrawSchedule,
    ExecutionReceipt,
    GeneratedMaterial,
    GeneratedValue,
    GenerationResult,
    OptionDContractError,
    OptionDExecutionError,
    RNGCounters,
    RNGStateBundle,
    WorldExecutionReceipt,
    float64_bits,
    float64_from_bits,
)


_ROOT = Path(__file__).resolve().parent
_NATIVE = _ROOT / "native"
_COMPILE_SOURCES: Final[tuple[Path, ...]] = (
    _NATIVE / "option_d_cuda.cpp",
    _NATIVE / "option_d_cuda.cu",
)
_SOURCE_BINDING_FILES: Final[tuple[Path, ...]] = _COMPILE_SOURCES + (
    _NATIVE / "glibc_2_43_fma_math.cuh",
)
_COUNTER_FIELDS: Final[tuple[str, ...]] = tuple(RNGCounters.__dataclass_fields__)
_STATUS_NAMES: Final[dict[int, str]] = {
    0: "ok",
    1: "invalid_opcode",
    2: "invalid_getrandbits",
    3: "invalid_randbelow",
    4: "gauss_disabled",
    5: "invalid_gaussian_strategy",
    6: "counter_overflow",
    7: "invalid_predicate",
    8: "invalid_predicate_source",
    9: "host_tape_exhausted",
}
_GAUSSIAN_STRATEGIES: Final[dict[str, int]] = {
    "disabled": 0,
    "libdevice": 1,
    "software": 2,
}
_MODULES: dict[str, Any] = {}


def cuda_source_digest() -> str:
    digest = sha256()
    for source in _SOURCE_BINDING_FILES:
        digest.update(source.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_cuda_extension(
    *,
    build_root: str | Path | None = None,
    verbose: bool = False,
) -> Any:
    if not torch.cuda.is_available():
        raise OptionDExecutionError("Option D CUDA backend requires an available CUDA device")
    source_digest = cuda_source_digest()
    module = _MODULES.get(source_digest)
    if module is not None:
        return module
    root = Path(
        build_root
        or os.environ.get(
            "EIEL_OPTION_D_CUDA_BUILD_ROOT",
            public_runtime_path('build_cache/option_d'),
        )
    )
    build_directory = root / source_digest[:16]
    build_directory.mkdir(parents=True, exist_ok=True)
    prior_max_jobs = os.environ.get("MAX_JOBS")
    os.environ.setdefault("MAX_JOBS", "2")
    try:
        module = load(
            name=f"eiel_option_d_cuda_{source_digest[:16]}",
            sources=[str(source) for source in _COMPILE_SOURCES],
            build_directory=str(build_directory),
            extra_cflags=["-O2", "-ffp-contract=off"],
            extra_cuda_cflags=[
                "-O3",
                "--fmad=false",
                "--prec-div=true",
                "--prec-sqrt=true",
                "-allow-unsupported-compiler",
                "-lineinfo",
            ],
            with_cuda=True,
            verbose=verbose,
        )
    finally:
        if prior_max_jobs is None:
            os.environ.pop("MAX_JOBS", None)
        else:
            os.environ["MAX_JOBS"] = prior_max_jobs
    _MODULES[source_digest] = module
    return module


def _state_tensors(
    bundle: RNGStateBundle,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    words = torch.tensor(
        [list(state.words) for state in bundle.worlds],
        dtype=torch.uint32,
        device=device,
    )
    indices = torch.tensor(
        [state.index for state in bundle.worlds],
        dtype=torch.int64,
        device=device,
    )
    has_gauss = torch.tensor(
        [state.gauss_next_bits is not None for state in bundle.worlds],
        dtype=torch.bool,
        device=device,
    )
    gauss_next = torch.tensor(
        [
            0.0
            if state.gauss_next_bits is None
            else float64_from_bits(state.gauss_next_bits)
            for state in bundle.worlds
        ],
        dtype=torch.float64,
        device=device,
    )
    counters = torch.tensor(
        [
            [getattr(state.counters, field) for field in _COUNTER_FIELDS]
            for state in bundle.worlds
        ],
        dtype=torch.int64,
        device=device,
    )
    return words, indices, has_gauss, gauss_next, counters


def _schedule_host_tensors(schedule: DrawSchedule) -> tuple[torch.Tensor, ...]:
    """Materialize the immutable schedule ABI on CPU.

    Keeping host preparation separate from device transfer lets the standalone
    D6 characterization report those costs independently without changing the
    execution contract.
    """

    worlds = len(schedule.worlds)
    capacity = max((len(world) for world in schedule.worlds), default=0)
    opcodes = torch.zeros((worlds, capacity), dtype=torch.int32)
    arguments = torch.zeros((worlds, capacity), dtype=torch.int64)
    mus = torch.zeros((worlds, capacity), dtype=torch.float64)
    sigmas = torch.ones((worlds, capacity), dtype=torch.float64)
    predicate_sources = torch.full((worlds, capacity), -1, dtype=torch.int64)
    predicate_kinds = torch.zeros((worlds, capacity), dtype=torch.int32)
    predicate_thresholds = torch.zeros((worlds, capacity), dtype=torch.float64)
    counts = torch.tensor(
        [len(world) for world in schedule.worlds],
        dtype=torch.int64,
    )
    for world_index, operations in enumerate(schedule.worlds):
        for operation_index, operation in enumerate(operations):
            opcodes[world_index, operation_index] = int(operation.operator)
            arguments[world_index, operation_index] = operation.argument
            mus[world_index, operation_index] = float64_from_bits(operation.mu_bits)
            sigmas[world_index, operation_index] = float64_from_bits(operation.sigma_bits)
            if operation.predicate is not None:
                predicate_sources[world_index, operation_index] = (
                    operation.predicate.source_operation
                )
                predicate_kinds[world_index, operation_index] = int(
                    operation.predicate.kind
                )
                predicate_thresholds[world_index, operation_index] = (
                    float64_from_bits(operation.predicate.threshold_bits)
                )
    return (
        opcodes,
        arguments,
        mus,
        sigmas,
        predicate_sources,
        predicate_kinds,
        predicate_thresholds,
        counts,
    )


def _schedule_tensors(
    schedule: DrawSchedule,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        tensor.to(device=device, non_blocking=False)
        for tensor in _schedule_host_tensors(schedule)
    )


def _check_counter_headroom(bundle: RNGStateBundle, schedule: DrawSchedule) -> None:
    maximum = (1 << 63) - 1
    for world_index, (state, operations) in enumerate(
        zip(bundle.worlds, schedule.worlds, strict=True)
    ):
        # This conservative bound treats every operation as a two-word,
        # rejection-capable cache miss.  It prevents signed counter overflow in
        # the device ABI without claiming a tighter branch-dependent bound.
        headroom = 8 * len(operations) + 2048
        for field in _COUNTER_FIELDS:
            if getattr(state.counters, field) > maximum - headroom:
                raise OptionDContractError(
                    f"world {world_index} counter {field} lacks CUDA headroom"
                )


def _gaussian_strategy_id(
    schedule: DrawSchedule,
    gaussian_strategy: str,
) -> int:
    if gaussian_strategy not in _GAUSSIAN_STRATEGIES:
        raise OptionDContractError(
            f"unknown Gaussian strategy {gaussian_strategy!r}"
        )
    if gaussian_strategy == "disabled" and any(
        operation.operator.name == "GAUSS"
        for world in schedule.worlds
        for operation in world
    ):
        raise OptionDContractError("gauss operations require an explicit Gaussian strategy")
    return _GAUSSIAN_STRATEGIES[gaussian_strategy]


def _decode_cuda_outputs(
    initial: RNGStateBundle,
    schedule: DrawSchedule,
    outputs: Any,
    *,
    gaussian_strategy: str,
    resident: bool,
) -> GenerationResult:
    """Validate one extension envelope before any owner publishes it."""

    if not isinstance(outputs, (tuple, list)) or len(outputs) != 9:
        raise OptionDExecutionError("Option D CUDA extension returned an invalid envelope")
    (
        successor_words,
        successor_indices,
        successor_has_gauss,
        successor_gauss_next,
        successor_counters,
        output_bits,
        executed,
        statuses,
        error_operations,
    ) = outputs
    status_values = statuses.cpu().tolist()
    if any(status != 0 for status in status_values):
        error_values = error_operations.cpu().tolist()
        failures = [
            {
                "world": world,
                "operation": error_values[world],
                "status": _STATUS_NAMES.get(status, f"unknown_{status}"),
            }
            for world, status in enumerate(status_values)
            if status != 0
        ]
        raise OptionDExecutionError(
            f"CUDA schedule failed transactionally; no successor published: {failures}"
        )

    words_cpu = successor_words.cpu().tolist()
    indices_cpu = successor_indices.cpu().tolist()
    has_gauss_cpu = successor_has_gauss.cpu().tolist()
    gauss_cpu = successor_gauss_next.cpu().tolist()
    counters_cpu = successor_counters.cpu().tolist()
    output_cpu = output_bits.cpu().tolist()
    executed_cpu = executed.cpu().tolist()
    successor_worlds: list[CpythonRNGState] = []
    material_worlds: list[tuple[GeneratedValue, ...]] = []
    world_receipts: list[WorldExecutionReceipt] = []
    for world_index, operations in enumerate(schedule.worlds):
        counter = RNGCounters(
            **dict(zip(_COUNTER_FIELDS, counters_cpu[world_index], strict=True))
        )
        successor = CpythonRNGState(
            words=tuple(words_cpu[world_index]),
            index=indices_cpu[world_index],
            gauss_next_bits=(
                float64_bits(gauss_cpu[world_index])
                if has_gauss_cpu[world_index]
                else None
            ),
            counters=counter,
        )
        successor.validate()
        values = tuple(
            GeneratedValue(
                operation.operator,
                output_cpu[world_index][offset],
                executed_cpu[world_index][offset],
            )
            for offset, operation in enumerate(operations)
        )
        initial_state = initial.worlds[world_index]
        successor_worlds.append(successor)
        material_worlds.append(values)
        world_receipts.append(
            WorldExecutionReceipt(
                world=world_index,
                operations=len(operations),
                initial_state_sha256=initial_state.digest(),
                successor_state_sha256=successor.digest(),
                counters_before=initial_state.counters,
                counters_after=successor.counters,
                counters_delta=successor.counters.delta(initial_state.counters),
            )
        )
    successor_bundle = replace(initial, worlds=tuple(successor_worlds))
    material = GeneratedMaterial(tuple(material_worlds))
    mode = "raw" if gaussian_strategy == "disabled" else f"gauss_{gaussian_strategy}"
    residence = "resident_" if resident else ""
    receipt = ExecutionReceipt(
        backend_id=f"option_d_cuda_{residence}{mode}_v1",
        source_binding=initial.source_binding,
        schedule_sha256=schedule.digest(),
        initial_bundle_sha256=initial.digest(),
        successor_bundle_sha256=successor_bundle.digest(),
        material_sha256=material.digest(),
        worlds=tuple(world_receipts),
    )
    result = GenerationResult(material, successor_bundle, receipt)
    result.validate(initial=initial, schedule=schedule)
    return result


def execute_cuda(
    bundle: RNGStateBundle,
    schedule: DrawSchedule,
    *,
    device: str | torch.device = "cuda",
    gaussian_strategy: str = "disabled",
    build_root: str | Path | None = None,
    verbose_build: bool = False,
) -> GenerationResult:
    """Execute on isolated successor tensors and publish only after status GO."""

    bundle.validate()
    schedule.validate(expected_worlds=len(bundle.worlds))
    strategy_id = _gaussian_strategy_id(schedule, gaussian_strategy)
    _check_counter_headroom(bundle, schedule)
    target = torch.device(device)
    if target.type != "cuda":
        raise OptionDContractError("Option D CUDA backend requires a CUDA device")
    module = load_cuda_extension(build_root=build_root, verbose=verbose_build)
    state_tensors = _state_tensors(bundle, device=target)
    schedule_tensors = _schedule_tensors(schedule, device=target)
    outputs = module.execute(
        *state_tensors,
        *schedule_tensors,
        strategy_id,
    )
    return _decode_cuda_outputs(
        bundle,
        schedule,
        outputs,
        gaussian_strategy=gaussian_strategy,
        resident=False,
    )


def probe_cuda_math(
    inputs: tuple[tuple[float, float], ...],
    *,
    gaussian_strategy: str,
    device: str | torch.device = "cuda",
    build_root: str | Path | None = None,
) -> tuple[tuple[float, float, float], ...]:
    """Run the bounded D3 sin/cos/log diagnostic over explicit inputs."""

    if gaussian_strategy not in {"libdevice", "software"}:
        raise OptionDContractError("math probe requires libdevice or software strategy")
    if not isinstance(inputs, tuple) or any(
        not isinstance(item, tuple) or len(item) != 2 for item in inputs
    ):
        raise OptionDContractError("math probe inputs must be immutable angle/log pairs")
    target = torch.device(device)
    if target.type != "cuda":
        raise OptionDContractError("Option D math probe requires a CUDA device")
    module = load_cuda_extension(build_root=build_root)
    tensor = torch.tensor(inputs, dtype=torch.float64, device=target)
    outputs = module.math_probe(tensor, _GAUSSIAN_STRATEGIES[gaussian_strategy])
    return tuple(tuple(row) for row in outputs.cpu().tolist())


class CudaResidentEngine:
    """Single-owner, failure-atomic resident continuation engine.

    Authoritative tensors stay on one CUDA device between windows.  A host
    mirror is updated only after device status, successor decoding, and the
    hash-bound receipt all validate successfully.
    """

    def __init__(
        self,
        bundle: RNGStateBundle,
        *,
        device: str | torch.device = "cuda",
        build_root: str | Path | None = None,
        verbose_build: bool = False,
    ) -> None:
        bundle.validate()
        target = torch.device(device)
        if target.type != "cuda":
            raise OptionDContractError("Option D resident engine requires a CUDA device")
        self._owner_thread = threading.get_ident()
        self._device = target
        self._module = load_cuda_extension(
            build_root=build_root,
            verbose=verbose_build,
        )
        self._state_tensors = _state_tensors(bundle, device=target)
        self._bundle = bundle
        self._window_count = 0

    def _require_owner(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise OptionDExecutionError(
                "resident Option D state may only be used by its owner thread"
            )

    @property
    def bundle(self) -> RNGStateBundle:
        self._require_owner()
        return self._bundle

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def resident_device_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self._state_tensors)

    @property
    def window_count(self) -> int:
        return self._window_count

    def execute(
        self,
        schedule: DrawSchedule,
        *,
        gaussian_strategy: str = "disabled",
    ) -> GenerationResult:
        """Execute and commit one bounded window, or leave resident state unchanged."""

        self._require_owner()
        schedule.validate(expected_worlds=len(self._bundle.worlds))
        strategy_id = _gaussian_strategy_id(schedule, gaussian_strategy)
        _check_counter_headroom(self._bundle, schedule)
        schedule_tensors = _schedule_tensors(schedule, device=self._device)
        outputs = self._module.execute(
            *self._state_tensors,
            *schedule_tensors,
            strategy_id,
        )
        result = _decode_cuda_outputs(
            self._bundle,
            schedule,
            outputs,
            gaussian_strategy=gaussian_strategy,
            resident=True,
        )
        # Publication is deliberately last: extension failures, copy failures,
        # malformed envelopes, and receipt mismatches cannot advance the owner.
        self._state_tensors = tuple(outputs[:5])
        self._bundle = result.successor
        self._window_count += 1
        return result

    def capture_checkpoint(
        self,
        *,
        artifact_binding: str,
        schedule_sha256: str | None = None,
    ) -> CheckpointSnapshot:
        """Capture immutable canonical bytes from the validated owner mirror."""

        self._require_owner()
        return _capture_checkpoint(
            self._bundle,
            artifact_binding=artifact_binding,
            schedule_sha256=schedule_sha256,
        )


__all__ = [
    "CudaResidentEngine",
    "cuda_source_digest",
    "execute_cuda",
    "load_cuda_extension",
    "probe_cuda_math",
]
