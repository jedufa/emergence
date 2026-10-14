"""Resident C1+C2+C3+C4 composition owner (repair seam).

This module is deliberately separate from the historical R11 audit and from
the R9/R10 component adapters.  It owns the repaired transaction envelope:
the V4 bundle and C5 replay context are canonical, C3 mechanics operate on a
device-local derived phase, and C4 settlement is guarded by one batch status.

The native graph/commit implementation is admitted only after the envelope
has been fully bound.  Calling the execution entry point before that binding
fails closed; no Python C3 -> R9 fallback is provided.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

import torch

from .c4_mating import TensorWorldC4MateBirthConfigV1
from .c4_resident_authority_r11 import (
    c4_resident_authority_map_receipt_r11_v1,
    validate_c4_resident_authority_map_r11_v1,
)
from .c4_identity_r11 import (
    TensorWorldC4IdentityStagingR11V1,
    _identity_staging_tensors_r11_v1,
    bind_c4_identity_authority_r11_v1,
    c4_identity_authority_r11_source_receipt_v1,
    make_c4_identity_staging_r11_v1,
)
from .c4_event_ring_r11 import (
    _event_ring_tensors,
    c4_event_ring_r11_source_receipt_v1,
    make_c4_event_ring_r11_v1,
)
from .c4_resident_birth_r2 import _half_tensors, _one_tensors
from .c4_resident_rng_admission_r11 import (
    C4ResidentRngNativeStagingR11V1,
    make_c4_resident_rng_native_staging_r11_v1,
)
from .c4_resident_population_r9 import (
    _death_authoritative,
    _inherit_tensors,
    _minus_one_tensors,
    _primary_tensors,
    _replay_tensors,
    _scalar_rng_tensors,
    _staging_tensors,
    _zero_tensors,
    _r9_turnover_tensors,
    TensorWorldC4ResidentPopulationR9StagingV1,
    make_c4_resident_population_r9_staging_v1,
)
from .c4_state import TensorWorldStateBundleV4
from .c5_determinism import TensorWorldC5ReplayContextV1
from .c5_gaussian_window_r11 import (
    C5GaussianWindowAdmissionR11V1,
    admit_c5_gaussian_window_r11_v1,
    gaussian_window_tensors_r11_v1,
)
from .c2_backend import TensorWorldC2ConfigV1, _config_tensors_v1, _native_state_v1
from .c2_native_loader import load_tensor_world_c2_extension
from .c3_backend import (
    _active_mechanics_cuda_v1,
    _chemistry_update_cuda_v1,
    _sync_compiled_policy_v1,
)
from .c3_mechanics import TensorWorldC3MechanicsConfigV1
from .contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[2]
R11_AUTHORITY = ROOT / "engine/tensor_world/c4_resident_authority_r11.py"
R11_AUDIT = ROOT / "engine/tensor_world/c4_resident_composition_r11.py"
C3_BACKEND = ROOT / "engine/tensor_world/c3_backend.py"
R9_ADAPTER = ROOT / "engine/tensor_world/c4_resident_population_r9.py"
R11_NATIVE_BINDING = (
    ROOT / "engine/tensor_world/native/tensor_world_c4_resident_graph_r11_binding.cpp"
)
R11_NATIVE_KERNEL = (
    ROOT / "engine/tensor_world/native/tensor_world_c4_resident_graph_r11.cu"
)
R9_NATIVE = ROOT / "engine/tensor_world/native/tensor_world_c4_resident_population_r9.cu"
R5_NATIVE = ROOT / "engine/tensor_world/native/tensor_world_c4_resident_population_r5.cu"
R4_NATIVE = ROOT / "engine/tensor_world/native/tensor_world_c4_resident_mutation_r4.cu"
R11_RNGV2_MUTATION = ROOT / "engine/tensor_world/native/tensor_world_c4_r11_rngv2_mutation.cuh"
TWRNGV2_DEVICE = ROOT / "engine/tw_rngv2/native/tw_rngv2.cuh"
R2_NATIVE = ROOT / "engine/tensor_world/native/tensor_world_c4_resident_birth_r2.cu"
R6_NATIVE = ROOT / "engine/tensor_world/native/tensor_world_c4_resident_preflight_r6.cu"
R13_TRANSPORT = ROOT / "engine/tensor_world/native/tensor_world_c4_resident_transport_r13.cu"

C4_RESIDENT_COMPOSED_R11_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_composed_r11_v1"
)
C4_RESIDENT_COMPOSED_R11_STAGING_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_composed_r11_staging_v1"
)
C4_RESIDENT_COMPOSED_R11_DECISION_V1: Final[str] = "NO_GATE_DECISION"
C4_RESIDENT_COMPOSED_R11_BATCH_V1: Final[int] = 48
C4_RESIDENT_COMPOSED_R11_CAPACITY_V1: Final[int] = 32
C4_RESIDENT_COMPOSED_R11_PRODUCTION_CAPACITY_V1: Final[int] = 512
C4_RESIDENT_COMPOSED_R11_ADMITTED_CAPACITIES_V1: Final[tuple[int, ...]] = (
    C4_RESIDENT_COMPOSED_R11_CAPACITY_V1,
    C4_RESIDENT_COMPOSED_R11_PRODUCTION_CAPACITY_V1,
)
C4_RESIDENT_COMPOSED_R11_GRAPH_WINDOW_TICKS_V1: Final[int] = 5
C4_RESIDENT_COMPOSED_R11_PEER_FAILED_STATUS_V1: Final[int] = 53

# These two policy payloads dominate the production C512 arena.  C3 only
# reads them; C4 writes newborn rows solely after its batch-global semantic
# gate.  Sharing those exact tensors between the canonical owner and its
# transaction shadow therefore preserves semantic rollback while avoiding a
# second roughly two-GiB policy allocation.  Any new shared path must first
# prove the same guarded-write property and extend the static contract tests.
_R11_GUARDED_SHARED_TENSOR_PATHS_V1: Final[frozenset[str]] = frozenset(
    {
        "base.base.core.policy.packed_weights",
        "base.base.policy.packed_weights",
    }
)


def _clone_resident_value_v1(value: object, *, path: str = "") -> object:
    """Allocate a same-structure device shadow once at owner binding."""

    if isinstance(value, torch.Tensor):
        if path in _R11_GUARDED_SHARED_TENSOR_PATHS_V1:
            return value
        return torch.empty_like(value)
    if isinstance(value, tuple):
        return tuple(
            _clone_resident_value_v1(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, list):
        return [
            _clone_resident_value_v1(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(
            **{
                field.name: _clone_resident_value_v1(
                    getattr(value, field.name),
                    path=f"{path}.{field.name}" if path else field.name,
                )
                for field in fields(value)
            }
        )
    return value


def _resident_tensor_pairs_v1(
    source: object, destination: object
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Return destination/source leaves in stable dataclass order."""

    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def visit(src: object, dst: object) -> None:
        if isinstance(src, torch.Tensor):
            if not isinstance(dst, torch.Tensor) or src.shape != dst.shape or src.dtype != dst.dtype:
                raise TensorWorldContractError("R11 shadow tensor structure drifted")
            if src is dst:
                return
            pairs.append((dst, src))
            return
        if isinstance(src, tuple):
            if not isinstance(dst, tuple) or len(src) != len(dst):
                raise TensorWorldContractError("R11 shadow tuple structure drifted")
            for source_item, destination_item in zip(src, dst, strict=True):
                visit(source_item, destination_item)
            return
        if isinstance(src, list):
            if not isinstance(dst, list) or len(src) != len(dst):
                raise TensorWorldContractError("R11 shadow list structure drifted")
            for source_item, destination_item in zip(src, dst, strict=True):
                visit(source_item, destination_item)
            return
        if is_dataclass(src) and not isinstance(src, type):
            if not is_dataclass(dst) or type(src) is not type(dst):
                raise TensorWorldContractError("R11 shadow dataclass structure drifted")
            for field in fields(src):
                visit(getattr(src, field.name), getattr(dst, field.name))

    visit(source, destination)
    return tuple(pairs)


@dataclass(slots=True)
class TensorWorldC4ResidentComposedR11StagingV1:
    """Caller-owned device transaction envelope.

    ``r9`` is the existing fixed-capacity population staging.  The remaining
    fields are intentionally small device scalars/vectors: they are the only
    cross-phase status and time surfaces, so normal execution never needs a
    D2H status read or a host snapshot.
    """

    r9: TensorWorldC4ResidentPopulationR9StagingV1
    identity: TensorWorldC4IdentityStagingR11V1
    rng: C4ResidentRngNativeStagingR11V1
    c3_status: torch.Tensor
    c4_status: torch.Tensor
    commit_status: torch.Tensor
    generation: torch.Tensor
    tick: torch.Tensor
    shadow_event_sequence: torch.Tensor
    requested_ticks: torch.Tensor
    completed_ticks: torch.Tensor
    current_tick: torch.Tensor
    global_status: torch.Tensor
    first_error_stage: torch.Tensor
    first_error_world: torch.Tensor
    first_error_code: torch.Tensor
    first_error_tick: torch.Tensor
    stop_requested: torch.Tensor
    receipt_write_index: torch.Tensor
    schema_version: str = C4_RESIDENT_COMPOSED_R11_STAGING_SCHEMA_VERSION_V1

    @property
    def device(self) -> torch.device:
        return self.c3_status.device

    @property
    def batch(self) -> int:
        return int(self.c3_status.shape[0])

    def validate(self, *, batch: int, capacity: int, grid_size: int, streams: int) -> None:
        if self.schema_version != C4_RESIDENT_COMPOSED_R11_STAGING_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("R11 composed staging schema is unsupported")
        if (
            batch != C4_RESIDENT_COMPOSED_R11_BATCH_V1
            or capacity not in C4_RESIDENT_COMPOSED_R11_ADMITTED_CAPACITIES_V1
        ):
            raise TensorWorldContractError(
                "R11 composed staging requires B48 and admitted C32/C512 capacity"
            )
        self.r9.validate(batch=batch, capacity=capacity, grid_size=grid_size, streams=streams)
        self.identity.validate(batch=batch, capacity=capacity)
        self.rng.validate(batch=batch, streams=streams, device=self.device)
        expected = (
            ("c3_status", self.c3_status, torch.int32),
            ("c4_status", self.c4_status, torch.int32),
            ("commit_status", self.commit_status, torch.int32),
            ("generation", self.generation, torch.int64),
            ("tick", self.tick, torch.int64),
            ("shadow_event_sequence", self.shadow_event_sequence, torch.int64),
            ("requested_ticks", self.requested_ticks, torch.int64),
            ("completed_ticks", self.completed_ticks, torch.int64),
            ("current_tick", self.current_tick, torch.int64),
            ("global_status", self.global_status, torch.int32),
            ("first_error_stage", self.first_error_stage, torch.int32),
            ("first_error_world", self.first_error_world, torch.int32),
            ("first_error_code", self.first_error_code, torch.int32),
            ("first_error_tick", self.first_error_tick, torch.int64),
            ("stop_requested", self.stop_requested, torch.int32),
            ("receipt_write_index", self.receipt_write_index, torch.int64),
        )
        devices = {self.device, self.r9.device, self.identity.device}
        for name, tensor, dtype in expected:
            if (
                tuple(tensor.shape) != (batch,)
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
                or tensor.device != self.device
            ):
                raise TensorWorldContractError(
                    f"R11 composed staging {name} ABI mismatch"
                )
            devices.add(tensor.device)
        if self.device.type != "cuda" or len(devices) != 1:
            raise TensorWorldContractError("R11 composed staging requires one CUDA device")


def make_c4_resident_composed_r11_staging_v1(
    *, batch: int, capacity: int, grid_size: int, streams: int,
    device: torch.device | str,
) -> TensorWorldC4ResidentComposedR11StagingV1:
    """Allocate the resident R11 envelope once, before steady-state ticks."""

    target = torch.device(device)
    if target.type != "cuda":
        raise TensorWorldContractError("R11 composed staging requires CUDA")
    if (
        batch != C4_RESIDENT_COMPOSED_R11_BATCH_V1
        or capacity not in C4_RESIDENT_COMPOSED_R11_ADMITTED_CAPACITIES_V1
    ):
        raise TensorWorldContractError(
            "R11 composed staging requires B48 and admitted C32/C512 capacity"
        )
    staging = TensorWorldC4ResidentComposedR11StagingV1(
        r9=make_c4_resident_population_r9_staging_v1(
            batch=batch, capacity=capacity, grid_size=grid_size,
            streams=streams, device=target,
        ),
        identity=make_c4_identity_staging_r11_v1(
            batch=batch, capacity=capacity, device=target
        ),
        rng=make_c4_resident_rng_native_staging_r11_v1(
            batch=batch, streams=streams, device=target
        ),
        c3_status=torch.zeros(batch, dtype=torch.int32, device=target),
        c4_status=torch.zeros(batch, dtype=torch.int32, device=target),
        commit_status=torch.zeros(batch, dtype=torch.int32, device=target),
        generation=torch.zeros(batch, dtype=torch.int64, device=target),
        tick=torch.zeros(batch, dtype=torch.int64, device=target),
        shadow_event_sequence=torch.zeros(batch, dtype=torch.int64, device=target),
        requested_ticks=torch.zeros(batch, dtype=torch.int64, device=target),
        completed_ticks=torch.zeros(batch, dtype=torch.int64, device=target),
        current_tick=torch.zeros(batch, dtype=torch.int64, device=target),
        global_status=torch.zeros(batch, dtype=torch.int32, device=target),
        first_error_stage=torch.full((batch,), -1, dtype=torch.int32, device=target),
        first_error_world=torch.full((batch,), -1, dtype=torch.int32, device=target),
        first_error_code=torch.zeros(batch, dtype=torch.int32, device=target),
        first_error_tick=torch.full((batch,), -1, dtype=torch.int64, device=target),
        stop_requested=torch.zeros(batch, dtype=torch.int32, device=target),
        receipt_write_index=torch.zeros(batch, dtype=torch.int64, device=target),
    )
    staging.validate(
        batch=batch, capacity=capacity, grid_size=grid_size, streams=streams
    )
    return staging


@dataclass(slots=True)
class TensorWorldC4ResidentComposedR11OwnerV1:
    """Canonical resident owner handle for the repaired composition route."""

    bundle: TensorWorldStateBundleV4
    replay: TensorWorldC5ReplayContextV1
    staging: TensorWorldC4ResidentComposedR11StagingV1
    shadow_bundle: TensorWorldStateBundleV4 | None = None
    shadow_replay: TensorWorldC5ReplayContextV1 | None = None
    shadow_staging: TensorWorldC4ResidentComposedR11StagingV1 | None = None
    copy_graph: Any | None = None
    commit_graph: Any | None = None
    c3_graph: Any | None = None
    transaction_graph: Any | None = None
    window_graph: Any | None = None
    c2_extension: Any | None = None
    transport_extension: Any | None = None
    transport_plan: Any | None = None
    r9_extension: Any | None = None
    r11_extension: Any | None = None
    c2_config: TensorWorldC2ConfigV1 | None = None
    c2_config_ints: torch.Tensor | None = None
    c2_config_floats: torch.Tensor | None = None
    c3_mechanics_config: TensorWorldC3MechanicsConfigV1 | None = None
    window_max_births_per_tick: int | None = None
    window_source_commit: str = "UNSPECIFIED_TEST_SOURCE"
    window_control_initialized: bool = False
    last_gaussian_window_admission: C5GaussianWindowAdmissionR11V1 | None = None
    native_error: str | None = None
    native_bound: bool = False
    rng_semantics_version: int = 1
    last_v2_window_receipt: dict[str, object] | None = None
    window_generation_base: torch.Tensor | None = None
    v2_hot_loop_seeded: bool = False
    v2_host_prepare_calls: int = 0
    v2_host_status_d2h: int = 0
    v2_host_reclaim_copies: int = 0
    schema_version: str = C4_RESIDENT_COMPOSED_R11_SCHEMA_VERSION_V1

    def validate(self) -> None:
        if self.schema_version != C4_RESIDENT_COMPOSED_R11_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("R11 composed owner schema is unsupported")
        if self.bundle.device.type != "cuda":
            raise TensorWorldContractError("R11 composed owner requires CUDA state")
        if (
            self.bundle.batch != C4_RESIDENT_COMPOSED_R11_BATCH_V1
            or self.bundle.capacity
            not in C4_RESIDENT_COMPOSED_R11_ADMITTED_CAPACITIES_V1
        ):
            raise TensorWorldContractError(
                "R11 composed owner requires B48 and admitted C32/C512 capacity"
            )
        if self.bundle.events is None or self.bundle.scalar_rng is None:
            raise TensorWorldContractError("R11 composed owner requires event and scalar RNG state")
        self.staging.validate(
            batch=self.bundle.batch, capacity=self.bundle.capacity,
            grid_size=self.bundle.grid_size,
            streams=int(self.replay.rng_state_words.shape[1]),
        )
        if self.staging.device != self.bundle.device or self.replay.device != self.bundle.device:
            raise TensorWorldContractError("R11 owner state must share one CUDA device")
        if self.native_bound:
            if (
                self.shadow_bundle is None
                or self.shadow_replay is None
                or self.shadow_staging is None
                or self.copy_graph is None
                or self.commit_graph is None
                or self.c3_graph is None
                or self.transaction_graph is None
                or self.c2_extension is None
                or self.transport_extension is None
                or self.transport_plan is None
                or self.r11_extension is None
                or self.c2_config_ints is None
                or self.c2_config_floats is None
            ):
                raise TensorWorldContractError("R11 native owner binding is incomplete")
            for name, tensor, dtype in (
                ("C2 integer config", self.c2_config_ints, torch.int64),
                ("C2 floating config", self.c2_config_floats, torch.float64),
            ):
                if (
                    tensor.device != self.bundle.device
                    or tensor.dtype != dtype
                    or tensor.ndim != 1
                    or tensor.numel() == 0
                    or not tensor.is_contiguous()
                ):
                    raise TensorWorldContractError(
                        f"R11 owner-retained {name} tensor is invalid"
                    )
            validate_transport_plan = getattr(self.transport_plan, "validate", None)
            if validate_transport_plan is None:
                raise TensorWorldContractError("R11 transport plan is invalid")
            validate_transport_plan()
            for path in _R11_GUARDED_SHARED_TENSOR_PATHS_V1:
                canonical: object = self.bundle
                shadow: object = self.shadow_bundle
                for component in path.split("."):
                    canonical = getattr(canonical, component)
                    shadow = getattr(shadow, component)
                if canonical is not shadow:
                    raise TensorWorldContractError(
                        f"R11 guarded shared tensor is not aliased: {path}"
                    )

    def assert_steady_state_ready(self) -> None:
        """Check only the immutable binding token used by normal ticks.

        Normal tick execution is deliberately not a validation boundary.  The
        full structural/semantic checks above run while binding/capturing the
        owner; repeating them from a tick would make Python part of the
        simulation loop.  This method therefore inspects only immutable
        object identity and graph presence, never tensor values or per-world
        state.
        """

        if not self.native_bound:
            raise TensorWorldContractError(
                "R11 composed native owner is not bound; host C3 -> R9 fallback is forbidden"
            )
        if (
            self.shadow_bundle is None
            or self.shadow_replay is None
            or self.shadow_staging is None
            or self.copy_graph is None
            or self.commit_graph is None
            or self.c3_graph is None
            or self.transaction_graph is None
            or self.c2_extension is None
            or self.transport_extension is None
            or self.transport_plan is None
            or self.r11_extension is None
            or self.c2_config_ints is None
            or self.c2_config_floats is None
        ):
            raise TensorWorldContractError("R11 steady-state graph owner is incomplete")


def _run_c3_phase_v1(
    state: TensorWorldStateBundleV4,
    native: Any,
    native_state: list[torch.Tensor],
    config_ints: torch.Tensor,
    config_floats: torch.Tensor,
    mechanics: TensorWorldC3MechanicsConfigV1,
) -> None:
    """Run the C3 eligibility-producing phase without host status reads."""

    c3 = state.base
    c3.transactions.pre_energy.copy_(c3.base.core.agents.energy)
    c3.transactions.pre_health.copy_(c3.base.core.agents.health)
    c3.transactions.pre_injury.copy_(c3.agent.injury)
    c3.transactions.pre_survival_steps.copy_(
        c3.base.core.agents.survival_steps.to(torch.int32)
    )
    _sync_compiled_policy_v1(c3)
    native.wdl_step(native_state, config_ints, config_floats)
    native.eed_step(native_state, config_ints, config_floats)
    native.patch_step(native_state, config_ints, config_floats)
    native.jungle_step(native_state, config_ints, config_floats)
    native.observe(native_state, config_ints, config_floats)
    native.policy_logits(native_state, config_ints, config_floats)
    native.select_actions(native_state, config_ints, config_floats)
    native.settle(native_state, config_ints, config_floats)
    _active_mechanics_cuda_v1(c3, mechanics)
    _chemistry_update_cuda_v1(c3, mechanics)


_R11_NATIVE_MODULE: Any | None = None
_R11_NATIVE_MODULE_V2: Any | None = None


def _load_r11_graph_extension_v1(
    *,
    build_directory: str | Path | None = None,
    verbose: bool = False,
    rng_semantics_version: int = 1,
) -> Any:
    """Load the R11 device-time ABI used during graph capture.

    This is deliberately a distinct extension from the historical R9 module:
    R9 keeps its host scalar generation/tick API, while R11 receives those
    values through resident int64 tensors so a captured graph can replay them
    without a Python semantic callback.
    """

    global _R11_NATIVE_MODULE, _R11_NATIVE_MODULE_V2
    if rng_semantics_version not in {1, 2}:
        raise TensorWorldContractError("R11 rng_semantics_version must be 1 or 2")
    cached = _R11_NATIVE_MODULE_V2 if rng_semantics_version == 2 else _R11_NATIVE_MODULE
    if cached is not None:
        return cached
    if not torch.cuda.is_available():
        raise TensorWorldContractError("R11 graph binding requires CUDA")
    from torch.utils.cpp_extension import load

    hashed = [
        R11_NATIVE_BINDING,
        R11_NATIVE_KERNEL,
        R9_ADAPTER,
        R9_NATIVE,
        R5_NATIVE,
        R4_NATIVE,
        R2_NATIVE,
        R6_NATIVE,
    ]
    if rng_semantics_version == 2:
        hashed.extend((R11_RNGV2_MUTATION, TWRNGV2_DEVICE))
    digest = sha256(b"\0".join(path.read_bytes() for path in hashed)).hexdigest()[:16]
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    includes = [
        str(ROOT / "engine" / "perf"),
        str(ROOT / "engine" / "tensor_world" / "native"),
    ]
    cuda_flags = ["-O2", "--fmad=false"]
    if rng_semantics_version == 2:
        includes.append(str(ROOT / "engine" / "tw_rngv2" / "native"))
        cuda_flags.append("-DEIEL_C4_R11_RNGV2=1")
        name = f"tensor_world_c4_resident_graph_r11_rngv2_{digest}"
    else:
        name = f"tensor_world_c4_resident_graph_r11_{digest}"
    module = load(
        name=name,
        sources=[str(R11_NATIVE_BINDING), str(R11_NATIVE_KERNEL)],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_include_paths=includes,
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=cuda_flags,
        with_cuda=True,
        verbose=bool(verbose),
    )
    if rng_semantics_version == 2:
        _R11_NATIVE_MODULE_V2 = module
    else:
        _R11_NATIVE_MODULE = module
    return module


def _r11_population_call_v1(
    extension: Any,
    shadow: TensorWorldStateBundleV4,
    replay: TensorWorldC5ReplayContextV1,
    shadow_staging: TensorWorldC4ResidentComposedR11StagingV1,
    config: TensorWorldC4MateBirthConfigV1,
) -> None:
    """Enqueue one fixed-ABI native population transaction."""

    max_births = shadow.capacity if config.max_births is None else config.max_births
    assert replay.raw64_transition_counts is not None
    assert replay.output32_return_counts is not None
    replay_tensors = _replay_tensors(replay) + [
        replay.raw64_transition_counts,
        replay.output32_return_counts,
    ]
    rng_staging = [
        shadow_staging.rng.post_raw64_transition_counts,
        shadow_staging.rng.post_output32_return_counts,
    ]
    extension.population_transaction_device_time(
        _primary_tensors(shadow), _death_authoritative(shadow),
        _inherit_tensors(shadow), _zero_tensors(shadow),
        _minus_one_tensors(shadow), _half_tensors(shadow),
        _one_tensors(shadow),
        _scalar_rng_tensors(shadow) + gaussian_window_tensors_r11_v1(replay),
        replay_tensors, _staging_tensors(shadow_staging.r9.mutation),
        _r9_turnover_tensors(shadow_staging.r9),
        _identity_staging_tensors_r11_v1(shadow_staging.identity),
        rng_staging,
        _event_ring_tensors(shadow.event_ring),
        shadow_staging.r9.preflight_status, shadow_staging.global_status,
        shadow_staging.generation,
        shadow_staging.tick, shadow_staging.requested_ticks,
        shadow_staging.completed_ticks, int(max_births),
        float(config.parent_energy_share), float(config.minimum_energy_buffer),
        int(config.cooldown_steps), float(config.start_health),
        float(config.max_energy), int(config.life_history_mode),
        int(config.maturity_start_ticks), int(config.maturity_rise_ticks),
        int(config.fertility_decline_start_ticks),
        int(config.fertility_decline_ticks), float(config.elder_fertility_floor),
        int(config.environment_stream), int(config.runner_stream),
    )


def _build_native_owner_v1(
    owner: TensorWorldC4ResidentComposedR11OwnerV1,
    config: TensorWorldC4MateBirthConfigV1,
    *,
    c2_extension_path: str | Path | None,
    native_build_directory: str | Path | None,
) -> None:
    """Build the device-local shadow/phase/commit graphs at initialization."""

    if not torch.cuda.is_available():
        raise TensorWorldContractError("R11 native binding requires CUDA")
    owner.window_max_births_per_tick = (
        owner.bundle.capacity if config.max_births is None else int(config.max_births)
    )
    shadow_bundle = _clone_resident_value_v1(owner.bundle)
    shadow_replay = _clone_resident_value_v1(owner.replay)
    if not isinstance(shadow_bundle, TensorWorldStateBundleV4) or not isinstance(
        shadow_replay, TensorWorldC5ReplayContextV1
    ):
        raise TensorWorldContractError("R11 shadow allocation produced an invalid state")
    if shadow_bundle.event_ring is None:
        raise TensorWorldContractError("R11 event ring is not attached to the resident owner")
    shadow_staging = make_c4_resident_composed_r11_staging_v1(
        batch=owner.bundle.batch,
        capacity=owner.bundle.capacity,
        grid_size=owner.bundle.grid_size,
        streams=int(owner.replay.rng_state_words.shape[1]),
        device=owner.bundle.device,
    )
    copy_pairs = _resident_tensor_pairs_v1(owner.bundle, shadow_bundle) + _resident_tensor_pairs_v1(
        owner.replay, shadow_replay
    )
    if not copy_pairs:
        raise TensorWorldContractError("R11 shadow has no resident tensor leaves")
    # R13's descriptor tensors and extension are created only at this control
    # boundary.  They are retained by the owner so graph nodes never point at
    # reclaimed descriptor storage or transient shadow objects.
    from .c4_resident_transport_r13 import (
        inventory_c4_resident_transport_r13_v1,
        load_c4_resident_transport_r13_extension,
        materialize_c4_resident_transport_plan_r13_v1,
    )

    transport_inventory = inventory_c4_resident_transport_r13_v1(
        owner.bundle, shadow_bundle, owner.replay, shadow_replay
    )
    transport_plan = materialize_c4_resident_transport_plan_r13_v1(
        transport_inventory
    )
    transport_extension = load_c4_resident_transport_r13_extension(
        build_directory=native_build_directory
    )
    copy_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(copy_graph):
        transport_extension.copy_to_shadow(
            transport_plan.copy_source_addresses,
            transport_plan.copy_destination_addresses,
            transport_plan.copy_byte_counts,
        )

    c2_config = TensorWorldC2ConfigV1(
        max_energy=float(config.max_energy), start_health=float(config.start_health)
    )
    mechanics = TensorWorldC3MechanicsConfigV1(
        max_energy=float(config.max_energy),
        start_health=float(config.start_health),
        enable_chemistry=False,
        chemistry_noise_required=False,
    )
    c2_native = load_tensor_world_c2_extension(c2_extension_path)
    c2_state = _native_state_v1(shadow_bundle.base.base)
    config_ints, config_floats = _config_tensors_v1(c2_config, shadow_bundle.base.base)
    # Custom C2 kernels capture these tensors as raw pointer arguments.  Keep
    # the exact tensor objects alive for the complete graph-owner lifetime;
    # otherwise unrelated control-boundary CUDA allocations may recycle their
    # storage and make a later graph replay dereference stale addresses.
    owner.c2_config_ints = config_ints
    owner.c2_config_floats = config_floats
    c3_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(c3_graph):
        _run_c3_phase_v1(
            shadow_bundle, c2_native, c2_state, config_ints, config_floats, mechanics
        )

    # Capture the R11 native transaction after C3 has produced a valid shadow.
    # The transaction receives generation/tick through device pointers, so the
    # graph topology is stable while time remains mutable at replay.
    shadow_staging.generation.zero_()
    shadow_staging.tick.zero_()
    copy_graph.replay()
    c3_graph.replay()
    torch.cuda.current_stream(owner.bundle.device).synchronize()
    # Capture must not exercise a newborn policy-row write because the two
    # enormous packed-policy payloads are deliberately shared with the live
    # owner.  This value change is confined to the transaction shadow at the
    # initialization boundary; every replay begins with copy_graph + C3 and
    # therefore restores the caller's authoritative threshold before native
    # preflight.  Fixed graph topology is independent of the captured birth
    # count, so a no-birth capture still admits device-varying births later.
    shadow_bundle.base.agent.repro_threshold.fill_(float("inf"))
    r11_extension = _load_r11_graph_extension_v1(
        build_directory=native_build_directory,
        rng_semantics_version=owner.rng_semantics_version,
    )
    transaction_graph = torch.cuda.CUDAGraph()
    shadow = shadow_bundle
    replay = shadow_replay
    with torch.cuda.graph(transaction_graph):
        _r11_population_call_v1(
            r11_extension, shadow, replay, shadow_staging, config
        )
    # Keep graph capture side effects out of the live owner.  The captured
    # status predicates are still dynamic on replay after the caller clears
    # this sentinel at the control boundary.
    shadow_staging.r9.preflight_status.fill_(1)
    shadow_staging.global_status.fill_(1)
    commit_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(commit_graph):
        ok = (shadow_staging.r9.preflight_status == 0).all()
        shadow_staging.c4_status.copy_(shadow_staging.r9.preflight_status)
        shadow_staging.commit_status.copy_(torch.where(
            shadow_staging.r9.preflight_status == 0,
            torch.zeros_like(shadow_staging.commit_status),
            shadow_staging.r9.preflight_status,
        ))
        shadow_staging.global_status.copy_(torch.where(
            ok, torch.zeros_like(shadow_staging.global_status),
            torch.ones_like(shadow_staging.global_status),
        ))
        shadow_staging.completed_ticks.copy_(torch.where(
            ok, shadow_staging.completed_ticks + 1,
            shadow_staging.completed_ticks,
        ))
        shadow_staging.current_tick.copy_(torch.where(
            ok, shadow_staging.tick, shadow_staging.current_tick,
        ))
        transport_extension.publish_if_all_ok(
            transport_plan.publish_source_addresses,
            transport_plan.publish_destination_addresses,
            transport_plan.publish_byte_counts,
            shadow_staging.global_status,
        )

    owner.shadow_bundle = shadow_bundle
    owner.shadow_replay = shadow_replay
    owner.shadow_staging = shadow_staging
    owner.copy_graph = copy_graph
    owner.commit_graph = commit_graph
    owner.c3_graph = c3_graph
    owner.c2_extension = c2_native
    owner.transport_extension = transport_extension
    owner.transport_plan = transport_plan
    owner.transaction_graph = transaction_graph
    owner.r11_extension = r11_extension
    owner.r9_extension = None
    owner.c2_config = c2_config
    owner.c3_mechanics_config = mechanics
    owner.native_bound = True


def capture_c4_resident_composed_r11_window_v1(
    owner: TensorWorldC4ResidentComposedR11OwnerV1,
    config: TensorWorldC4MateBirthConfigV1,
    *,
    ticks: int = C4_RESIDENT_COMPOSED_R11_GRAPH_WINDOW_TICKS_V1,
) -> None:
    """Capture a fixed multi-tick graph with device-only time/failure state.

    Capture is an initialization/control-boundary operation.  At replay the
    graph performs all five ticks, including owner-to-shadow copies, C3, the
    R11 transaction, guarded publish, time advancement, and failure latching;
    there is no Python tick loop or status decision between those ticks.
    """

    if type(ticks) is not int or ticks != C4_RESIDENT_COMPOSED_R11_GRAPH_WINDOW_TICKS_V1:
        raise TensorWorldContractError("R11 graph windows are fixed to five ticks")
    if not owner.native_bound or owner.shadow_bundle is None or owner.shadow_replay is None:
        raise TensorWorldContractError("R11 native owner must be bound before window capture")
    if owner.shadow_staging is None or owner.r11_extension is None:
        raise TensorWorldContractError("R11 window capture state is incomplete")
    if owner.c3_mechanics_config is None or owner.c2_extension is None:
        raise TensorWorldContractError("R11 window capture C3 state is incomplete")
    shadow = owner.shadow_bundle
    replay = owner.shadow_replay
    shadow_staging = owner.shadow_staging
    if owner.transport_extension is None or owner.transport_plan is None:
        raise TensorWorldContractError("R11 transport graph state is unavailable")
    c2_state = _native_state_v1(shadow.base.base)
    c2_config = owner.c2_config
    if c2_config is None:
        raise TensorWorldContractError("R11 window capture C2 config is unavailable")
    config_ints = owner.c2_config_ints
    config_floats = owner.c2_config_floats
    if config_ints is None or config_floats is None:
        raise TensorWorldContractError("R11 owner-retained C2 graph inputs are unavailable")
    shadow_staging.r9.preflight_status.fill_(1)
    shadow_staging.global_status.fill_(1)
    # Retain the raw graph at this explicit capture boundary so qualification
    # can query the exact node topology/fingerprint.  Instantiation is still
    # completed before the owner becomes replayable; no graph construction or
    # upload occurs in the normal window path.
    window_graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(window_graph):
        if owner.rng_semantics_version == 2:
            if (
                owner.window_generation_base is None
                or owner.bundle.event_ring is None
                or owner.shadow_bundle is None
                or owner.shadow_bundle.event_ring is None
                or owner.r11_extension is None
            ):
                raise TensorWorldContractError(
                    "V2 window capture requires generation base and live/shadow event rings"
                )
            owner.r11_extension.prepare_v2_window_replay(
                shadow_staging.global_status,
                shadow_staging.completed_ticks,
                shadow_staging.receipt_write_index,
                shadow_staging.generation,
                owner.window_generation_base,
                owner.bundle.event_ring.consumer_cursor,
                owner.bundle.event_ring.producer_cursor,
                owner.shadow_bundle.event_ring.consumer_cursor,
                owner.shadow_bundle.event_ring.producer_cursor,
            )
        for _step in range(ticks):
            owner.transport_extension.copy_to_shadow(
                owner.transport_plan.copy_source_addresses,
                owner.transport_plan.copy_destination_addresses,
                owner.transport_plan.copy_byte_counts,
            )
            _run_c3_phase_v1(
                shadow, owner.c2_extension, c2_state, config_ints, config_floats,
                owner.c3_mechanics_config,
            )
            _r11_population_call_v1(
                owner.r11_extension, shadow, replay, shadow_staging, config
            )
            owner.r11_extension.merge_window_control(
                shadow_staging.r9.preflight_status,
                shadow_staging.global_status,
                shadow_staging.c4_status,
                shadow_staging.commit_status,
                shadow_staging.stop_requested,
                shadow_staging.first_error_stage,
                shadow_staging.first_error_world,
                shadow_staging.first_error_code,
                shadow_staging.first_error_tick,
                shadow_staging.tick,
                shadow_staging.completed_ticks,
                shadow_staging.current_tick,
                shadow_staging.receipt_write_index,
            )
            owner.transport_extension.publish_if_all_ok(
                owner.transport_plan.publish_source_addresses,
                owner.transport_plan.publish_destination_addresses,
                owner.transport_plan.publish_byte_counts,
                shadow_staging.global_status,
            )
            owner.r11_extension.advance_window_clock(
                shadow_staging.global_status,
                shadow_staging.tick,
                shadow_staging.generation,
            )
    window_graph.instantiate()
    owner.window_graph = window_graph


def bind_c4_resident_composed_r11_v1(
    bundle: TensorWorldStateBundleV4,
    replay: TensorWorldC5ReplayContextV1,
    staging: TensorWorldC4ResidentComposedR11StagingV1,
    *,
    enable_native: bool = False,
    population_config: TensorWorldC4MateBirthConfigV1 | None = None,
    c2_extension_path: str | Path | None = None,
    native_build_directory: str | Path | None = None,
    source_commit: str = "UNSPECIFIED_TEST_SOURCE",
    rng_semantics_version: int = 1,
) -> TensorWorldC4ResidentComposedR11OwnerV1:
    """Bind caller-owned state after one initialization-time structural check."""

    validate_c4_resident_authority_map_r11_v1()
    if bundle.device.type != "cuda":
        raise TensorWorldContractError("R11 composition is CUDA-only")
    if bundle.events is None:
        raise TensorWorldContractError("R11 composition requires the typed event tape")
    if bundle.event_ring is None:
        # Ring allocation is an explicit owner-construction operation, never a
        # normal-tick fallback.  Callers may provide a checkpoint-restored ring
        # instead; the default keeps older fixture construction source-compatible.
        bundle.event_ring = make_c4_event_ring_r11_v1(
            batch=bundle.batch,
            capacity=bundle.capacity,
            event_capacity=bundle.events.capacity,
            device=bundle.device,
        )
    # Full value validation is intentionally an initialization operation.  It
    # may inspect host-visible fixtures here; the steady-state entry point
    # below never calls it and never materializes device status.
    bundle.validate()
    # Bind the monotonic-ID/slot-token views at construction.  This is a
    # semantic boundary check only; steady-state execution uses the device
    # shadow and does not perform host lookup or dense-ID indexing.
    bind_c4_identity_authority_r11_v1(bundle, semantic=True)
    replay.validate_pcg64_for_bundle(bundle)
    replay.validate_typed_continuation_for_bundle(bundle)
    if not isinstance(source_commit, str) or not source_commit:
        raise TensorWorldContractError("R11 source commit identity is incomplete")
    if rng_semantics_version not in {1, 2}:
        raise TensorWorldContractError("R11 rng_semantics_version must be 1 or 2")
    if rng_semantics_version == 2:
        if bundle.scalar_rng is None or int(bundle.scalar_rng.tape_capacity) != 0:
            raise TensorWorldContractError("V2 graph fusion requires tape_capacity=0")
    owner = TensorWorldC4ResidentComposedR11OwnerV1(
        bundle=bundle,
        replay=replay,
        staging=staging,
        native_bound=False,
        window_source_commit=source_commit,
        rng_semantics_version=int(rng_semantics_version),
        window_generation_base=(
            torch.zeros(bundle.batch, dtype=torch.int64, device=bundle.device)
            if rng_semantics_version == 2
            else None
        ),
    )
    if enable_native:
        _build_native_owner_v1(
            owner,
            population_config
            or TensorWorldC4MateBirthConfigV1(
                max_births=1, allow_mutation=True, max_energy=160.0
            ),
            c2_extension_path=c2_extension_path,
            native_build_directory=native_build_directory,
        )
    owner.validate()
    return owner


def execute_c4_resident_composed_r11_cuda_v1(
    owner: TensorWorldC4ResidentComposedR11OwnerV1,
    config: TensorWorldC4MateBirthConfigV1,
    *,
    generation: int,
    tick: int,
) -> TensorWorldC4ResidentComposedR11StagingV1:
    """Run one device-local composed transaction.

    The copy graph creates the write-ahead shadow, the C3 graph emits
    eligibility/mechanics into that shadow, R9 performs its guarded population
    transaction there, and the commit graph copies the shadow back only when
    the batch status is zero.  No status is read by the host on this path.
    """

    # This is a normal tick entry point.  Do not call ``owner.validate`` here:
    # all structural and semantic checks belong to the explicit bind/capture
    # control boundary.  The steady-state check below only inspects immutable
    # graph ownership and never reads a device tensor to make a decision.
    owner.assert_steady_state_ready()
    if type(generation) is not int or generation < 0 or type(tick) is not int or tick < 0:
        raise TensorWorldContractError("R11 generation/tick must be nonnegative ints")
    if not isinstance(config, TensorWorldC4MateBirthConfigV1) or not config.allow_mutation:
        raise TensorWorldContractError("R11 requires source-current mutating mate configuration")

    shadow_staging = owner.shadow_staging
    # These are device writes, not semantic host reads.  The captured native
    # transaction consumes the pointers on replay and publishes status into
    # the same resident staging envelope.
    shadow_staging.generation.fill_(int(generation))
    shadow_staging.tick.fill_(int(tick))
    shadow_staging.requested_ticks.fill_(1)
    shadow_staging.completed_ticks.zero_()
    shadow_staging.current_tick.fill_(int(tick))
    shadow_staging.global_status.zero_()
    shadow_staging.first_error_stage.fill_(-1)
    shadow_staging.first_error_world.fill_(-1)
    shadow_staging.first_error_code.zero_()
    shadow_staging.first_error_tick.fill_(-1)
    shadow_staging.receipt_write_index.zero_()
    shadow_staging.stop_requested.zero_()
    owner.copy_graph.replay()
    owner.c3_graph.replay()
    owner.transaction_graph.replay()
    owner.commit_graph.replay()
    return shadow_staging


def execute_c4_resident_composed_r11_window_cuda_v1(
    owner: TensorWorldC4ResidentComposedR11OwnerV1,
) -> TensorWorldC4ResidentComposedR11StagingV1:
    """Replay one prepared five-tick R11 window with one graph launch.

    This is the qualified normal-window call.  The caller must prepare device
    run control at an explicit boundary first.  All five ticks, status
    latching, event/RNG state, and publish-last commits occur inside this one
    graph replay without an escaped eager CUDA launch.
    """

    owner.window_graph.replay()
    return owner.shadow_staging


def prepare_c4_resident_composed_r11_window_control_v1(
    owner: TensorWorldC4ResidentComposedR11OwnerV1,
    *,
    generation: int,
    tick: int,
) -> TensorWorldC4ResidentComposedR11StagingV1:
    """Prepare device run control at an explicit, unmeasured boundary."""

    if owner.window_graph is None or owner.shadow_staging is None:
        raise TensorWorldContractError(
            "R11 five-tick graph window has not been captured"
        )
    if type(generation) is not int or generation < 0 or type(tick) is not int or tick < 0:
        raise TensorWorldContractError("R11 generation/tick must be nonnegative ints")
    staging = owner.shadow_staging
    if owner.window_max_births_per_tick is None:
        raise TensorWorldContractError("R11 Gaussian admission birth bound is unavailable")
    if owner.rng_semantics_version == 2 and owner.v2_hot_loop_seeded:
        owner.last_v2_window_receipt = {
            "schema_version": "tw_rngv2_graph_window_receipt_v1",
            "rng_semantics_version": 2,
            "boundary_kind": "device_hot_loop",
            "refill_performed": False,
            "mutation_birth_tape_bytes": 0,
            "mutation_birth_h2d_bytes": 0,
            "mutation_birth_host_generation_ms": 0.0,
            "graph_recapture": False,
            "requested_ticks": C4_RESIDENT_COMPOSED_R11_GRAPH_WINDOW_TICKS_V1,
            "host_prepare": False,
            "host_status_d2h": False,
            "host_event_reclaim": False,
        }
        return staging
    boundary_kind: str = "initialization"
    if owner.window_control_initialized:
        requested = staging.requested_ticks.detach().cpu()
        completed = staging.completed_ticks.detach().cpu()
        status = staging.global_status.detach().cpu()
        current = staging.current_tick.detach().cpu()
        owner.v2_host_status_d2h += 1
        if not (
            bool((requested == C4_RESIDENT_COMPOSED_R11_GRAPH_WINDOW_TICKS_V1).all())
            and torch.equal(completed, requested)
            and bool((status == 0).all())
            and bool((current == int(tick) - 1).all())
        ):
            raise TensorWorldContractError(
                "R11 next-window control boundary is not a completed successful window"
            )
        boundary_kind = "control"
    if owner.rng_semantics_version == 2:
        if owner.bundle.scalar_rng is None or int(owner.bundle.scalar_rng.tape_capacity) != 0:
            raise TensorWorldContractError("V2 graph window cannot admit mutation/birth tapes")
        owner.last_v2_window_receipt = {
            "schema_version": "tw_rngv2_graph_window_receipt_v1",
            "rng_semantics_version": 2,
            "boundary_kind": boundary_kind,
            "refill_performed": False,
            "mutation_birth_tape_bytes": 0,
            "mutation_birth_h2d_bytes": 0,
            "mutation_birth_host_generation_ms": 0.0,
            "graph_recapture": False,
            "requested_ticks": C4_RESIDENT_COMPOSED_R11_GRAPH_WINDOW_TICKS_V1,
            "host_prepare": True,
            "host_status_d2h": False,
            "host_event_reclaim": False,
        }
        owner.last_gaussian_window_admission = None
        owner.v2_host_prepare_calls += 1
    else:
        owner.last_gaussian_window_admission = admit_c5_gaussian_window_r11_v1(
            owner.bundle,
            owner.replay,
            requested_ticks=C4_RESIDENT_COMPOSED_R11_GRAPH_WINDOW_TICKS_V1,
            max_births_per_tick=owner.window_max_births_per_tick,
            boundary_kind=boundary_kind,
            source_commit=owner.window_source_commit,
            boundary_label=f"generation_{generation}_tick_{tick}",
        )
    staging.generation.fill_(int(generation))
    staging.tick.fill_(int(tick))
    staging.requested_ticks.fill_(C4_RESIDENT_COMPOSED_R11_GRAPH_WINDOW_TICKS_V1)
    staging.completed_ticks.zero_()
    staging.current_tick.fill_(int(tick))
    staging.global_status.zero_()
    staging.first_error_stage.fill_(-1)
    staging.first_error_world.fill_(-1)
    staging.first_error_code.zero_()
    staging.first_error_tick.fill_(-1)
    staging.stop_requested.zero_()
    staging.receipt_write_index.zero_()
    if owner.rng_semantics_version == 2 and owner.window_generation_base is not None:
        owner.window_generation_base.fill_(int(generation))
        owner.v2_hot_loop_seeded = True
    owner.window_control_initialized = True
    return staging


def execute_c4_resident_composed_r11_window_compat_cuda_v1(
    owner: TensorWorldC4ResidentComposedR11OwnerV1,
    *,
    generation: int,
    tick: int,
) -> TensorWorldC4ResidentComposedR11StagingV1:
    """Unqualified compatibility wrapper with eager control preparation.

    This wrapper is retained only for explicit compatibility use.  It is not
    the qualified normal-window execution surface because its control tensor
    writes launch eager CUDA kernels before the graph replay.
    """

    prepare_c4_resident_composed_r11_window_control_v1(
        owner, generation=generation, tick=tick
    )
    return execute_c4_resident_composed_r11_window_cuda_v1(owner)


def c4_resident_composed_r11_source_receipt_v1() -> dict[str, object]:
    """Return source and authority provenance without a gate claim."""

    validate_c4_resident_authority_map_r11_v1()
    sources = {
        "authority_map": R11_AUTHORITY,
        "historical_audit": R11_AUDIT,
        "c3_backend_semantics": C3_BACKEND,
        "r9_adapter": R9_ADAPTER,
        "r11_native_binding": R11_NATIVE_BINDING,
        "r11_native_kernel": R11_NATIVE_KERNEL,
        "r11_native_r9_dependency": R9_NATIVE,
        "r11_native_r5_dependency": R5_NATIVE,
        "r11_native_r4_dependency": R4_NATIVE,
        "r11_native_r2_dependency": R2_NATIVE,
        "r11_native_r6_dependency": R6_NATIVE,
        "r11_transport_r13_dependency": R13_TRANSPORT,
    }
    return {
        "schema_version": C4_RESIDENT_COMPOSED_R11_SCHEMA_VERSION_V1,
        "staging_schema_version": C4_RESIDENT_COMPOSED_R11_STAGING_SCHEMA_VERSION_V1,
        "route": "R11_B48_C32_FIXTURE_AND_C512_PRODUCTION_COMPOSED_RESIDENT_SEAM",
        "admitted_capacities": list(C4_RESIDENT_COMPOSED_R11_ADMITTED_CAPACITIES_V1),
        "production_geometry": "B48_W32_P256_C512",
        "decision": C4_RESIDENT_COMPOSED_R11_DECISION_V1,
        "execution_classification": "B_FIXED_TOPOLOGY_CUDA_GRAPH_OWNER_WITH_DEVICE_TIME",
        "graph_window_ticks": C4_RESIDENT_COMPOSED_R11_GRAPH_WINDOW_TICKS_V1,
        "graph_topology_receipt": (
            "raw_cuda_graph_retained_and_instantiated_at_capture_boundary"
        ),
        "canonical_owner": "TensorWorldStateBundleV4_plus_TensorWorldC5ReplayContextV1",
        "identity_authority": c4_identity_authority_r11_source_receipt_v1(),
        "event_ring_contract": c4_event_ring_r11_source_receipt_v1(),
        "typed_continuation": "C5_TYPED_CONTINUATION_SCHEMA_R11_V1",
        "staging": (
            "R9_fixed_capacity_plus_monotonic_identity_ABA_staging_plus_"
            "device_status_and_time_vectors_plus_"
            "typed_event_ring_banks"
        ),
        "event_authority": (
            "V4_event_ring_double_buffer_native_append_with_legacy_tape_as_"
            "transaction_scratch"
        ),
        "host_tick_authority": False,
        "host_status_authority": False,
        "host_snapshot_rollback": False,
        "python_c3_to_r9_fallback": False,
        "composition_ready": True,
        "gate_claim": False,
        "rollback_model": "device_local_write_ahead_shadow_and_guarded_commit",
        "transport": {
            "bound": True,
            "copy_kernel": "r13_raw_transport_copy_to_shadow",
            "publish_kernel": "r13_raw_transport_publish_if_all_ok",
            "descriptor_owner": "TensorWorldC4ResidentComposedR11OwnerV1",
            "shared_policy_aliases_excluded": sorted(
                _R11_GUARDED_SHARED_TENSOR_PATHS_V1
            ),
        },
        "shadow_policy": {
            "guarded_shared_tensors": sorted(
                _R11_GUARDED_SHARED_TENSOR_PATHS_V1
            ),
            "reason": (
                "C3_read_only_and_C4_batch_global_guarded_newborn_row_writes"
            ),
            "full_policy_clone": False,
        },
        "steady_state_host_boundary": {
            "status_d2h": False,
            "synchronization": False,
            "snapshot_materialization": False,
            "per_world_python_loop": False,
            "bounded_host_launch": True,
            "graph_replays_per_window": 1,
            "eager_device_control_launches": 0,
            "control_preparation": "explicit_boundary_outside_qualified_replay",
        },
        "compatibility_window_wrapper": {
            "qualified": False,
            "reason": "performs_eager_device_control_preparation_before_replay",
        },
        "remaining_gate_work": [
            "five-tick graph-window exact CPU/native parity and target trace",
            "event-ring checkpoint/restore and long-run wrap parity",
            "typed continuation checkpoint/restore parity across the graph window",
            "mechanical residency counters on the final provider route",
            "B48/W32/P256/C512 resident VRAM and rollback workspace evidence",
            "C4 performance distribution against the frozen target",
            "runtime launch-fault fail-stop recovery evidence beyond guarded semantic failures",
        ],
        "authority_map": c4_resident_authority_map_receipt_r11_v1(),
        "sources": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in sources.items()
        },
    }


__all__ = [
    "C4_RESIDENT_COMPOSED_R11_BATCH_V1",
    "C4_RESIDENT_COMPOSED_R11_CAPACITY_V1",
    "C4_RESIDENT_COMPOSED_R11_PRODUCTION_CAPACITY_V1",
    "C4_RESIDENT_COMPOSED_R11_ADMITTED_CAPACITIES_V1",
    "C4_RESIDENT_COMPOSED_R11_DECISION_V1",
    "C4_RESIDENT_COMPOSED_R11_GRAPH_WINDOW_TICKS_V1",
    "C4_RESIDENT_COMPOSED_R11_SCHEMA_VERSION_V1",
    "C4_RESIDENT_COMPOSED_R11_STAGING_SCHEMA_VERSION_V1",
    "TensorWorldC4ResidentComposedR11OwnerV1",
    "TensorWorldC4ResidentComposedR11StagingV1",
    "bind_c4_resident_composed_r11_v1",
    "c4_resident_composed_r11_source_receipt_v1",
    "capture_c4_resident_composed_r11_window_v1",
    "execute_c4_resident_composed_r11_cuda_v1",
    "execute_c4_resident_composed_r11_window_compat_cuda_v1",
    "execute_c4_resident_composed_r11_window_cuda_v1",
    "make_c4_resident_composed_r11_staging_v1",
    "prepare_c4_resident_composed_r11_window_control_v1",
]
