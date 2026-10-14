"""C512 exact-success/failure qualification harness for composed C4 R11.

The independent comparator is validation-only: it runs the qualified C2/C3
CUDA phases eagerly, materializes the full canonical boundary, then applies
the transactional CPU C4 oracle and R11 event-ring projection.  Reference and
resident candidate run sequentially and this route is never a production path.
"""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass, replace
import gc
import os
from pathlib import Path
import runpy
import subprocess
from typing import Final, Mapping

import pytest
import torch

from engine.tensor_world.c2_backend import (
    TensorWorldC2ConfigV1,
    _config_tensors_v1,
    _native_state_v1,
)
from engine.tensor_world.c2_native_loader import load_tensor_world_c2_extension
from engine.tensor_world.c3_mechanics import TensorWorldC3MechanicsConfigV1
from engine.tensor_world.c4_event_ring_r11 import (
    TensorWorldC4EventRingRowR11V1,
    _event_ring_tensors,
    make_c4_event_ring_r11_v1,
)
from engine.tensor_world.c4_events import C4_EVENT_DEATH_COMMITTED_V1
from engine.tensor_world.c4_fixture import make_c4_fixture_v1
from engine.tensor_world.c4_mating import TensorWorldC4MateBirthConfigV1
from engine.tensor_world.c4_resident_composed_r11 import (
    _run_c3_phase_v1,
    bind_c4_resident_composed_r11_v1,
    capture_c4_resident_composed_r11_window_v1,
    execute_c4_resident_composed_r11_window_cuda_v1,
    make_c4_resident_composed_r11_staging_v1,
    prepare_c4_resident_composed_r11_window_control_v1,
)
from engine.tensor_world.c4_transaction import (
    execute_c4_population_transaction_oracle_v1,
)
from engine.tensor_world.c5_determinism import make_c5_pcg64_context_v1
from engine.tensor_world.contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_PROBE = ROOT / (
    "tests/test_eiel_tensor_world_c4_resident_composed_r11_production_cuda_v1.py"
)
COMPOSED = ROOT / "engine/tensor_world/c4_resident_composed_r11.py"
C2_BACKEND = ROOT / "engine/tensor_world/c2_backend.py"
C3_NATIVE = ROOT / "engine/tensor_world/c3_native.py"
C3_MECHANICS = ROOT / "engine/tensor_world/c3_mechanics.py"
C4_TRANSACTION = ROOT / "engine/tensor_world/c4_transaction.py"

QUALIFICATION_ENV: Final[str] = "EIEL_C4_R11_C512_EXACT_QUALIFICATION"
PURE_CPU_ORACLE_GAP: Final[str] = (
    "source-current CPU has no complete C2 WDL/EED/patch/jungle/observe/"
    "policy/select/settle trajectory and no integrated R11 event-ring/run-control "
    "projection; C4-only parity is not an exact five-tick composed oracle"
)
REFERENCE_CLASSIFICATION: Final[str] = (
    "validation_only_host_orchestrated_eager_C2_C3_CUDA_then_CPU_C4"
)
RUN_CONTROL_FIELDS: Final[tuple[str, ...]] = (
    "c3_status",
    "c4_status",
    "commit_status",
    "generation",
    "tick",
    "shadow_event_sequence",
    "requested_ticks",
    "completed_ticks",
    "current_tick",
    "global_status",
    "first_error_stage",
    "first_error_world",
    "first_error_code",
    "first_error_tick",
    "stop_requested",
    "receipt_write_index",
)


def _tensor_leaves(value: object, path: str) -> dict[str, torch.Tensor]:
    leaves: dict[str, torch.Tensor] = {}
    if isinstance(value, torch.Tensor):
        leaves[path] = value
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            leaves.update(_tensor_leaves(item, f"{path}[{index}]"))
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            leaves.update(
                _tensor_leaves(getattr(value, field.name), f"{path}.{field.name}")
            )
    return leaves


def _canonical_leaves(bundle: object, replay: object) -> dict[str, torch.Tensor]:
    """Inventory every bundle/replay tensor, including the nested event ring."""

    leaves = _tensor_leaves(bundle, "bundle")
    replay_leaves = _tensor_leaves(replay, "replay")
    if set(leaves).intersection(replay_leaves):
        raise AssertionError("canonical bundle and replay tensor paths overlap")
    leaves.update(replay_leaves)
    if not any(path.startswith("bundle.event_ring.") for path in leaves):
        raise AssertionError("canonical C512 inventory omitted the R11 event ring")
    return leaves


def _run_control_leaves(staging: object) -> dict[str, torch.Tensor]:
    return {
        f"control.{name}": getattr(staging, name)
        for name in RUN_CONTROL_FIELDS
    }


def _byte_snapshot(tensors: Mapping[str, torch.Tensor]) -> dict[str, bytes]:
    return {
        path: tensor.detach()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .cpu()
        .numpy()
        .tobytes()
        for path, tensor in tensors.items()
    }


def _assert_maps_byte_exact(
    expected: Mapping[str, torch.Tensor], actual: Mapping[str, torch.Tensor]
) -> None:
    assert expected.keys() == actual.keys()
    for path in expected:
        left = expected[path]
        right = actual[path]
        assert left.dtype == right.dtype, path
        assert left.shape == right.shape, path
        assert _byte_snapshot({path: left})[path] == _byte_snapshot({path: right})[
            path
        ], path


def _production_helpers() -> Mapping[str, object]:
    """Reuse the frozen production fixture without copying its large setup."""

    return runpy.run_path(str(PRODUCTION_PROBE), run_name="_c4_r11_c512_probe_helpers")


def _copy_tensor_maps_(
    destination: Mapping[str, torch.Tensor], source: Mapping[str, torch.Tensor]
) -> None:
    """Copy the complete canonical tensor inventory at a validation boundary."""

    if destination.keys() != source.keys():
        raise AssertionError("reference boundary tensor inventory drifted")
    with torch.no_grad():
        for path in destination:
            target = destination[path]
            value = source[path]
            if target.dtype != value.dtype or target.shape != value.shape:
                raise AssertionError(f"reference boundary ABI drifted at {path}")
            target.copy_(value)


def _reset_r11_local_event_scratch_(bundle: object, replay: object) -> None:
    """Mirror ``reset_local_event_cursor`` before the CPU C4 transaction."""

    events = bundle.events
    ring = bundle.event_ring
    if events is None or ring is None:
        raise TensorWorldContractError("hybrid reference requires events and ring")
    if not torch.equal(replay.event_sequence, ring.producer_cursor):
        raise TensorWorldContractError(
            "hybrid reference event ring/replay cursor mismatch"
        )
    events.active_count.zero_()
    events.status.zero_()
    replay.event_sequence.zero_()


def _ring_row_from_scratch(bundle: object, world: int, row: int):
    tape = bundle.events
    if tape is None:
        raise TensorWorldContractError("hybrid reference event scratch is absent")
    return TensorWorldC4EventRingRowR11V1(
        event_type=int(tape.typed_event_type[world, row]),
        generation=int(tape.typed_generation[world, row]),
        tick=int(tape.typed_tick[world, row]),
        actor_id=int(tape.typed_actor_id[world, row]),
        target_id=int(tape.typed_target_id[world, row]),
        value=float(tape.typed_value[world, row]),
        actor_slot=int(tape.typed_actor_slot[world, row]),
        target_slot=int(tape.typed_target_slot[world, row]),
        parent_id=int(tape.typed_parent_id[world, row]),
        child_id=int(tape.typed_child_id[world, row]),
        founder_id=int(tape.typed_founder_id[world, row]),
        lineage_depth=int(tape.typed_lineage_depth[world, row]),
        subtype=int(tape.typed_subtype[world, row]),
        reason=int(tape.typed_reason[world, row]),
        success=int(tape.typed_success[world, row]),
    )


def _publish_r11_event_ring_oracle_(
    bundle: object, replay: object, *, generation: int
) -> None:
    """Project local CPU C4 rows into the durable R11 ring atomically."""

    events = bundle.events
    ring = bundle.event_ring
    if events is None or ring is None:
        raise TensorWorldContractError("hybrid reference requires events and ring")
    rows = torch.arange(events.capacity, dtype=torch.int64).unsqueeze(0)
    valid = rows < events.active_count.unsqueeze(1)
    death = valid & (events.typed_event_type == C4_EVENT_DEATH_COMMITTED_V1)
    events.generation.masked_fill_(death, generation)
    events.typed_generation.masked_fill_(death, generation)

    shadow = copy.deepcopy(ring)
    for world in range(bundle.batch):
        count = int(events.active_count[world])
        shadow.append(
            world=world,
            rows=tuple(
                _ring_row_from_scratch(bundle, world, row)
                for row in range(count)
            ),
        )
    for target, value in zip(
        _event_ring_tensors(ring), _event_ring_tensors(shadow), strict=True
    ):
        target.copy_(value)
    events.active_count.zero_()
    events.status.zero_()
    replay.event_sequence.copy_(ring.producer_cursor)


def _successful_run_control(
    *, batch: int, generation: int, tick: int
) -> Mapping[str, torch.Tensor]:
    def i64(value: int) -> torch.Tensor:
        return torch.full((batch,), value, dtype=torch.int64)

    def i32(value: int) -> torch.Tensor:
        return torch.full((batch,), value, dtype=torch.int32)

    return {
        "control.c3_status": i32(0),
        "control.c4_status": i32(0),
        "control.commit_status": i32(0),
        "control.generation": i64(generation + 5),
        "control.tick": i64(tick + 5),
        "control.shadow_event_sequence": i64(0),
        "control.requested_ticks": i64(5),
        "control.completed_ticks": i64(5),
        "control.current_tick": i64(tick + 4),
        "control.global_status": i32(0),
        "control.first_error_stage": i32(-1),
        "control.first_error_world": i32(-1),
        "control.first_error_code": i32(0),
        "control.first_error_tick": i64(-1),
        "control.stop_requested": i32(0),
        "control.receipt_write_index": i64(5),
    }


def _run_host_orchestrated_reference_window(
    bundle: object,
    replay: object,
    *,
    config: TensorWorldC4MateBirthConfigV1,
    generation: int,
    tick: int,
    c2_extension_path: str | Path,
    move_in_place: object,
) -> tuple[object, object, Mapping[str, torch.Tensor]]:
    """Run eager qualified C2/C3, then independent CPU C4, for five ticks.

    This route is validation-only.  Every canonical tensor is materialized at
    the declared C3/C4 boundary and republished before the next tick.  The
    reference device state is released before the resident candidate exists.
    """

    if bundle.device.type != "cpu" or replay.device.type != "cpu":
        raise TensorWorldContractError("hybrid reference requires CPU owners")
    if not callable(move_in_place):
        raise TensorWorldContractError("hybrid reference move helper is invalid")
    device = torch.device("cuda")
    cuda_bundle = move_in_place(copy.deepcopy(bundle), device)
    cuda_replay = move_in_place(copy.deepcopy(replay), device)
    c2_config = TensorWorldC2ConfigV1(
        max_energy=float(config.max_energy), start_health=float(config.start_health)
    )
    mechanics = TensorWorldC3MechanicsConfigV1(
        max_energy=float(config.max_energy),
        start_health=float(config.start_health),
        enable_chemistry=False,
        chemistry_noise_required=False,
    )
    native = load_tensor_world_c2_extension(str(c2_extension_path))
    native_state = _native_state_v1(cuda_bundle.base.base)
    config_ints, config_floats = _config_tensors_v1(
        c2_config, cuda_bundle.base.base
    )

    for offset in range(5):
        _run_c3_phase_v1(
            cuda_bundle,
            native,
            native_state,
            config_ints,
            config_floats,
            mechanics,
        )
        torch.cuda.synchronize(device)
        _copy_tensor_maps_(
            _canonical_leaves(bundle, replay),
            _canonical_leaves(cuda_bundle, cuda_replay),
        )
        current_generation = generation + offset
        current_tick = tick + offset
        bundle.base.base.core.tick = current_tick
        _reset_r11_local_event_scratch_(bundle, replay)
        # The resident R11 transaction writes events into a local scratch tape
        # and publishes them into the durable ring only after the whole C4
        # transaction commits.  The standalone CPU C4 oracle predates that
        # outer ring owner, so temporarily detach the ring while it advances
        # the local event cursor, then project its committed rows below.
        durable_ring = bundle.event_ring
        bundle.event_ring = None
        try:
            execute_c4_population_transaction_oracle_v1(
                bundle,
                mate_config=replace(config, generation=current_generation),
                rng_context=replay,
            )
        finally:
            bundle.event_ring = durable_ring
        _publish_r11_event_ring_oracle_(
            bundle, replay, generation=current_generation
        )
        if offset != 4:
            _copy_tensor_maps_(
                _canonical_leaves(cuda_bundle, cuda_replay),
                _canonical_leaves(bundle, replay),
            )
            torch.cuda.synchronize(device)

    del native_state, config_ints, config_floats, cuda_bundle, cuda_replay
    gc.collect()
    torch.cuda.empty_cache()
    return bundle, replay, _successful_run_control(
        batch=bundle.batch, generation=generation, tick=tick
    )


def test_c512_harness_binds_frozen_fixture_explicit_control_and_full_inventory() -> None:
    production = PRODUCTION_PROBE.read_text(encoding="utf-8")
    composed = COMPOSED.read_text(encoding="utf-8")

    for sentinel in (
        "batch=48",
        "capacity=512",
        "active=256",
        "width=32",
        "tape_capacity=65_536",
        "max_births=64",
    ):
        assert sentinel in production
    assert "prepare_c4_resident_composed_r11_window_control_v1(" in production
    assert "execute_c4_resident_composed_r11_window_cuda_v1(owner)" in production
    assert "execute_c4_resident_composed_r11_window_compat_cuda_v1" not in production
    assert "for _step in range(ticks):" in composed
    assert "_run_c3_phase_v1(" in composed
    assert "_r11_population_call_v1(" in composed
    assert RUN_CONTROL_FIELDS == (
        "c3_status",
        "c4_status",
        "commit_status",
        "generation",
        "tick",
        "shadow_event_sequence",
        "requested_ticks",
        "completed_ticks",
        "current_tick",
        "global_status",
        "first_error_stage",
        "first_error_world",
        "first_error_code",
        "first_error_tick",
        "stop_requested",
        "receipt_write_index",
    )


def test_c512_hybrid_reference_closes_only_the_exact_missing_composition_seam() -> None:
    c2 = C2_BACKEND.read_text(encoding="utf-8")
    c3 = C3_NATIVE.read_text(encoding="utf-8")
    mechanics = C3_MECHANICS.read_text(encoding="utf-8")
    transaction = C4_TRANSACTION.read_text(encoding="utf-8")

    assert 'raise TensorWorldContractError("Tensor World C2 is a CUDA-only backend")' in c2
    for phase in (
        "wdl_step",
        "eed_step",
        "patch_step",
        "jungle_step",
        "observe",
        "policy_logits",
        "select_actions",
        "settle",
    ):
        assert f"self._native.{phase}(" in c2
    assert "def c3_observation_cpu_oracle_v1(" in c3
    assert "def c3_fixed_mlp_policy_cpu_oracle_v1(" in c3
    assert "def apply_c3_active_mechanics_oracle_v1(" in mechanics
    assert "def execute_c4_population_transaction_oracle_v1(" in transaction
    assert "event_ring" not in transaction
    assert "C4-only parity is not an exact five-tick composed oracle" in (
        PURE_CPU_ORACLE_GAP
    )
    assert REFERENCE_CLASSIFICATION == (
        "validation_only_host_orchestrated_eager_C2_C3_CUDA_then_CPU_C4"
    )

    source = Path(__file__).read_text(encoding="utf-8")
    reference = source.split(
        "def _run_host_orchestrated_reference_window", 1
    )[1].split("def test_c512_harness_binds", 1)[0]
    assert "for offset in range(5):" in reference
    assert reference.index("_run_c3_phase_v1(") < reference.index(
        "execute_c4_population_transaction_oracle_v1("
    )
    assert "_copy_tensor_maps_(" in reference
    assert "_publish_r11_event_ring_oracle_(" in reference
    assert "torch.cuda.empty_cache()" in reference


def test_hybrid_ring_projection_preserves_r11_device_time_and_absolute_cursor() -> None:
    bundle, _ = make_c4_fixture_v1(
        batch=2,
        capacity=4,
        active=2,
        width=4,
        seed=151,
        world_local_identity=True,
    )
    assert bundle.events is not None
    bundle.event_ring = make_c4_event_ring_r11_v1(
        batch=2,
        capacity=4,
        event_capacity=bundle.events.capacity,
        device="cpu",
    )
    replay = make_c5_pcg64_context_v1(bundle, streams=5, seed=151, typed=True)
    events = bundle.events
    events.active_count.fill_(1)
    events.event_type[:, 0] = C4_EVENT_DEATH_COMMITTED_V1
    events.typed_event_type[:, 0] = C4_EVENT_DEATH_COMMITTED_V1
    events.generation[:, 0] = 0
    events.typed_generation[:, 0] = 0
    events.tick[:, 0] = 911
    events.typed_tick[:, 0] = 911
    events.actor_id[:, 0] = torch.tensor([0, 1])
    events.typed_actor_id[:, 0] = torch.tensor([0, 1])

    _publish_r11_event_ring_oracle_(bundle, replay, generation=37)

    assert bundle.event_ring.producer_cursor.tolist() == [1, 1]
    assert replay.event_sequence.tolist() == [1, 1]
    assert events.active_count.tolist() == [0, 0]
    assert events.status.tolist() == [0, 0]
    for world in range(2):
        rows = bundle.event_ring.rows(world=world)
        assert len(rows) == 1
        assert rows[0].event_type == C4_EVENT_DEATH_COMMITTED_V1
        assert rows[0].generation == 37
        assert rows[0].tick == 911


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(
    os.environ.get(QUALIFICATION_ENV) != "1",
    reason="C512 exact qualification is explicitly opt-in",
)
def test_c512_five_tick_exact_success_and_later_world_rollback(
    tmp_path: Path,
) -> None:
    """Qualify full success parity and deterministic world-47 rollback."""

    helpers = _production_helpers()
    prepare_fixture = helpers["_prepare_m64_fixture"]
    move_in_place = helpers["_move_in_place"]
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert len(source_commit) == 40
    cpu_bundle, cpu_replay = prepare_fixture(source_commit=source_commit)
    config = TensorWorldC4MateBirthConfigV1(
        max_births=64,
        allow_mutation=True,
        max_energy=160.0,
    )
    c2_extension_path = os.environ.get(
        "TENSOR_WORLD_C2_PREBUILT",
        "/tmp/emergence-test/runtime/evidence_raw/tensor_world/production_c0_c7/"
        "20260815_c2_430489534_build_01/tensor_world_c2_ext.so",
    )
    expected_bundle, expected_replay, expected_control = (
        _run_host_orchestrated_reference_window(
            cpu_bundle,
            cpu_replay,
            config=config,
            generation=100,
            tick=1_000,
            c2_extension_path=c2_extension_path,
            move_in_place=move_in_place,
        )
    )

    def make_candidate():
        source_bundle, source_replay = prepare_fixture(source_commit=source_commit)
        bundle = move_in_place(source_bundle, torch.device("cuda"))
        replay = move_in_place(source_replay, torch.device("cuda"))
        staging = make_c4_resident_composed_r11_staging_v1(
            batch=48,
            capacity=512,
            grid_size=32,
            streams=int(replay.rng_state_words.shape[1]),
            device="cuda",
        )
        owner = bind_c4_resident_composed_r11_v1(
            bundle,
            replay,
            staging,
            enable_native=True,
            population_config=config,
            c2_extension_path=c2_extension_path,
            native_build_directory=Path(
                os.environ.get("EIEL_C4_R11_BUILD_DIR", tmp_path / "build")
            ),
            source_commit=source_commit,
        )
        capture_c4_resident_composed_r11_window_v1(owner, config)
        prepare_c4_resident_composed_r11_window_control_v1(
            owner, generation=100, tick=1_000
        )
        return bundle, replay, owner

    bundle, replay, owner = make_candidate()
    result = execute_c4_resident_composed_r11_window_cuda_v1(owner)
    torch.cuda.synchronize()
    _assert_maps_byte_exact(
        _canonical_leaves(expected_bundle, expected_replay),
        _canonical_leaves(bundle, replay),
    )
    _assert_maps_byte_exact(expected_control, _run_control_leaves(result))

    del result, owner, bundle, replay
    gc.collect()
    torch.cuda.empty_cache()

    failed_bundle, failed_replay, failed_owner = make_candidate()
    assert failed_bundle.event_ring is not None
    failed_replay.event_sequence[47] = (
        failed_bundle.event_ring.producer_cursor[47] + 1
    )
    authoritative_before = _byte_snapshot(
        _canonical_leaves(failed_bundle, failed_replay)
    )
    failure = execute_c4_resident_composed_r11_window_cuda_v1(failed_owner)
    torch.cuda.synchronize()
    assert failure.global_status.cpu().tolist() == [1] * 48
    assert failure.completed_ticks.cpu().tolist() == [0] * 48
    assert failure.first_error_stage.cpu().tolist() == [4] * 48
    assert failure.first_error_world.cpu().tolist() == [47] * 48
    assert failure.first_error_code.cpu().tolist() == [45] * 48
    assert failure.first_error_tick.cpu().tolist() == [1_000] * 48
    assert _byte_snapshot(
        _canonical_leaves(failed_bundle, failed_replay)
    ) == authoritative_before
