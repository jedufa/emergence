"""Physical-CUDA regression for generic compact science acquisition."""

from __future__ import annotations

import os
from pathlib import Path
import struct
from tempfile import TemporaryDirectory

import pytest

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.science_cuda_v2_native_world_v1 import (
    project_science_cuda_v2_native_world_v1,
)
from engine.native_runtime.cuda_v2_compact_science_capture_v1 import (
    reconstruct_compact_birth_genome_v1,
)
from engine.native_runtime.cuda_v2_science_capture_v1 import (
    _agent_rows,
    _tensor_bytes,
    _world_segment,
)
from engine.perf.cuda_full_evolution_v2.contract import (
    CudaFullEvolutionNativeContractV2,
)
from engine.perf.cuda_full_evolution_v2.direct_sparse_runtime import (
    CudaFullEvolutionDirectSparseOwnerV3,
    seal_direct_sparse_durable_sink_receipt_v3,
)
from engine.perf.cuda_full_evolution_v2.direct_sparse_sustained_benchmark import (
    _publish_prepared_dispatch,
)
from engine.perf.cuda_full_evolution_v2.layout import state_layout_v2


_PREBUILT = os.environ.get("EIEL_FULL_EVOLUTION_CUDA_V2_PREBUILT")

pytestmark = pytest.mark.skipif(
    not _PREBUILT or torch is None or not torch.cuda.is_available(),
    reason="requires a bound CUDA V2 artifact and physical CUDA device",
)


def _world(index: int, profile: str, pressure: float) -> dict[str, object]:
    descriptor = {"algorithm": "sha256", "hex": "1" * 64}
    controls = [
        {
            "control_id": control_id,
            "resolved_value": value,
            "schedule": {"kind": "static"},
            "descriptor_sha256": descriptor,
        }
        for control_id, value in (
            ("reproduction.energy_threshold", 1),
            ("reproduction.energy_cost", 0),
            ("environment.resource.uptake_base", 32),
            ("mutation.rate", 0.125),
            ("mutation.step", 3),
            ("selection.pressure", pressure),
        )
    ]
    semantic = {
        "world": {
            "world_profile_id": "generic-w32",
            "geometry": {"width": 32, "height": 32},
            "resident_capacity": 512,
        },
        "initialization": {
            "initial_population": 256,
            "initial_live_count": 256,
            "founder_profiles": [profile],
            "exact_founder_composition": [
                {"founder_profile_id": profile, "count": 256}
            ],
        },
        "dimension_bindings": {},
        "control_bindings": controls,
        "seed_bindings": {
            "ecology": 7000 + index,
            "founder": 7100 + index,
            "policy": 7200 + index,
            "mutation": 7300 + index,
            "task_environment": 7400 + index,
            "initialization": 7500 + index,
        },
        "time": {
            "authority": "absolute_tick_v1",
            "start_next_tick": 0,
            "total_ticks": 512,
            "checkpoint_ticks": [512],
        },
        "initial_state_identity": {
            "algorithm": "sha256",
            "hex": canonical_sha256({"generic_world": index}),
        },
    }
    science = {
        "test_id": f"GENERIC-{index}",
        "test_version": "1",
        "study_id": "COMPACT-GPU-REGRESSION",
        "arm_id": f"arm-{index}",
        "replicate": index,
    }
    return project_science_cuda_v2_native_world_v1(
        {"semantic_inputs": semantic, "science": science}
    ).to_dict()


def _ack(owner, dispatch, step: int) -> None:
    sample_id = f"compact-science-gpu-{step}"
    with TemporaryDirectory(prefix="eiel-compact-science-gpu-") as temporary:
        prepared = owner.prepare_sparse_event_durable_envelope_v3(
            dispatch, sample_id=sample_id
        )
        _, envelope = _publish_prepared_dispatch(
            prepared,
            output_directory=Path(temporary),
            sample_id=sample_id,
        )
        try:
            receipt = seal_direct_sparse_durable_sink_receipt_v3(
                dispatch,
                sink_id="compact-science-gpu-regression",
                sink_receipt_id=sample_id,
                sample_id=sample_id,
                durable_envelope=envelope,
            )
            owner.acknowledge_sparse_event_drain_v3(
                receipt, durable_envelope=envelope
            )
        finally:
            envelope.close()


def test_mixed_world_compact_metrics_align_with_sparse_events() -> None:
    controls = (
        _world(1, "founder-neutral-v1", 0.5),
        _world(2, "founder-neutral-v1", 1.5),
    )
    owner = CudaFullEvolutionDirectSparseOwnerV3.initialize(
        CudaFullEvolutionNativeContractV2(
            B=2, total_ticks=512, temporal_k=64, start_next_tick=0
        ),
        seed=7001,
        artifact_path=Path(str(_PREBUILT)),
        device="cuda:0",
        world_uids=(101, 202),
        initial_capacity_rows=1024,
        native_world_controls=controls,
        compact_science_capture=True,
    )
    event_count = 0
    event_metric_rows = 0
    birth_provenance_rows = 0
    birth_event_count = 0
    event_metric_sum = 0
    genomes: dict[int, dict[int, dict[str, list[int]]]] = {}
    initial_raw = _tensor_bytes(
        owner.state_projection(), field="initial compact science state"
    )
    for world_index, world_uid in enumerate((101, 202)):
        founders = _agent_rows(
            _world_segment(initial_raw, world_index=world_index),
            world_uid=world_uid,
            tick=0,
        )
        genomes[world_uid] = {
            agent_id: {
                "policy": list(agent["_native_genome_projection"]["policy"]),
                "trait": list(agent["_native_genome_projection"]["trait"]),
            }
            for agent_id, agent in founders.items()
        }
    mutation_origins = {101: 0, 202: 0}
    try:
        for step in range(8):
            dispatch, capture = owner.advance_and_drain_compact_science_v1(64)
            event_count += dispatch.event_count
            event_metric_rows += int(capture.event_agent_metrics.shape[0])
            birth_provenance_rows += int(
                capture.event_birth_provenance.shape[0]
            )
            event_metric_sum += int(capture.event_agent_metrics.sum().item())
            for row_index, (_, _, _, packed_key) in enumerate(
                struct.iter_unpack("<QQQQ", dispatch.packed_events)
            ):
                event_kind = (packed_key >> 6) & 0x3
                provenance = capture.event_birth_provenance[row_index]
                if event_kind == 1:
                    assert not bool(provenance.any().item())
                else:
                    assert event_kind == 2
                    birth_event_count += 1
                    mutation_codes = provenance[:128]
                    assert not bool((mutation_codes & 0xF0).any().item())
                    assert not bool(
                        (((mutation_codes & 0x3) == 3).any()).item()
                    )
                    assert not bool(
                        ((((mutation_codes >> 2) & 0x3) == 3).any()).item()
                    )
            unpacked = tuple(struct.iter_unpack("<QQQQ", dispatch.packed_events))
            for world_index, world_uid in enumerate(dispatch.world_uids):
                begin = dispatch.world_offsets[world_index]
                end = dispatch.world_offsets[world_index + 1]
                for row_index in range(begin, end):
                    child_id, parent_a_id, parent_b_id, packed_key = unpacked[
                        row_index
                    ]
                    if ((packed_key >> 6) & 0x3) != 2:
                        continue
                    reconstruction = reconstruct_compact_birth_genome_v1(
                        parent_a_genome=genomes[world_uid][parent_a_id],
                        parent_b_genome=genomes[world_uid][parent_b_id],
                        birth_provenance=(
                            capture.event_birth_provenance[row_index].tolist()
                        ),
                        mutation_step=3,
                    )
                    genomes[world_uid][child_id] = reconstruction["child_genome"]
                    mutation_origins[world_uid] += int(
                        reconstruction["mutation_origin_count"]
                    )
            _ack(owner, dispatch, step)
            current_state = owner.state_projection()
            current_raw = _tensor_bytes(
                current_state, field="compact science reconstruction state"
            )
            headers = current_state[:, :192].view(torch.int64).cpu().tolist()
            for world_index, world_uid in enumerate(dispatch.world_uids):
                agents = _agent_rows(
                    _world_segment(current_raw, world_index=world_index),
                    world_uid=world_uid,
                    tick=(step + 1) * 64,
                )
                for agent_id, agent in agents.items():
                    projection = agent["_native_genome_projection"]
                    assert list(projection["policy"]) == genomes[world_uid][agent_id][
                        "policy"
                    ]
                    assert list(projection["trait"]) == genomes[world_uid][agent_id][
                        "trait"
                    ]
                assert int(headers[world_index][17]) == mutation_origins[world_uid]
        active = owner.compact_science_active_metrics_projection_v1()
        binding = owner.native_world_control_binding_v1
        receipt = owner.execution_receipt_v3()
    finally:
        owner.close()
    assert event_count == event_metric_rows
    assert event_count == birth_provenance_rows
    assert event_count > 0
    assert birth_event_count > 0
    assert event_metric_sum > 0 or int(active["tensor"].sum().item()) > 0
    assert binding["control_set_sha256"] == (
        receipt["native_world_control"]["control_set_sha256"]
    )


def test_native_snapshot_capture_detaches_exact_resident_state_to_pinned_host() -> None:
    """Exercise the coarse resident CUDA-to-bounded-host snapshot boundary."""

    controls = (
        _world(1, "founder-neutral-v1", 0.5),
        _world(2, "founder-neutral-v1", 1.5),
    )
    owner = CudaFullEvolutionDirectSparseOwnerV3.initialize(
        CudaFullEvolutionNativeContractV2(
            B=2, total_ticks=1, temporal_k=1, start_next_tick=0
        ),
        seed=7001,
        artifact_path=Path(str(_PREBUILT)),
        device="cuda:0",
        world_uids=(101, 202),
        initial_capacity_rows=1024,
        native_world_controls=controls,
        compact_science_capture=True,
    )
    try:
        expected_state = owner.state_projection().cpu().contiguous()
        expected_metrics = owner.compact_science_active_metrics_projection_v1()[
            "tensor"
        ]
        state_destination = torch.empty(
            tuple(expected_state.shape),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        metrics_destination = torch.empty(
            tuple(expected_metrics.shape),
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )

        fence = owner.capture_native_snapshot_to_pinned_v1(
            absolute_tick=0,
            state_destination=state_destination,
            active_science_metrics_destination=metrics_destination,
        )
        capture = fence.wait()

        assert torch.equal(state_destination, expected_state)
        assert torch.equal(metrics_destination, expected_metrics)
        assert capture == {
            "schema_version": "eiel_cuda_pinned_snapshot_capture_fence_v1",
            "absolute_tick": 0,
            "source_device": "cuda:0",
            "state_bytes": expected_state.numel() * expected_state.element_size(),
            "active_science_metrics_bytes": (
                expected_metrics.numel() * expected_metrics.element_size()
            ),
            "d2h_bytes": (
                expected_state.numel() * expected_state.element_size()
                + expected_metrics.numel() * expected_metrics.element_size()
            ),
            "cuda_event_time_ns": capture["cuda_event_time_ns"],
        }
        assert capture["cuda_event_time_ns"] >= 0
        assert state_destination.is_pinned()
        assert metrics_destination.is_pinned()

        canonical_bytes = state_layout_v2().canonical_checkpoint_bytes_per_world
        canonical_state_destination = torch.empty(
            (len(controls), canonical_bytes),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        canonical_metrics_destination = torch.empty(
            tuple(expected_metrics.shape),
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )
        canonical_fence = owner.capture_native_snapshot_to_pinned_v1(
            absolute_tick=0,
            state_destination=canonical_state_destination,
            active_science_metrics_destination=canonical_metrics_destination,
        )
        canonical_capture = canonical_fence.wait()
        assert torch.equal(
            canonical_state_destination,
            expected_state[:, :canonical_bytes],
        )
        assert torch.equal(canonical_metrics_destination, expected_metrics)
        assert canonical_capture["state_bytes"] == (
            len(controls) * canonical_bytes
        )
        assert canonical_capture["d2h_bytes"] == (
            canonical_capture["state_bytes"]
            + canonical_capture["active_science_metrics_bytes"]
        )
        assert canonical_capture["cuda_event_time_ns"] >= 0
    finally:
        owner.close()
