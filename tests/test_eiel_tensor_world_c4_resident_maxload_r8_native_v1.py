"""Focused static and CUDA parity/rollback tests for the native C4 R8 probe."""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
import os
from pathlib import Path

import pytest
import torch

from engine.tensor_world.c4_population import (
    apply_c4_death_settlement_oracle_v1,
    commit_c4_ordinary_births_oracle_v1,
    plan_c4_ordinary_births_oracle_v1,
)
from engine.tensor_world.c4_resident_birth_r2 import (
    _half_tensors,
    _inherit_tensors,
    _minus_one_tensors,
    _one_tensors,
    _primary_tensors,
    _zero_tensors,
)
from engine.tensor_world.c4_resident_maxload_r8 import (
    make_c4_resident_maxload_r8_fixture_v1,
)
from engine.tensor_world.c4_resident_maxload_r8_native import (
    C4_RESIDENT_MAXLOAD_R8_NATIVE_DECISION_V1,
    C4_RESIDENT_MAXLOAD_R8_NATIVE_SCHEMA_VERSION_V1,
    c4_resident_maxload_r8_native_source_receipt_v1,
    execute_c4_resident_maxload_r8_cuda_v1,
)
from engine.tensor_world.c4_resident_mutation_r4 import (
    _replay_tensors,
    _scalar_rng_tensors,
)
from engine.tensor_world.c4_resident_turnover_r3 import (
    _death_authoritative,
    make_c4_resident_turnover_r3_staging_v1,
)
from engine.tensor_world.contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[1]


def _move_dataclass_tensors(value: object, device: torch.device) -> object:
    clone = copy.deepcopy(value)

    def move(owner: object) -> None:
        if not is_dataclass(owner):
            return
        for field in fields(owner):
            item = getattr(owner, field.name)
            if isinstance(item, torch.Tensor):
                setattr(owner, field.name, item.to(device=device))
            elif is_dataclass(item):
                move(item)

    move(clone)
    return clone


def _authoritative(bundle, replay) -> list[torch.Tensor]:
    return [
        *_primary_tensors(bundle),
        *_death_authoritative(bundle),
        *_inherit_tensors(bundle),
        *_zero_tensors(bundle),
        *_minus_one_tensors(bundle),
        *_half_tensors(bundle),
        *_one_tensors(bundle),
        *_scalar_rng_tensors(bundle),
        *_replay_tensors(replay),
    ]


def _make_cuda_case(seed: int):
    fixture = make_c4_resident_maxload_r8_fixture_v1(seed=seed)
    cuda_bundle = _move_dataclass_tensors(fixture.bundle, torch.device("cuda"))
    cuda_replay = _move_dataclass_tensors(fixture.replay, torch.device("cuda"))
    staging = make_c4_resident_turnover_r3_staging_v1(
        batch=4, capacity=32, grid_size=32, device="cuda"
    )
    staging.birth.spawn_uniforms.copy_(fixture.spawn_uniforms.to(device="cuda"))
    staging.birth.event_sequence.copy_(cuda_replay.event_sequence)
    return fixture, cuda_bundle, cuda_replay, staging


def test_c4_r8_native_receipt_freezes_exact_probe_and_r5_seam() -> None:
    receipt = c4_resident_maxload_r8_native_source_receipt_v1()
    assert receipt["schema_version"] == C4_RESIDENT_MAXLOAD_R8_NATIVE_SCHEMA_VERSION_V1
    assert receipt["decision"] == C4_RESIDENT_MAXLOAD_R8_NATIVE_DECISION_V1
    assert receipt["route"] == "R8_B4_C32_EXACT_8_DEATH_8_ORDINARY_BIRTH_MAXLOAD_PROBE"
    assert receipt["admission"] == (
        "every_world_exactly_8_deaths_8_births_and_all_children_reuse_new_deaths"
    )
    assert "mate_mutation_only" in receipt["frozen_r5_seam"]
    assert receipt["execution_classification"] == (
        "host_launched_component_probe_not_production_residency"
    )
    assert receipt["gate_claim"] is False
    for source in receipt["sources"].values():
        assert len(source["sha256"]) == 64
        assert (ROOT / source["path"]).is_file()


def test_c4_r8_native_source_has_b4_abi_and_batch_global_admission() -> None:
    source = (
        ROOT
        / "engine/tensor_world/native/tensor_world_c4_resident_maxload_r8.cu"
    ).read_text()
    assert "primary[0].size(0) == 4" in source
    assert "primary[0].size(1) == 32" in source
    assert "turn.death_count[world] == 8" in source
    assert "birth.planned_count[world] == 8" in source
    assert "merge_status_r3<<<1, 1" in source
    assert "publish_replay_sequence_r8" in source
    assert '#include "tensor_world_c4_resident_turnover_r3.cu"' in source


def test_c4_r8_native_rejects_cpu_without_loading_or_fallback() -> None:
    fixture = make_c4_resident_maxload_r8_fixture_v1()
    with pytest.raises(TensorWorldContractError, match="requires a CUDA V4 bundle"):
        execute_c4_resident_maxload_r8_cuda_v1(
            fixture.bundle,
            None,  # type: ignore[arg-type]
            fixture.birth_config,
            replay=fixture.replay,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_c4_r8_native_exact_maxload_parity_and_three_overflow_rollbacks(
    tmp_path: Path,
) -> None:
    build_dir = Path(os.environ.get("EIEL_C4_R8_BUILD_DIR", tmp_path / "build"))
    expected, actual, actual_replay, staging = _make_cuda_case(seed=67)
    apply_c4_death_settlement_oracle_v1(
        expected.bundle, event_sequence=expected.replay.event_sequence
    )
    plan_c4_ordinary_births_oracle_v1(
        expected.bundle,
        expected.birth_config,
        spawn_uniforms=expected.spawn_uniforms,
    )
    commit_c4_ordinary_births_oracle_v1(expected.bundle, expected.birth_config)
    expected.replay.event_sequence.copy_(expected.bundle.events.active_count)

    execute_c4_resident_maxload_r8_cuda_v1(
        actual,
        staging,
        expected.birth_config,
        replay=actual_replay,
        build_directory=build_dir,
    )
    torch.cuda.synchronize()
    assert staging.death_status.cpu().tolist() == [0, 0, 0, 0]
    assert staging.death_count.cpu().tolist() == [8, 8, 8, 8]
    assert staging.birth.status.cpu().tolist() == [0, 0, 0, 0]
    assert staging.birth.planned_count.cpu().tolist() == [8, 8, 8, 8]
    assert staging.birth.child_slots[:, :8].cpu().tolist() == [list(range(8, 16))] * 4
    assert actual.events.typed_event_type[:, :16].cpu().tolist() == [
        [4] * 8 + [1] * 8
    ] * 4
    for index, (cpu_tensor, cuda_tensor) in enumerate(
        zip(
            _authoritative(expected.bundle, expected.replay),
            _authoritative(actual, actual_replay),
            strict=True,
        )
    ):
        torch.testing.assert_close(
            cuda_tensor.cpu(),
            cpu_tensor,
            equal_nan=True,
            msg=f"R8 maximum authoritative tensor {index}",
        )

    for case_index, overflow in enumerate(("event", "queue", "identity")):
        failed, failed_bundle, failed_replay, failed_staging = _make_cuda_case(
            seed=71 + case_index
        )
        if overflow == "event":
            failed_bundle.events.active_count[2] = failed_bundle.events.capacity - 15
            failed_replay.event_sequence.copy_(failed_bundle.events.active_count)
            failed_staging.birth.event_sequence.copy_(failed_replay.event_sequence)
        elif overflow == "queue":
            population = failed_bundle.population
            population.free_queue_count[3] = 25
            population.free_queue[3, :16] = (
                failed_bundle.base.base.core.identity.slot_ids[3, 16:32]
            )
        else:
            failed_bundle.population.next_agent_id[1] = 30
        before = [
            tensor.clone() for tensor in _authoritative(failed_bundle, failed_replay)
        ]
        execute_c4_resident_maxload_r8_cuda_v1(
            failed_bundle,
            failed_staging,
            failed.birth_config,
            replay=failed_replay,
            build_directory=build_dir,
        )
        torch.cuda.synchronize()
        statuses = (
            failed_staging.death_status.cpu().tolist()
            + failed_staging.birth.status.cpu().tolist()
        )
        assert any(status != 0 for status in statuses), overflow
        for index, (before_tensor, after_tensor) in enumerate(
            zip(
                before,
                _authoritative(failed_bundle, failed_replay),
                strict=True,
            )
        ):
            torch.testing.assert_close(
                after_tensor,
                before_tensor,
                equal_nan=True,
                msg=f"R8 {overflow} rollback authoritative tensor {index}",
            )
