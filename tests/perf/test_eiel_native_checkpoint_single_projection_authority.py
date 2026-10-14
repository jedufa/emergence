from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
import torch

from native_runtime.checkpoint_digests import (
    checkpoint_source_projection_authority_manifest,
)
from native_runtime.lifecycle_events import empty_typed_lifecycle_event_tensors
from perf.cxxbackbone10_resident_state_owner import (
    migrate_native_checkpoint_projection_v1_to_v2,
)
from simulation import checkpoint_bundle_v2 as bundle_module
from simulation.checkpoint_bundle_v2 import (
    CheckpointStrictError,
    _checkpoint_diagnostic_dump_enabled,
    checkpoint_bundle_v2_load,
    checkpoint_bundle_v2_save,
)
from simulation.control_plane_pkg.util_json import (
    canonical_json_bytes,
    canonical_json_bytes_lossless,
)


AUTHORITY_FIELDS = (
    "authority_record_schema_version",
    "authority_source",
    "source_projection_sha256",
    "source_projection_schema_version",
    "source_generation",
    "source_tick",
    "source_phase",
    "source_authority_epoch",
    "source_action_cache_sha256",
    "source_action_cache_valid_slot_count",
    "source_action_cache_boundary_status",
    "source_rng_sha256",
)


def test_checkpoint_diagnostic_dumps_are_explicitly_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEBUG_DETERMINISM", raising=False)
    monkeypatch.delenv("EIEL_STATE_DUMP", raising=False)
    assert _checkpoint_diagnostic_dump_enabled() is False

    monkeypatch.setenv("DEBUG_DETERMINISM", "true")
    assert _checkpoint_diagnostic_dump_enabled() is True

    monkeypatch.delenv("DEBUG_DETERMINISM")
    monkeypatch.setenv("EIEL_STATE_DUMP", "1")
    assert _checkpoint_diagnostic_dump_enabled() is True


def _clone_projection(projection: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(projection)
    out["tensors"] = {
        name: tensor.detach().clone()
        for name, tensor in projection["tensors"].items()
    }
    return out


def _pcg_state(value: int) -> dict[str, Any]:
    return {
        "bit_generator": "PCG64",
        "state": {"state": int(value), "inc": 1442695040888963407},
        "has_uint32": 0,
        "uinteger": 0,
    }


def _projection(*, generation: int = 0, tick: int = 0) -> dict[str, Any]:
    capacity = 2
    env_state = _pcg_state(11)
    tensors: dict[str, torch.Tensor] = {
        "agent_grid": torch.tensor([[0, -1], [-1, -1]], dtype=torch.int32),
        "agent_ids_by_alive_order": torch.tensor([1], dtype=torch.int64),
        "slot_to_agent_id": torch.tensor([1, -1], dtype=torch.int64),
        "agent_id_to_slot": torch.tensor([-1, 0], dtype=torch.int32),
        "agent_id_to_slot_agent_ids": torch.tensor([1], dtype=torch.int64),
        "agent_id_to_slot_slots": torch.tensor([0], dtype=torch.int32),
        "slot_tokens": torch.tensor([10, 20], dtype=torch.int64),
        "slot_free_queue": torch.tensor([1], dtype=torch.int64),
        "alive_mask": torch.tensor([True, False], dtype=torch.bool),
        "pos_y": torch.tensor([0, 0], dtype=torch.int32),
        "pos_x": torch.tensor([0, 0], dtype=torch.int32),
        "energy": torch.tensor([13.5, 0.0], dtype=torch.float32),
        "health": torch.tensor([1.0, 0.0], dtype=torch.float32),
        "injury": torch.zeros((capacity,), dtype=torch.float32),
        "age": torch.tensor([2, 0], dtype=torch.int32),
        "drives": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 0.0, 0.0]]),
        "hmem": torch.empty((capacity, 0), dtype=torch.float32),
        "mlp_pack": torch.tensor([[0.5], [0.0]], dtype=torch.float32),
        "policy_attack_bias": torch.zeros((capacity,), dtype=torch.float32),
        "policy_action_temperature": torch.ones((capacity,), dtype=torch.float32),
        "policy_drive_logit_gain": torch.ones((capacity,), dtype=torch.float32),
        "policy_use_gru": torch.zeros((capacity,), dtype=torch.bool),
        "action_cache_actions": torch.tensor(
            [[1, 2, 3], [-1, -1, -1]],
            dtype=torch.int32,
        ),
        "action_cache_token": torch.tensor([10, -1], dtype=torch.int64),
        "action_cache_cursor": torch.tensor([1, -1], dtype=torch.int32),
        "action_cache_next": torch.tensor([2, -1], dtype=torch.int32),
        "event_tape_type": torch.empty((0,), dtype=torch.int32),
        "event_tape_actor_id": torch.empty((0,), dtype=torch.int64),
        "event_tape_target_id": torch.empty((0,), dtype=torch.int64),
        "event_tape_value": torch.empty((0,), dtype=torch.float32),
        "rng_manager_mutation_torch_cpu_state": torch.tensor(
            [7, 8, 9],
            dtype=torch.uint8,
        ),
    }
    tensors.update(
        {
            f"typed_lifecycle_event_{name}": tensor
            for name, tensor in empty_typed_lifecycle_event_tensors().items()
        }
    )
    phase = "generation_boundary" if tick == 0 else "partial_generation"
    metadata = {
        "world": "flat",
        "generation": int(generation),
        "tick": int(tick),
        "configured_steps_per_generation": 4,
        "policy_tick_stride": 2,
        "alive_count": 1,
        "population_next_id": 2,
        "agent_grid_phase_contract_schema_version": "eiel_agent_grid_phase_contract_v1",
        "agent_grid_runner_phase": phase,
        "agent_grid_authority_epoch": f"generation:{generation}:tick:{tick}",
        "agent_grid_preserve_exact_stored": True,
        "event_tape_active_count": 0,
        "event_tape_cursor": 0,
        "typed_lifecycle_event_schema_version": "eiel_typed_lifecycle_event_v1",
        "typed_lifecycle_event_active_count": 0,
        "typed_lifecycle_event_source": "test_fixture",
        "native_runtime_numpy_env_state": copy.deepcopy(env_state),
        "native_runtime_numpy_runner_state": _pcg_state(22),
        "native_runtime_numpy_policy_state": copy.deepcopy(env_state),
        "native_runtime_env_policy_numpy_alias": True,
        "native_runtime_numpy_population_state": _pcg_state(33),
        "native_runtime_python_random_state": {"version": 3, "state": [1, 2, 3]},
        "native_runtime_rng_manager_stream_indices": {
            "policy": {"py": 1, "np": 2, "torch_cpu": 3, "torch_cuda": {}},
            "mutation": {"py": 4, "np": 5, "torch_cpu": 6, "torch_cuda": {}},
        },
    }
    return {
        "schema_version": "cxxbackbone10_checkpoint_projection_v1",
        "metadata": metadata,
        "tensors": tensors,
        "tensor_manifest": {
            name: {
                "dtype": str(tensor.dtype),
                "shape": [int(dim) for dim in tensor.shape],
                "device": str(tensor.device),
            }
            for name, tensor in tensors.items()
        },
        "restore_count": 0,
        "generation_calls": 0,
        "event_cursor": 0,
        "runtime_event_cursor": 0,
        "last_applied_actions": 0,
        "last_moved": 0,
        "last_gathered": 0,
        "last_policy_refill_calls": 0,
        "last_policy_refill_rows": 0,
        "pending_generation_free_agent_ids": [],
        "pending_generation_free_slots": [],
        "last_world": "flat",
    }


class _ProjectionEngine:
    def __init__(
        self,
        projection: dict[str, Any],
        *,
        on_export: Callable[[], None] | None = None,
    ) -> None:
        self._projection = _clone_projection(projection)
        self._on_export = on_export
        self.world = "flat"
        self.capability_mask = {"flat": True, "patch": False, "churn": True}
        self.counters = SimpleNamespace(
            to_dict=lambda: {
                "engine_constructions": 1,
                "normal_python_agent_traversals": 0,
                "normal_apply_owner_checkpoint_to_runner_calls": 0,
            }
        )

    def export_checkpoint_projection(self) -> dict[str, Any]:
        if self._on_export is not None:
            self._on_export()
        return _clone_projection(self._projection)


def _sim(projection: dict[str, Any], *, on_export: Callable[[], None] | None = None) -> Any:
    return SimpleNamespace(
        _native_eiel_engine=_ProjectionEngine(projection, on_export=on_export),
        sim_cfg=SimpleNamespace(
            branch_id="test",
            campaign_id="single_projection",
            structural_live_effect_scale=1.0,
        ),
        python_population_marker={"energy": -99.0, "policy": "stale"},
        logger=None,
    )


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _write_lossless(path: Path, value: Any) -> None:
    payload_bytes = canonical_json_bytes_lossless(value)
    if path.name.endswith(".gz"):
        path.write_bytes(gzip.compress(payload_bytes, mtime=0))
    else:
        path.write_bytes(payload_bytes)


def _read(path: Path) -> Any:
    if path.name.endswith(".gz"):
        return json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def _native_sidecar_path(bundle: Path) -> Path:
    path = bundle / bundle_module.NATIVE_RUNTIME_CHECKPOINT_PROJECTION_GZIP_FILENAME
    if path.exists():
        return path
    return bundle / bundle_module.NATIVE_RUNTIME_CHECKPOINT_PROJECTION_FILENAME


def _refresh_resident_hash(bundle: Path) -> None:
    resident_path = bundle / "resident_runtime_state_v2.json"
    resident = _read(resident_path)
    resident["resident_payload_sha256"] = hashlib.sha256(
        canonical_json_bytes_lossless(
            {key: value for key, value in resident.items() if key != "resident_payload_sha256"}
        )
    ).hexdigest()
    _write_lossless(resident_path, resident)
    metadata_path = bundle / "metadata.json"
    metadata = _read(metadata_path)
    metadata["resident_runtime_state_hash"] = hashlib.sha256(
        canonical_json_bytes_lossless(resident)
    ).hexdigest()
    _write_lossless(metadata_path, metadata)


def test_native_save_captures_once_and_stamps_every_derived_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    projection = _projection()
    sim = _sim(projection, on_export=lambda: calls.append("export"))
    python_before = copy.deepcopy(sim.python_population_marker)
    monkeypatch.setattr(
        bundle_module,
        "_flush_transient_caches",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("native save must not flush Python state")
        ),
    )
    monkeypatch.setattr(
        bundle_module,
        "_canonicalize_runner_drive_arrays",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("native save must not canonicalize Python state")
        ),
    )

    bundle = tmp_path / "bundle"
    checkpoint_bundle_v2_save(sim, str(bundle), generation_index=0, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(str(bundle), strict=True)

    assert calls == ["export"]
    assert sim.python_population_marker == python_before
    assert not hasattr(sim, "_last_checkpoint_perf")
    assert payload["metadata"]["metrics_state_hash"] == hashlib.sha256(
        canonical_json_bytes(payload["metrics_state"])
    ).hexdigest()
    authority = checkpoint_source_projection_authority_manifest(
        migrate_native_checkpoint_projection_v1_to_v2(projection)
    )
    authority["authority_source"] = bundle_module.NATIVE_SINGLE_PROJECTION_AUTHORITY
    views = {
        "metadata": payload["metadata"],
        "sidecar": payload["native_runtime_checkpoint_projection"],
        "resident": payload["resident_runtime_state_v2"],
        "world": payload["world"],
        "population": payload["population"],
        "rng": payload["rng_snapshot_v2"],
        "runner": payload["runner_state"],
        "control": payload["control_plane"],
        "metrics": payload["metrics_state"],
        "weights": payload["metadata"]["weights_info"],
        "coherence": payload["native_checkpoint_bundle_coherence_report"],
        "complete": payload["bundle_complete"],
    }
    resident = payload["resident_runtime_state_v2"]
    for name in (
        "environment_state",
        "generation_counters",
        "population_transaction_tape",
        "policy_execution_plan",
    ):
        views[f"resident.{name}"] = resident[name]
    for view_name, view in views.items():
        for field in AUTHORITY_FIELDS:
            assert view[field] == authority[field], (view_name, field)

    plan = resident["policy_execution_plan"]
    source_cache = payload["native_checkpoint_bundle_coherence_report"]["action_cache"]
    assert plan["action_cache_digest"] == source_cache["digest"]
    assert plan["action_cache_valid_slot_count"] == 1
    assert plan["action_cache_valid_slots"] == [0]
    assert plan["action_cache_boundary_status"] == "captured"
    assert plan["action_cache_validity"]["sha256"] == source_cache["validity"]["sha256"]
    assert set(resident["arrays"]) >= {
        "action_cache_actions",
        "action_cache_token",
        "action_cache_cursor",
        "action_cache_next",
    }


def test_repeated_save_and_interrupted_previous_recovery_are_idempotent(tmp_path: Path) -> None:
    projection = _projection()
    sim = _sim(projection)
    bundle = tmp_path / "bundle"

    checkpoint_bundle_v2_save(sim, str(bundle), generation_index=0, strict_replay_mode=True)
    first = _tree_bytes(bundle)
    checkpoint_bundle_v2_save(sim, str(bundle), generation_index=0, strict_replay_mode=True)
    assert _tree_bytes(bundle) == first
    assert not Path(f"{bundle}.tmp").exists()
    assert not Path(f"{bundle}.previous").exists()

    os.replace(bundle, Path(f"{bundle}.previous"))
    stale_tmp = Path(f"{bundle}.tmp")
    stale_tmp.mkdir()
    (stale_tmp / "partial").write_text("not complete", encoding="utf-8")
    checkpoint_bundle_v2_save(sim, str(bundle), generation_index=0, strict_replay_mode=True)

    assert _tree_bytes(bundle) == first
    assert not stale_tmp.exists()
    assert not Path(f"{bundle}.previous").exists()
    assert checkpoint_bundle_v2_load(str(bundle), strict=True)["bundle_complete"]["status"] == "complete"


def test_failed_serialization_preserves_final_bundle_and_cleans_transaction_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sim = _sim(_projection())
    bundle = tmp_path / "bundle"
    checkpoint_bundle_v2_save(sim, str(bundle), generation_index=0, strict_replay_mode=True)
    before = _tree_bytes(bundle)
    previous = Path(f"{bundle}.previous")
    previous.mkdir()
    (previous / "stale").write_text("stale", encoding="utf-8")
    original_write = bundle_module._write_canonical_json_lossless

    def _failing_write(path: str, value: Any) -> None:
        if path.endswith("population.json"):
            raise OSError("deliberate serialization failure")
        original_write(path, value)

    monkeypatch.setattr(bundle_module, "_write_canonical_json_lossless", _failing_write)
    with pytest.raises(OSError, match="deliberate serialization failure"):
        checkpoint_bundle_v2_save(sim, str(bundle), generation_index=0, strict_replay_mode=True)

    assert _tree_bytes(bundle) == before
    assert not Path(f"{bundle}.tmp").exists()
    assert not previous.exists()
    checkpoint_bundle_v2_load(str(bundle), strict=True)


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda projection: projection["tensors"].pop("action_cache_actions"),
            "missing authoritative checkpoint tensor fields",
        ),
        (
            lambda projection: projection["metadata"].pop("native_runtime_numpy_runner_state"),
            "missing authoritative checkpoint metadata fields",
        ),
        (
            lambda projection: projection["metadata"].update(
                {
                    "generation": 1,
                    "agent_grid_authority_epoch": "generation:1:tick:0",
                }
            ),
            "generation contradiction",
        ),
        (
            lambda projection: projection["metadata"].__setitem__("alive_count", 2),
            "alive_count/alive_mask contradiction",
        ),
        (
            lambda projection: projection["metadata"].__setitem__(
                "agent_grid_runner_phase", "partial_generation"
            ),
            "phase contradiction",
        ),
        (
            lambda projection: projection["metadata"].__setitem__(
                "agent_grid_authority_epoch", "generation:0:tick:1"
            ),
            "epoch contradiction",
        ),
    ],
)
def test_incomplete_or_contradictory_export_fails_before_finalization(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], Any],
    error: str,
) -> None:
    projection = _projection()
    mutate(projection)
    sim = _sim(projection)
    bundle = tmp_path / "bundle"

    with pytest.raises(CheckpointStrictError, match=error):
        checkpoint_bundle_v2_save(sim, str(bundle), generation_index=0, strict_replay_mode=True)

    assert not bundle.exists()
    assert not Path(f"{bundle}.tmp").exists()
    assert not Path(f"{bundle}.previous").exists()


def _corrupt_source_digest(bundle: Path) -> None:
    sidecar_path = _native_sidecar_path(bundle)
    sidecar = _read(sidecar_path)
    sidecar["source_projection_sha256"] = "0" * 64
    _write_lossless(sidecar_path, sidecar)
    metadata_path = bundle / "metadata.json"
    metadata = _read(metadata_path)
    metadata["native_runtime_checkpoint_hash"] = hashlib.sha256(
        canonical_json_bytes_lossless(sidecar)
    ).hexdigest()
    _write_lossless(metadata_path, metadata)


def _corrupt_resident(bundle: Path) -> None:
    path = bundle / "resident_runtime_state_v2.json"
    resident = _read(path)
    resident["arrays"]["energy"]["sha256"] = "1" * 64
    _write_lossless(path, resident)
    _refresh_resident_hash(bundle)


def _corrupt_rng(bundle: Path) -> None:
    path = bundle / "rng_snapshot_v2.json"
    rng = _read(path)
    rng["source_rng_sha256"] = "2" * 64
    _write_lossless(path, rng)
    metadata_path = bundle / "metadata.json"
    metadata = _read(metadata_path)
    metadata["rng_snapshot_hash"] = hashlib.sha256(canonical_json_bytes(rng)).hexdigest()
    _write_lossless(metadata_path, metadata)


def _corrupt_cache(bundle: Path) -> None:
    path = bundle / "resident_runtime_state_v2.json"
    resident = _read(path)
    resident["policy_execution_plan"]["action_cache_boundary_status"] = "flushed_before_checkpoint"
    _write_lossless(path, resident)
    _refresh_resident_hash(bundle)


def _corrupt_phase(bundle: Path) -> None:
    path = bundle / "world.json"
    world = _read(path)
    world["source_phase"] = "partial_generation"
    _write_lossless(path, world)
    metadata_path = bundle / "metadata.json"
    metadata = _read(metadata_path)
    metadata["world_hash"] = hashlib.sha256(canonical_json_bytes(world)).hexdigest()
    _write_lossless(metadata_path, metadata)


def _corrupt_epoch(bundle: Path) -> None:
    path = bundle / "metrics_state.json"
    metrics = _read(path)
    metrics["source_authority_epoch"] = "generation:0:tick:1"
    _write_lossless(path, metrics)


@pytest.mark.parametrize(
    ("corrupt", "error"),
    [
        (_corrupt_source_digest, "source projection digest mismatch"),
        (_corrupt_resident, "coherence verification failed"),
        (_corrupt_rng, "coherence verification failed"),
        (_corrupt_cache, "coherence verification failed"),
        (_corrupt_phase, "coherence verification failed"),
        (_corrupt_epoch, "metrics_state hash mismatch"),
    ],
)
def test_strict_load_rejects_derived_authority_contradictions(
    tmp_path: Path,
    corrupt: Callable[[Path], None],
    error: str,
) -> None:
    sim = _sim(_projection())
    bundle = tmp_path / "bundle"
    checkpoint_bundle_v2_save(sim, str(bundle), generation_index=0, strict_replay_mode=True)
    corrupt(bundle)

    with pytest.raises(CheckpointStrictError, match=error):
        checkpoint_bundle_v2_load(str(bundle), strict=True)
