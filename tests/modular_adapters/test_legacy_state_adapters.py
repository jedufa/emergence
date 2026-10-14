from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from eiel_composition.adapters import (
    LegacyCompositeCheckpointAdapter,
    LegacyRNGAuthorityAdapter,
)
from emergence_contracts import (
    ArtifactId,
    BranchId,
    ComponentId,
    ContentDigest,
    LeaseId,
    RNGLeaseDescriptorV1,
    RunArtifactRefV1,
    RunCursorV1,
    RunId,
    SchemaVersion,
)
from engine.perf.composite_runtime_state_v1 import (
    compose_composite_runtime_state_v1 as authoritative_compose,
)
from engine.qrr.composite_runtime_state_v1 import (
    compose_composite_runtime_state_v1 as qrr_compatibility_compose,
)
from simulation.checkpoint_bundle_v2 import (
    checkpoint_bundle_v2_apply,
    checkpoint_bundle_v2_load,
    checkpoint_bundle_v2_save,
)
from simulation.rng_manager import RNGManager
from perf.cxxbackbone10_resident_state_owner import projection_digest
from tests.evolvability.phase_e.test_e1_metrics_engine import (
    _capture_main_fp,
    _make_sim,
)


def _bundle(tmp_path: Path) -> tuple[object, Path]:
    sim = _make_sim(tmp_path / "run", seed=123, generations=1)
    sim.run(start_gen=0, skip_init=False)
    path = tmp_path / "bundle_roundtrip"
    checkpoint_bundle_v2_save(
        sim, str(path), generation_index=1, strict_replay_mode=True
    )
    return sim, path


def _assert_nested_equal(left: object, right: object) -> None:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        assert np.array_equal(np.asarray(left), np.asarray(right))
        return
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        assert torch.equal(torch.as_tensor(left), torch.as_tensor(right))
        return
    if isinstance(left, dict) and isinstance(right, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
        return
    assert left == right


def test_composite_view_maps_every_existing_bundle_file_without_new_state(tmp_path: Path) -> None:
    _sim, bundle = _bundle(tmp_path)
    adapter = LegacyCompositeCheckpointAdapter(bundle)
    direct = checkpoint_bundle_v2_load(str(bundle), strict=True)

    view = adapter.describe(
        run_id=RunId("run-sentinel"),
        branch_id=BranchId("default"),
        run_cursor=RunCursorV1(total_ticks=160, next_tick=80),
        assembly_identity=ContentDigest("sha256", "1" * 64),
    )

    _assert_nested_equal(adapter.load(strict=True), direct)
    represented = {
        Path(view.runtime_state_ref.locator).relative_to(bundle).as_posix(),
        Path(view.rng_root_snapshot_ref.locator).relative_to(bundle).as_posix(),
        *(Path(row.payload_ref.locator).relative_to(bundle).as_posix() for row in view.participants),
    }
    if view.backend_state_ref is not None:
        represented.add(Path(view.backend_state_ref.locator).relative_to(bundle).as_posix())
    # control_plane.json is held as the canonical control-position payload.
    expected = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() and path.name != "control_plane.json"
    }
    assert represented == expected
    assert {str(row.descriptor.module_id) for row in view.participants} == {"eiel"}
    assert view.compatibility_metadata.to_value()["state_ownership"] == (
        "current_checkpoint_bundle_v2_integrated_owner"
    )


def test_delegated_load_and_apply_match_direct_bundle_v2(tmp_path: Path) -> None:
    _source, bundle = _bundle(tmp_path)
    payload = checkpoint_bundle_v2_load(str(bundle), strict=True)
    direct = _make_sim(tmp_path / "direct", seed=123, generations=1)
    wrapped = _make_sim(tmp_path / "wrapped", seed=123, generations=1)

    checkpoint_bundle_v2_apply(direct, payload, restore_rng=True)
    LegacyCompositeCheckpointAdapter(bundle).apply(wrapped, restore_rng=True)

    _assert_nested_equal(_capture_main_fp(direct), _capture_main_fp(wrapped))


def _rng_ref(tmp_path: Path, snapshot: dict[str, object]) -> RunArtifactRefV1:
    path = tmp_path / "rng_snapshot_v2.json"
    data = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(data)
    return RunArtifactRefV1(
        artifact_id=ArtifactId("rng-snapshot-sentinel"),
        artifact_kind="rng_snapshot",
        artifact_schema_version=SchemaVersion("rng_snapshot_v2"),
        locator=str(path),
        content_digest=ContentDigest("sha256", hashlib.sha256(data).hexdigest()),
        size_bytes=len(data),
    )


def test_rng_adapter_construction_and_lease_consume_zero_draws(tmp_path: Path) -> None:
    manager = RNGManager(777)
    before_indices = manager.indices_snapshot()
    snapshot = manager.capture_snapshot(generation_index=0)
    ref = _rng_ref(tmp_path, snapshot)
    adapter = LegacyRNGAuthorityAdapter(manager, root_snapshot_ref=ref)
    request = RNGLeaseDescriptorV1(
        lease_id=LeaseId("policy-py-view"),
        owner_component_id=ComponentId("controller.fixed_mlp.v1"),
        semantic_domain="policy",
        stream_identity="policy.py",
        draw_index=before_indices.get("policy.py", 0),
        budget_contract_ref=None,
        device_scope="cpu",
    )

    lease = adapter.lease(request)

    assert manager.indices_snapshot() == before_indices
    assert lease.index_snapshot() == before_indices.get("policy.py", 0)
    assert lease.tracked_handle() is manager.py_rng("policy")
    assert adapter.root_snapshot_ref() == ref


def test_rng_adapter_restore_matches_current_fresh_owner_restore(tmp_path: Path) -> None:
    source = RNGManager(31337)
    source.policy_py.random()
    source.mutation_np.random(3)
    snapshot = source.capture_snapshot(generation_index=2)
    ref = _rng_ref(tmp_path, snapshot)
    direct = RNGManager(1)
    wrapped = RNGManager(1)

    direct.apply_snapshot_dict(snapshot)
    LegacyRNGAuthorityAdapter(wrapped, root_snapshot_ref=ref).restore_root_snapshot(ref)

    assert wrapped.indices_snapshot() == direct.indices_snapshot()
    assert wrapped.capture_snapshot(generation_index=2) == direct.capture_snapshot(
        generation_index=2
    )


def test_qrr_is_a_compatibility_consumer_and_perf_no_longer_imports_qrr() -> None:
    assert qrr_compatibility_compose is not authoritative_compose
    path = Path("engine/perf/cuda_architecture_b_direct_resident_wdl_jungle_v1.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "engine.qrr.composite_runtime_state_v1" not in imports
    assert "engine.perf.composite_runtime_state_v1" in imports


def test_authoritative_projector_wire_payload_is_unchanged_through_qrr_export() -> None:
    base = {
        "metadata": {"cell_id": "cell-1", "capability_mask": {"jungle": True}},
        "tensors": {"patch": torch.tensor([1], dtype=torch.int64)},
    }
    kwargs = {
        "base_component_id": "world.patch.v1",
        "component_tensors": {
            "world.overlay.jungle.v1": {
                "jungle": torch.tensor([2], dtype=torch.int64)
            }
        },
        "execution_markers": {"wdl_jungle_native_owner": True},
        "expected_cell_id": "cell-1",
    }

    direct = authoritative_compose(
        base, projection_digest_fn=projection_digest, **kwargs
    )
    compatibility = qrr_compatibility_compose(base, **kwargs)

    assert direct.projection_sha256 == compatibility.projection_sha256
    assert direct.component_ids == compatibility.component_ids
    assert direct.tensor_owners == compatibility.tensor_owners
    assert direct.wire_projection().keys() == compatibility.wire_projection().keys()
