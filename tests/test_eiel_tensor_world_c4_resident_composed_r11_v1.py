"""Static contract tests for the resident R11 repair seam."""

import ast
from pathlib import Path
from types import SimpleNamespace

from engine.tensor_world.c4_resident_composed_r11 import (
    C4_RESIDENT_COMPOSED_R11_ADMITTED_CAPACITIES_V1,
    C4_RESIDENT_COMPOSED_R11_DECISION_V1,
    C4_RESIDENT_COMPOSED_R11_PRODUCTION_CAPACITY_V1,
    C4_RESIDENT_COMPOSED_R11_SCHEMA_VERSION_V1,
    _clone_resident_value_v1,
    _resident_tensor_pairs_v1,
    c4_resident_composed_r11_source_receipt_v1,
    execute_c4_resident_composed_r11_window_cuda_v1,
)
from engine.tensor_world.c4_fixture import make_c4_fixture_v1


ROOT = Path(__file__).resolve().parents[1]


def test_r11_composed_receipt_binds_authority_map_and_forbids_host_fallback() -> None:
    receipt = c4_resident_composed_r11_source_receipt_v1()
    assert receipt["schema_version"] == C4_RESIDENT_COMPOSED_R11_SCHEMA_VERSION_V1
    assert receipt["decision"] == C4_RESIDENT_COMPOSED_R11_DECISION_V1
    assert receipt["composition_ready"] is True
    assert receipt["execution_classification"] == (
        "B_FIXED_TOPOLOGY_CUDA_GRAPH_OWNER_WITH_DEVICE_TIME"
    )
    assert receipt["rollback_model"] == (
        "device_local_write_ahead_shadow_and_guarded_commit"
    )
    assert receipt["graph_window_ticks"] == 5
    assert receipt["graph_topology_receipt"] == (
        "raw_cuda_graph_retained_and_instantiated_at_capture_boundary"
    )
    assert receipt["python_c3_to_r9_fallback"] is False
    assert receipt["host_status_authority"] is False
    assert receipt["host_snapshot_rollback"] is False
    assert receipt["event_authority"].startswith("V4_event_ring_double_buffer_native")
    assert receipt["admitted_capacities"] == list(
        C4_RESIDENT_COMPOSED_R11_ADMITTED_CAPACITIES_V1
    )
    assert C4_RESIDENT_COMPOSED_R11_PRODUCTION_CAPACITY_V1 == 512
    assert receipt["production_geometry"] == "B48_W32_P256_C512"
    assert receipt["steady_state_host_boundary"]["graph_replays_per_window"] == 1
    assert receipt["steady_state_host_boundary"]["eager_device_control_launches"] == 0
    assert receipt["compatibility_window_wrapper"]["qualified"] is False
    assert receipt["shadow_policy"]["full_policy_clone"] is False
    assert receipt["authority_map"]["authority_count"] >= 14
    assert all(
        (ROOT / item["path"]).is_file()
        for item in receipt["sources"].values()
    )


def test_r11_composed_source_contains_no_tick_boundary_materialization() -> None:
    source = (ROOT / "engine/tensor_world/c4_resident_composed_r11.py").read_text(
        encoding="utf-8"
    )
    for forbidden in ("detach().clone()", ".tolist()", "torch.cuda.synchronize"):
        assert forbidden not in source


def test_r11_composed_execution_fails_closed_without_native_binding() -> None:
    # This test is source-bound only: importing the module must not silently
    # select the historical host-owned R10 loop.
    assert "host C3 -> R9 fallback is forbidden" in (
        ROOT / "engine/tensor_world/c4_resident_composed_r11.py"
    ).read_text(encoding="utf-8")


def test_r11_normal_tick_has_no_cpu_semantic_validation_or_tick_loop() -> None:
    source = (ROOT / "engine/tensor_world/c4_resident_composed_r11.py").read_text(
        encoding="utf-8"
    )
    start = source.index("def execute_c4_resident_composed_r11_cuda_v1(")
    end = source.index("def execute_c4_resident_composed_r11_window_cuda_v1(", start)
    execute = source[start:end]
    assert "owner.validate()" not in execute
    assert "bind_c4_identity_authority" not in execute
    assert "validate_pcg64" not in execute
    assert "validate_typed_continuation" not in execute
    assert "for " not in execute
    assert "torch.cuda.synchronize" not in execute
    assert "owner.assert_steady_state_ready()" in execute


def test_r11_captured_c2_config_storage_is_owner_retained() -> None:
    source = (ROOT / "engine/tensor_world/c4_resident_composed_r11.py").read_text(
        encoding="utf-8"
    )
    build_start = source.index("def _build_native_owner_v1(")
    window_start = source.index("def capture_c4_resident_composed_r11_window_v1(")
    bind_start = source.index("def bind_c4_resident_composed_r11_v1(")
    build = source[build_start:window_start]
    window = source[window_start:bind_start]
    assert build.count("_config_tensors_v1(") == 1
    assert "owner.c2_config_ints = config_ints" in build
    assert "owner.c2_config_floats = config_floats" in build
    assert "_config_tensors_v1(" not in window
    assert "config_ints = owner.c2_config_ints" in window
    assert "config_floats = owner.c2_config_floats" in window


def test_r11_first_error_latch_excludes_batch_peer_status() -> None:
    source = (ROOT / "engine/tensor_world/c4_resident_composed_r11.py").read_text(
        encoding="utf-8"
    )
    native = (
        ROOT / "engine/tensor_world/native/tensor_world_c4_resident_graph_r11.cu"
    ).read_text(encoding="utf-8")
    window_start = source.index("def capture_c4_resident_composed_r11_window_v1(")
    bind_start = source.index("def bind_c4_resident_composed_r11_v1(")
    window = source[window_start:bind_start]
    assert "owner.r11_extension.merge_window_control(" in window
    assert "primary = control.batch" in native
    assert "status != r6::kPeerFailed" in native
    assert "first_code = control.preflight_status[safe_world]" in native


def test_r11_event_ring_is_bound_to_the_native_transaction_abi() -> None:
    source = (ROOT / "engine/tensor_world/c4_resident_composed_r11.py").read_text(
        encoding="utf-8"
    )
    native = (
        ROOT / "engine/tensor_world/native/tensor_world_c4_resident_graph_r11.cu"
    ).read_text(encoding="utf-8")
    assert "_event_ring_tensors(shadow.event_ring)" in source
    assert "reset_local_event_cursor" in native
    assert "preflight_event_ring" in native
    assert "commit_event_ring" in native
    assert "_identity_staging_tensors_r11_v1(shadow_staging.identity)" in source
    assert "identity.size() == 6" in native
    assert "kIdentitySortedMonotonic" in native


def test_r11_window_is_one_graph_replay_with_device_failure_latch() -> None:
    source = (ROOT / "engine/tensor_world/c4_resident_composed_r11.py").read_text(
        encoding="utf-8"
    )
    assert "execute_c4_resident_composed_r11_window_cuda_v1" in source
    assert "capture_c4_resident_composed_r11_window_v1" in source
    assert "owner.window_graph.replay()" in source
    assert "first_error_world" in source
    assert "torch.cuda.CUDAGraph(keep_graph=True)" in source
    assert "window_graph.instantiate()" in source


def test_r11_qualified_window_api_is_exactly_one_replay_without_eager_writes() -> None:
    source = (ROOT / "engine/tensor_world/c4_resident_composed_r11.py").read_text(
        encoding="utf-8"
    )
    start = source.index("def execute_c4_resident_composed_r11_window_cuda_v1(")
    end = source.index(
        "def prepare_c4_resident_composed_r11_window_control_v1(", start
    )
    qualified = source[start:end]
    assert qualified.count("owner.window_graph.replay()") == 1
    function = ast.parse(qualified).body[0]
    calls = [ast.unparse(node.func) for node in ast.walk(function) if isinstance(node, ast.Call)]
    assert calls == ["owner.window_graph.replay"]
    assert not any(
        token in qualified for token in (".fill_(", ".zero_(", ".copy_(", "torch.")
    )

    prepare_start = end
    prepare_end = source.index(
        "def execute_c4_resident_composed_r11_window_compat_cuda_v1(",
        prepare_start,
    )
    prepare = source[prepare_start:prepare_end]
    assert ".generation.fill_(" in prepare
    assert ".requested_ticks.fill_(" in prepare
    assert ".global_status.zero_(" in prepare
    assert "window_graph.replay()" not in prepare


def test_r11_qualified_window_api_executes_one_fake_graph_launch() -> None:
    class FakeGraph:
        calls = 0

        def replay(self) -> None:
            self.calls += 1

    graph = FakeGraph()
    staging = object()
    owner = SimpleNamespace(window_graph=graph, shadow_staging=staging)
    assert execute_c4_resident_composed_r11_window_cuda_v1(owner) is staging
    assert graph.calls == 1


def test_r11_shadow_aliases_only_batch_guarded_packed_policy_payloads() -> None:
    bundle, _ = make_c4_fixture_v1(
        batch=1, capacity=8, active=4, width=8, seed=3
    )
    shadow = _clone_resident_value_v1(bundle)
    assert shadow.base.base.core.policy.packed_weights is (
        bundle.base.base.core.policy.packed_weights
    )
    assert shadow.base.base.policy.packed_weights is (
        bundle.base.base.policy.packed_weights
    )
    assert shadow.base.base.core.agents.energy is not (
        bundle.base.base.core.agents.energy
    )
    pairs = _resident_tensor_pairs_v1(bundle, shadow)
    paired_sources = {id(source) for _, source in pairs}
    assert id(bundle.base.base.core.policy.packed_weights) not in paired_sources
    assert id(bundle.base.base.policy.packed_weights) not in paired_sources
    assert id(bundle.base.base.core.agents.energy) in paired_sources

    native = (
        ROOT / "engine/tensor_world/native/tensor_world_c4_resident_birth_r2.cu"
    ).read_text(encoding="utf-8")
    inherit_start = native.index("__global__ void inherit_rows_kernel")
    inherit_end = native.index("__global__ void fill_rows_kernel", inherit_start)
    assert "!all_ok(stage, batch)" in native[inherit_start:inherit_end]
