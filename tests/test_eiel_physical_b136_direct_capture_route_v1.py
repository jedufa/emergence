"""Fail-closed provider route contract for physical tick-native B136 ACS."""

from __future__ import annotations

import ast
import ctypes
import ctypes.util
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch

import engine.campaigns.providers as providers
from engine.campaigns.providers import ProductionEngineLifecycleV1


def _execute_segments_tree() -> ast.Module:
    source = textwrap.dedent(
        inspect.getsource(ProductionEngineLifecycleV1._execute_segments)
    )
    return ast.parse(source)


def _call_names(node: ast.AST) -> tuple[str, ...]:
    names: list[str] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Name):
            names.append(child.func.id)
        elif isinstance(child.func, ast.Attribute):
            names.append(child.func.attr)
    return tuple(names)


def _if_nodes(tree: ast.AST, expression: str) -> list[ast.If]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and ast.unparse(node.test) == expression
    ]


def _physical_capture_branches(tree: ast.AST) -> list[ast.If]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "physical_batch_detached_capture_route" in ast.unparse(node.test)
        and "terminal_physical_capture_route" in ast.unparse(node.test)
    ]


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _is_descendant(
    node: ast.AST, ancestor: ast.AST, parents: dict[ast.AST, ast.AST]
) -> bool:
    current = node
    while current in parents:
        current = parents[current]
        if current is ancestor:
            return True
    return False


def _physical_route_assignment(tree: ast.Module) -> ast.Assign:
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "physical_batch_detached_capture_route"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1, (
        "the production route must have exactly one authoritative physical-B136 "
        "selector"
    )
    return assignments[0]


def _route_value(
    *, detached_acs: bool, profile_id: str, streamed: bool, cell_count: int
) -> bool:
    assignment = _physical_route_assignment(_execute_segments_tree())
    assert isinstance(assignment.value, ast.Call)
    assert isinstance(assignment.value.func, ast.Name)
    assert assignment.value.func.id == "bool"
    expression = ast.Expression(assignment.value.args[0])
    compiled = compile(expression, "<physical-b136-route>", "eval")
    request = SimpleNamespace(
        profile_id=profile_id,
        native_streamed_checkpoint=streamed,
        cell_ids=tuple(f"cell-{index}" for index in range(cell_count)),
    )
    return bool(
        eval(  # noqa: S307 - this evaluates the inspected production AST only.
            compiled,
            {"PRODUCTION_PROFILE_ID": providers.PRODUCTION_PROFILE_ID},
            {"tick_native_detached_acs": detached_acs, "request": request},
        )
    )


def _descendant_calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and (
            (isinstance(child.func, ast.Name) and child.func.id == name)
            or (isinstance(child.func, ast.Attribute) and child.func.attr == name)
        )
    ]


def test_physical_b136_route_selector_is_exact_and_fail_closed() -> None:
    """The production selector admits only native streamed physical B136 ACS."""

    production = providers.PRODUCTION_PROFILE_ID
    assert (
        _route_value(
            detached_acs=True,
            profile_id=production,
            streamed=True,
            cell_count=136,
        )
        is True
    )

    for detached_acs, profile_id, streamed, cell_count in (
        (False, production, True, 136),
        (True, "production_cpu", True, 136),
        (True, production, False, 136),
        (True, production, True, 135),
        (True, production, True, 137),
    ):
        assert (
            _route_value(
                detached_acs=detached_acs,
                profile_id=profile_id,
                streamed=streamed,
                cell_count=cell_count,
            )
            is False
        )


def test_physical_b136_capture_replaces_producer_settlement() -> None:
    """The physical branch captures once and cannot call the producer projector."""

    tree = _execute_segments_tree()
    settlement_branches = _physical_capture_branches(tree)
    assert len(settlement_branches) == 2, (
        "the provider must retain separate settlement and physical-capture gates"
    )
    settlement_branch = next(
        node
        for node in settlement_branches
        if any(
            isinstance(child, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "settlement"
                for target in child.targets
            )
            for child in node.body
        )
    )
    assert "build_direct_resident_physical_capture_v1" not in _call_names(
        settlement_branch
    )
    assert "projector" not in _call_names(
        ast.Module(body=settlement_branch.body, type_ignores=[])
    )
    assert "SimpleNamespace" in _call_names(settlement_branch)
    assert "object" in _call_names(settlement_branch)
    assert len(settlement_branch.orelse) > 0
    assert "projector" in _call_names(
        ast.Module(body=settlement_branch.orelse, type_ignores=[])
    )

    physical_capture_branch = next(
        node
        for node in settlement_branches
        if "build_direct_resident_physical_capture_v1" in _call_names(node)
    )
    capture_calls = _descendant_calls(
        physical_capture_branch, "build_direct_resident_physical_capture_v1"
    )
    assert len(capture_calls) == 1
    keyword_values = {
        keyword.arg: keyword.value for keyword in capture_calls[0].keywords
    }
    assert ast.unparse(keyword_values["cpu_request_staging_enable"]) == "False"
    assert ast.unparse(keyword_values["reporting_enabled"]) == (
        "bool(projector_arguments['reporting_enabled'])"
    )
    assert "projector" not in _call_names(physical_capture_branch)
    assert "_compact_segment_settlement_v1" not in _call_names(physical_capture_branch)


def test_physical_b136_capture_carries_terminal_metrics_to_detached_closeout() -> None:
    tree = _execute_segments_tree()
    physical_capture_branch = next(
        node
        for node in _physical_capture_branches(tree)
        if "build_direct_resident_physical_capture_v1" in _call_names(node)
    )
    assignments = [
        node
        for node in ast.walk(physical_capture_branch)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "cursor_capture"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    value = assignments[0].value
    assert isinstance(value, ast.Dict)
    compact_rows = next(
        item
        for key, item in zip(value.keys, value.values, strict=True)
        if isinstance(key, ast.Constant) and key.value == "compact_result_rows"
    )
    assert ast.unparse(compact_rows) == "()"
    assert "_validated_physical_compact_result_rows_v1" in _call_names(tree)


def test_physical_terminal_metric_rows_require_exact_ordered_cell_coverage() -> None:
    cell_ids = ("cell-0", "cell-1")
    rows = providers._validated_physical_compact_result_rows_v1(
        {
            "physical_compact_result_rows": [
                {"cell_id": cell_id, "metrics": {"population.alive": 1}}
                for cell_id in cell_ids
            ]
        },
        cell_ids=cell_ids,
    )
    assert [row["cell_id"] for row in rows] == list(cell_ids)

    for invalid in (
        [],
        [{"cell_id": "cell-1", "metrics": {}}] * 2,
        [{"cell_id": "cell-0", "metrics": {}}],
        [{"cell_id": "cell-0", "metrics": {}}, {"cell_id": "cell-1"}],
    ):
        with pytest.raises(
            providers.CampaignProviderError,
            match="physical world checkpoint metric",
        ):
            providers._validated_physical_compact_result_rows_v1(
                {"physical_compact_result_rows": invalid},
                cell_ids=cell_ids,
            )


def test_legacy_staging_and_source_cursor_mutation_are_outside_physical_route() -> None:
    """Physical B136 must not enter legacy staging or mutate a source owner."""

    tree = _execute_segments_tree()
    acs_setup = _if_nodes(tree, "request.acs_checkpoint_slots")
    assert len(acs_setup) == 1
    tick_native_setup = next(
        node
        for node in ast.walk(acs_setup[0])
        if isinstance(node, ast.If) and ast.unparse(node.test) == "tick_native"
    )
    assert "ACSNativeDetachedSlotsV1" in _call_names(tick_native_setup)
    assert "ACSCheckpointStaging" not in ast.unparse(
        ast.Module(body=tick_native_setup.body, type_ignores=[])
    )
    assert "ACSCheckpointStaging" in ast.unparse(
        ast.Module(body=tick_native_setup.orelse, type_ignores=[])
    )

    guarded = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "tick_native_detached_acs" in ast.unparse(node.test)
        and "projection_cursor" in ast.unparse(node.test)
        and "not physical_batch_detached_capture_route" in ast.unparse(node.test)
        and "rebind_detached_spool" in ast.unparse(node)
    ]
    assert len(guarded) == 1, (
        "source-owner cursor mutation must remain explicitly excluded for physical B136"
    )
    guard = guarded[0]
    guarded_source = ast.unparse(guard)
    assert "rebind_detached_spool" in guarded_source
    assert "release_source" in guarded_source

    parents = _parent_map(tree)
    release_calls = _descendant_calls(tree, "release_source")
    assert release_calls, "expected guarded source-owner release"
    assert all(_is_descendant(call, guard, parents) for call in release_calls)

    source = ast.unparse(tree)
    assert source.count("rebind_detached_spool") == 1
    assert guarded_source.count("rebind_detached_spool") == 1


def test_projector_call_is_only_in_nonphysical_settlement_else() -> None:
    """A future route edit cannot reintroduce 136 producer-side settlements."""

    tree = _execute_segments_tree()
    branch = next(
        node
        for node in _physical_capture_branches(tree)
        if any(
            isinstance(child, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "settlement"
                for target in child.targets
            )
            for child in node.body
        )
    )
    parents = _parent_map(tree)
    projector_calls = _descendant_calls(tree, "projector")
    assert len(projector_calls) == 1
    projector_call = projector_calls[0]
    assert _is_descendant(projector_call, branch, parents)
    assert branch.orelse
    assert _is_descendant(projector_call, branch.orelse[0], parents)


def test_synchronous_physical_capture_releases_before_host_finalization() -> None:
    """ACS0 persistence cannot begin while the resident GPU owner is live."""

    tree = _execute_segments_tree()
    release_branch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "terminal_physical_capture_route"
        and "finalize_direct_resident_physical_capture_v1" in _call_names(node)
    )
    source = ast.unparse(release_branch)
    assert source.index("immutable_source_projections = None") < source.index(
        "release_resident_owner(owner_to_close)"
    )
    assert source.index("release_resident_owner(owner_to_close)") < source.index(
        "finalize_direct_resident_physical_capture_v1(detached_capture)"
    )
    assert source.count("release_resident_owner(owner_to_close)") == 1
    assert "persistence began before GPU owner release" in source
    assert "persistence_started_after_gpu_release" in source
    assert "writer_started_perf_ns <= safe_release_perf_ns" in source
    assert "writer_finished_perf_ns <= writer_started_perf_ns" in source
    assert "gpu_release_to_durable_set_seconds" in source


def test_direct_route_cannot_reenter_legacy_bounded_owner_release() -> None:
    tree = _execute_segments_tree()
    guarded = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "projection_cursor is not None" in ast.unparse(node.test)
        and "not terminal_physical_capture_route" in ast.unparse(node.test)
        and "bounded terminal capture has no resident owner" in ast.unparse(node)
    ]
    assert len(guarded) == 1


def test_deliberate_restore_boundary_still_terminates_after_publication() -> None:
    source = ast.unparse(_execute_segments_tree())
    direct_finalize = source.index(
        "finalize_direct_resident_physical_capture_v1(detached_capture)"
    )
    interruption = source.index(
        "if segment_end_next_tick == deliberate_interruption_tick:",
        direct_finalize,
    )
    assert "deliberately_interrupted = True" in source[interruption:]
    assert "break" in source[interruption:]


def _owner_release_receipt(
    *,
    allocated: int = 0,
    reserved: int = 0,
    live_alias_count: int = 0,
    synchronized: bool = True,
):
    return providers.OwnerReleaseReceiptV1(
        schema_version=providers.OWNER_RELEASE_RECEIPT_SCHEMA_VERSION_V1,
        owner_identity="owner-1",
        final_next_absolute_tick=3359,
        release_synchronization_complete=synchronized,
        live_alias_count=live_alias_count,
        post_release_cuda_allocated_bytes=allocated,
        post_release_cuda_reserved_bytes=reserved,
        released_perf_ns=100,
        released_unix_ns=200,
    )


def _terminal_capture(**updates: object) -> dict[str, object]:
    capture: dict[str, object] = {
        "schema_version": "eiel_test_detached_capture_v1",
        "capture_path": "/tmp/eiel-test-terminal-capture",
        "destination": "/tmp/eiel-test-terminal-destination",
        "logical_bytes": 0,
    }
    capture.update(updates)
    return capture


def test_terminal_cuda_context_reset_rejects_live_owner_alias() -> None:
    with pytest.raises(
        providers.CampaignProviderError,
        match="alias-free owner release",
    ):
        providers._reset_terminal_cuda_primary_context_v1(
            execution_device="cuda:0",
            owner_release_receipt=_owner_release_receipt(live_alias_count=1),
            terminal_capture=_terminal_capture(),
            live_owner_count=0,
        )


def test_terminal_cuda_context_reset_rejects_tracked_live_owner() -> None:
    with pytest.raises(
        providers.CampaignProviderError,
        match="alias-free owner release",
    ):
        providers._reset_terminal_cuda_primary_context_v1(
            execution_device="cuda:0",
            owner_release_receipt=_owner_release_receipt(),
            terminal_capture=_terminal_capture(),
            live_owner_count=1,
        )


def test_terminal_cuda_context_reset_rejects_unsynchronized_release() -> None:
    with pytest.raises(
        providers.CampaignProviderError,
        match="alias-free owner release",
    ):
        providers._reset_terminal_cuda_primary_context_v1(
            execution_device="cuda:0",
            owner_release_receipt=_owner_release_receipt(synchronized=False),
            terminal_capture=_terminal_capture(),
            live_owner_count=0,
        )


def test_terminal_cuda_context_reset_emits_exact_release_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class Reset:
        argtypes = None
        restype = None

        def __call__(self) -> int:
            calls.append("cudaDeviceReset")
            return 0

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "memory_allocated",
        lambda *_args, **_kwargs: pytest.fail(
            "allocator queried after terminal context reset"
        ),
    )
    monkeypatch.setattr(
        torch.cuda,
        "memory_reserved",
        lambda *_args, **_kwargs: pytest.fail(
            "allocator queried after terminal context reset"
        ),
    )
    monkeypatch.setattr(ctypes.util, "find_library", lambda _name: "libcudart.so")
    monkeypatch.setattr(
        ctypes,
        "CDLL",
        lambda _library: SimpleNamespace(cudaDeviceReset=Reset()),
    )

    receipt = providers._reset_terminal_cuda_primary_context_v1(
        execution_device="cuda:0",
        owner_release_receipt=_owner_release_receipt(
            allocated=11_264,
            reserved=2_097_152,
        ),
        terminal_capture=_terminal_capture(),
        live_owner_count=0,
    )

    assert calls == ["cudaDeviceReset"]
    assert receipt["schema_version"] == (
        "eiel_terminal_cuda_primary_context_release_v1"
    )
    assert receipt["execution_device"] == "cuda:0"
    assert receipt["cuda_device_reset_result"] == 0
    assert receipt["pre_reset_cuda_allocator_residue"] == {
        "allocated_bytes": 11_264,
        "reserved_bytes": 2_097_152,
    }
    assert receipt["live_owner_count_before_reset"] == 0
    assert receipt["detached_capture_authority"]["tensor_free"] is True
    assert receipt["detached_capture_authority"]["callable_free"] is True
    assert receipt["post_reset_allocator_observation"] == (
        "not_queried_to_avoid_cuda_primary_context_reinitialization"
    )
    assert receipt["released_perf_ns"] > 100
    assert receipt["released_unix_ns"] > 200


def test_terminal_cuda_context_reset_rejects_tensor_bearing_capture() -> None:
    with pytest.raises(
        providers.CampaignProviderError,
        match="host-only capture",
    ):
        providers._reset_terminal_cuda_primary_context_v1(
            execution_device="cuda:0",
            owner_release_receipt=_owner_release_receipt(),
            terminal_capture=_terminal_capture(alias=torch.tensor([1])),
            live_owner_count=0,
        )


def test_terminal_cuda_context_reset_rejects_cuda_reset_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Reset:
        argtypes = None
        restype = None

        def __call__(self) -> int:
            return 1

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(ctypes.util, "find_library", lambda _name: "libcudart.so")
    monkeypatch.setattr(
        ctypes,
        "CDLL",
        lambda _library: SimpleNamespace(cudaDeviceReset=Reset()),
    )
    with pytest.raises(
        providers.CampaignProviderError,
        match="returned an error",
    ):
        providers._reset_terminal_cuda_primary_context_v1(
            execution_device="cuda:0",
            owner_release_receipt=_owner_release_receipt(),
            terminal_capture=_terminal_capture(),
            live_owner_count=0,
        )


def test_detached_terminal_release_covers_deliberate_restore_before_break() -> None:
    source = ast.unparse(_execute_segments_tree())
    reset_call = source.index("_reset_terminal_cuda_primary_context_v1(")
    deferred_submit = source.index(
        "acs_pipeline.submit_capture(acs_capture_token, deferred_terminal_capture)",
        reset_call,
    )
    interruption = source.index(
        "if segment_end_next_tick == deliberate_interruption_tick:",
        deferred_submit,
    )
    release_block = source[reset_call:interruption]
    assert reset_call < deferred_submit < interruption
    assert "segment_end_next_tick == request.total_ticks" in source[:reset_call]
    assert (
        "segment_end_next_tick == deliberate_interruption_tick" in source[:reset_call]
    )
    assert "context_release_receipt['released_perf_ns']" in release_block
    assert "context_release_receipt['released_unix_ns']" in release_block
