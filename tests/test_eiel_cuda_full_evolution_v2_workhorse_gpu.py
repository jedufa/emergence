"""Guards for the retired direct Workhorse GPU authority path.

Physical semantic execution is admitted only by the canonical Launchpad
provider. This module intentionally contains no environment-selected Python
loader and no direct CUDA test entrypoint.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from engine.perf.cuda_full_evolution_v2.workhorse.attempt_authority import (
    WorkhorseAttemptAuthorityError,
    resolve_injected_workhorse_attempt_authority_v2,
)
from engine.perf.cuda_full_evolution_v2.workhorse.probe import (
    FullEvolutionWorkhorseProbeError,
    run_workhorse_semantic_gate_v2,
    run_workhorse_shape_tuning_v2,
    run_workhorse_tuning_v2,
)


def test_retired_adapter_is_rejected_without_invoking_caller_code() -> None:
    invoked = False

    def hostile_adapter(_request):
        nonlocal invoked
        invoked = True
        raise AssertionError("retired authority adapter executed caller code")

    with pytest.raises(WorkhorseAttemptAuthorityError, match="retired"):
        resolve_injected_workhorse_attempt_authority_v2(
            hostile_adapter,
            request={},
        )
    assert invoked is False


def test_gpu_guard_has_no_dynamic_authority_loader() -> None:
    module = inspect.getmodule(test_gpu_guard_has_no_dynamic_authority_loader)
    assert module is not None
    tree = ast.parse(inspect.getsource(module))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "importlib" not in imported_modules
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "import_module"
        for node in ast.walk(tree)
    )
    assert not hasattr(module, "AUTHORITY_ADAPTER_ENV")


def test_all_direct_gpu_entrypoints_fail_before_caller_code() -> None:
    invoked = False

    def hostile_adapter(_request):
        nonlocal invoked
        invoked = True
        raise AssertionError("retired entrypoint invoked caller code")

    with pytest.raises(FullEvolutionWorkhorseProbeError, match="semantic GPU"):
        run_workhorse_semantic_gate_v2(
            "reference.so",
            "candidate.so",
            event_sink_root="events",
            checkpoint_root="checkpoints",
            semantic_evidence_root="semantic",
            receipt_output_path="receipt.json",
            attempt_authority_validator=hostile_adapter,
        )
    with pytest.raises(FullEvolutionWorkhorseProbeError, match="direct Workhorse tuning"):
        run_workhorse_tuning_v2(
            "candidate.so",
            authoritative_micro_receipt_path="micro.json",
            semantic_gate_receipt_path="semantic.json",
            event_sink_root="events",
            receipt_output_path="row.json",
            attempt_authority_validator=hostile_adapter,
        )
    with pytest.raises(FullEvolutionWorkhorseProbeError, match="multi-row"):
        run_workhorse_shape_tuning_v2(
            "graph.so",
            "fused.so",
            authoritative_micro_receipt_path="micro.json",
            semantic_gate_receipts={},
            event_sink_root="events",
            receipt_root="rows",
            output_path="aggregate.json",
            attempt_authority_validator=hostile_adapter,
        )
    assert invoked is False
