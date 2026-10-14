"""C6 external-boundary preparation tests; no provider launch is performed."""

from __future__ import annotations

import math

import pytest

from engine.tensor_world.c6_contract import TensorWorldC6RouteRequestV1
from engine.tensor_world.c6_external import (
    TensorWorldC6ControlKindV1,
    TensorWorldC6ControlRequestV1,
    TensorWorldC6ControlStateV1,
    TensorWorldC6MetricSurfaceV1,
    TensorWorldC6SafeBoundaryV1,
    apply_c6_control_oracle_v1,
    run_c6_cpu_boundary_rehearsal_v1,
)
from engine.tensor_world.contracts import TensorWorldContractError


def _request() -> TensorWorldC6RouteRequestV1:
    return TensorWorldC6RouteRequestV1(
        batch=48,
        grid_size=32,
        initial_population=256,
        slot_capacity=512,
        policy_family="fixed_mlp",
        mechanics=("patch", "jungle", "wdl", "eed", "observation", "fixed_mlp"),
        strict=True,
    )


def test_c6_control_rehearsal_is_explicitly_boundary_scoped() -> None:
    state = TensorWorldC6ControlStateV1()
    request = TensorWorldC6ControlRequestV1(
        sequence=1,
        control_id="pressure-1",
        kind=TensorWorldC6ControlKindV1.PRESSURE_UPDATE,
        boundary=TensorWorldC6SafeBoundaryV1.TICK_BOUNDARY,
        payload={"pressure": 0.5},
    )
    unchanged, rejected = apply_c6_control_oracle_v1(
        state,
        request,
        actual_boundary=TensorWorldC6SafeBoundaryV1.CHECKPOINT_BOUNDARY,
    )
    assert unchanged == state
    assert rejected.accepted is False
    changed, accepted = apply_c6_control_oracle_v1(
        state,
        request,
        actual_boundary=TensorWorldC6SafeBoundaryV1.TICK_BOUNDARY,
    )
    assert accepted.accepted is True
    assert accepted.applied is True
    assert changed.pressure == 0.5


def test_c6_control_sequence_and_reconfiguration_fail_closed() -> None:
    state = TensorWorldC6ControlStateV1()
    pause = TensorWorldC6ControlRequestV1(
        sequence=1,
        control_id="pause-1",
        kind=TensorWorldC6ControlKindV1.PAUSE,
        boundary=TensorWorldC6SafeBoundaryV1.TICK_BOUNDARY,
    )
    state, receipt = apply_c6_control_oracle_v1(
        state, pause, actual_boundary=TensorWorldC6SafeBoundaryV1.TICK_BOUNDARY
    )
    assert receipt.accepted is True
    duplicate_state, duplicate = apply_c6_control_oracle_v1(
        state, pause, actual_boundary=TensorWorldC6SafeBoundaryV1.TICK_BOUNDARY
    )
    assert duplicate_state == state
    assert duplicate.accepted is False

    reconfigure = TensorWorldC6ControlRequestV1(
        sequence=2,
        control_id="reconfigure-1",
        kind=TensorWorldC6ControlKindV1.RECONFIGURE,
        boundary=TensorWorldC6SafeBoundaryV1.TICK_BOUNDARY,
        payload={"cadence": 8},
    )
    unchanged, rejected = apply_c6_control_oracle_v1(
        state, reconfigure, actual_boundary=TensorWorldC6SafeBoundaryV1.TICK_BOUNDARY
    )
    assert unchanged == state
    assert rejected.accepted is False


def test_c6_metric_surface_rejects_nonfinite_values() -> None:
    with pytest.raises(TensorWorldContractError, match="finite"):
        TensorWorldC6MetricSurfaceV1(
            route_fingerprint="a" * 64,
            tick=1,
            metrics={"latency": math.inf},
        )


def test_c6_cpu_rehearsal_is_deterministic_and_never_admits_default_route() -> None:
    first = run_c6_cpu_boundary_rehearsal_v1(
        _request(),
        source_commit="source-current-c6-prep",
        evidence_root="/tmp/emergence-test/runtime/evidence_raw/tensor_world/c6-prep",
    )
    second = run_c6_cpu_boundary_rehearsal_v1(
        _request(),
        source_commit="source-current-c6-prep",
        evidence_root="/tmp/emergence-test/runtime/evidence_raw/tensor_world/c6-prep",
    )
    assert first.to_dict() == second.to_dict()
    assert first.fingerprint() == second.fingerprint()
    assert first.route_accepted is False
    assert first.control_boundary_rehearsal is True
    assert first.metrics_schema_rehearsal is True
    assert first.status == "PREPARATION_ONLY"
    assert first.gate_decision == "NO_GATE_DECISION"


def test_c6_rehearsal_cannot_encode_relative_route_metrics() -> None:
    request = _request()
    manifest = run_c6_cpu_boundary_rehearsal_v1(
        request,
        source_commit="source-current-c6-prep",
        evidence_root="/tmp/emergence-test/runtime/evidence_raw/tensor_world/c6-prep",
    )
    assert manifest.route_fingerprint != "0" * 64
