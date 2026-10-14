from __future__ import annotations

from pathlib import Path

from tools.scripts.eiel_b48_wdl_jungle_eed_hapb_h4_continuation_v1 import (
    _detach_host_snapshot_v1,
    persist_hapb_midpoint_via_acs_v1,
)


class _FakeTensor:
    def __init__(self) -> None:
        self.cpu_calls = 0

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        self.cpu_calls += 1
        return self

    def contiguous(self) -> "_FakeTensor":
        return self


def test_detach_host_snapshot_calls_tensor_cpu_as_method() -> None:
    tensor = _FakeTensor()
    out = _detach_host_snapshot_v1({"bank": tensor})
    assert out["bank"] is tensor
    assert tensor.cpu_calls == 1


def test_hapb_midpoint_acs_persist_writes_through_two_slot_writer(tmp_path: Path) -> None:
    target = tmp_path / "hapb_serialized_midpoint.pt"
    envelope = {"schema": "test-envelope", "cells": 48}

    seen: list[object] = []

    def save(payload: object, path: Path) -> None:
        seen.append(payload)
        assert payload is not envelope
        assert payload == envelope
        path.write_bytes(b"hapb-midpoint-bytes")

    result = persist_hapb_midpoint_via_acs_v1(
        envelope,
        target,
        slots=2,
        compute_frontier=3,
        save=save,
    )

    assert target.read_bytes() == b"hapb-midpoint-bytes"
    assert result["bytes"] == len(b"hapb-midpoint-bytes")
    assert result["host_snapshot_detached"] is True
    assert result["acs_checkpoint_slots"] == 2
    assert result["acs_compute_frontier"] == 3
    assert result["sha256"]
    telemetry = result["acs_telemetry"]
    assert telemetry["slots"] == 2
    assert telemetry["durable_checkpoint_frontier"] == 3
    assert telemetry["frontier_lag"] == 0
    assert telemetry["queue_occupancy"] == 0
