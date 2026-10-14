from __future__ import annotations

from engine.campaigns.providers import ResidentBatchRequestV1
from engine.campaigns.tick_native_control import TickNativeControlV1
from tools.scripts.eiel_b48_wdl_jungle_eed_hapb_h4_continuation_v1 import (
    _h4_private_adapter_bounds_v1,
    _h4_tick_dispatch_spans_v1,
)


def _control() -> TickNativeControlV1:
    return TickNativeControlV1.from_execution(
        {
            "tick_native_control": {
                "time_authority": "absolute_tick_v1",
                "total_ticks": 960,
                "next_absolute_tick": 0,
                "resident_owner_span_ticks": 160,
                "native_dispatch_window_ticks": 160,
                "refill_chunk_ticks": 160,
                "source_window_ticks": 160,
                "event_window_capacity": 65536,
                "checkpoint_ticks": [479, 960],
            }
        }
    )


def test_non_lattice_checkpoint_uses_absolute_tick_spans() -> None:
    control = _control()
    assert _h4_private_adapter_bounds_v1(
        start_tick=0, end_tick=479, chunk_ticks=160
    ) == (0, 3)
    assert _h4_private_adapter_bounds_v1(
        start_tick=479, end_tick=960, chunk_ticks=160
    ) == (2, 6)
    assert _h4_tick_dispatch_spans_v1(
        control, start_tick=0, end_tick=479
    ) == (160, 160, 159)
    assert _h4_tick_dispatch_spans_v1(
        control, start_tick=479, end_tick=960
    ) == (160, 160, 160, 1)


def test_tick_native_request_payload_needs_no_generation_aliases() -> None:
    control = _control()
    request = ResidentBatchRequestV1(
        profile_id="eiel_full_evolution_v1",
        batch_id="b48",
        cell_ids=tuple(f"cell-{index:02d}" for index in range(48)),
        checkpoint_milestones=(),
        parameter_arrays=(),
        seed_arrays=(),
        total_ticks=control.total_ticks,
        start_next_tick=control.start_next_tick,
        checkpoint_ticks=control.resolved_checkpoint_ticks,
        internal_lifecycle_interval_ticks=control.refill_chunk_ticks,
        tick_native_control=control.payload(),
    )
    payload = request.payload()
    assert payload["time_authority"] == "absolute_tick_v1"
    assert payload["total_ticks"] == 960
    assert payload["checkpoint_ticks"] == [479, 960]
    assert "generations" not in payload
    assert "ticks_per_generation" not in payload
