from tools.scripts.eiel_b48_wdl_jungle_eed_hapb_feasibility_v1 import (
    PolicyRefillTraceV1,
    decide_feasibility_v1,
    traffic_for_trace_v1,
)


def _trace(rows_per_cell: int) -> PolicyRefillTraceV1:
    rows = [[cell, slot] for cell in range(48) for slot in range(rows_per_cell)]
    return PolicyRefillTraceV1.from_mapping(
        {
            "schema_version": "eiel_hapb_policy_refill_trace_v1",
            "route_identity": "B48W32P256C512GPU-patch+wdl_delayed_hazard+jungle+genome_v2_ordinary_birth+genome_v2_mutation+recombination_profile+eed_resource_impact",
            "source_commit": "0123456789abcdef",
            "ticks_per_generation": 160,
            "rows_by_tick": [rows] * 160,
        }
    )


def test_pure_hapb_accounts_for_all_active_refills_at_p256_and_c512() -> None:
    p256 = traffic_for_trace_v1(_trace(256), representation="pure", tile_rows=512)
    c512 = traffic_for_trace_v1(_trace(512), representation="pure", tile_rows=512)

    assert p256["policy_bytes_per_generation"] == 78_572_421_120
    assert c512["policy_bytes_per_generation"] == 157_144_842_240
    assert p256["tile_count_per_generation"] == 24 * 160
    assert c512["tile_count_per_generation"] == 48 * 160


def test_hot_prefix_only_tiles_overflow_and_h1_gate_is_frozen() -> None:
    p256 = traffic_for_trace_v1(_trace(256), representation="p256_hot_prefix", tile_rows=4096)
    c512 = traffic_for_trace_v1(_trace(512), representation="p256_hot_prefix", tile_rows=4096)
    decision = decide_feasibility_v1(
        _trace(512),
        representation="p256_hot_prefix",
        tile_rows=4096,
        measured_seconds_per_generation=3_000.0,
        host_pressure_passed=True,
    )

    assert p256["policy_bytes_per_generation"] == 0
    assert c512["policy_bytes_per_generation"] == 78_572_421_120
    assert c512["tile_count_per_generation"] == 480
    assert decision["accepted"] is False
    assert decision["qualification_claim"] is False
