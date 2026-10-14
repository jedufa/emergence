"""Source-bound write-set checks for the Phase-C single-shadow arm."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GRAPH = (ROOT / "engine/tensor_world/native/tensor_world_c4_resident_graph_r11.cu").read_text(
    encoding="utf-8"
)
COMPOSED = (ROOT / "engine/tensor_world/c4_resident_composed_r11.py").read_text(
    encoding="utf-8"
)
POPULATION = (
    ROOT / "engine/tensor_world/native/tensor_world_c4_resident_population_r5.cu"
).read_text(encoding="utf-8")


def _body(source: str, signature: str, next_signature: str) -> str:
    start = source.index(signature)
    end = source.index(next_signature, start)
    return source[start:end]


def test_c3_uses_outer_shadow_only_behind_r11_compile_fence() -> None:
    assert "#define EIEL_C4_PHASE_C_SINGLE_SHADOW 1" in GRAPH
    assert "prepare_post_death_outer" in GRAPH
    assert "commit_deaths_outer" in GRAPH
    assert "prepare_post_death_outer" in POPULATION
    assert "commit_deaths_outer" in POPULATION
    assert "r5::prepare_post_death_shadow" in GRAPH
    assert "r5::commit_deaths" in GRAPH


def test_c3_selects_complete_death_ledger_before_mutating_outer_shadow() -> None:
    body = _body(
        POPULATION,
        "__global__ void prepare_post_death_outer(",
        "__global__ void merge_all_status(",
    )
    selected = body.index("turn.death_ids[offset + row] = selected_id")
    first_mutation = body.index("live.alive[offset + slot] = false")
    assert selected < first_mutation
    assert "live.agent_id[offset + slot] = -1" in body
    assert "live.occupancy_agent[cell] = -1" in body
    assert "live.free_count[world] = virtual_count" in body


def test_c3_death_commit_does_not_repeat_authoritative_row_mutations() -> None:
    body = _body(
        POPULATION,
        "__global__ void commit_deaths_outer(",
        "TurnStage make_turn(",
    )
    for forbidden in (
        "state.alive[",
        "state.agent_id[",
        "state.pos_y[",
        "state.pos_x[",
        "state.occupancy_agent[",
        "state.occupancy_slot[",
        "state.patch_agent[",
        "state.free_queue[",
        "state.free_count[",
    ):
        assert forbidden not in body
    assert "state.event_active_count[world] = turn.event_count[world]" in body
    assert "death.pending_count[world] = 0" in body


def test_c3_turn_stage_aliases_are_the_same_outer_write_set() -> None:
    aliases = (
        "turn.alive = live.alive",
        "turn.agent_id = live.agent_id",
        "turn.occupancy_agent = live.occupancy_agent",
        "turn.occupancy_slot = live.occupancy_slot",
        "turn.patch_agent = live.patch_agent",
        "turn.id_to_slot = live.agent_id_to_slot",
        "turn.virtual_queue = live.free_queue",
        "turn.virtual_count = live.free_count",
    )
    for alias in aliases:
        assert alias in GRAPH


def test_c3_publish_remains_after_batch_guarded_status_merge() -> None:
    capture_start = COMPOSED.index("def capture_c4_resident_composed_r11_window_v1(")
    capture_end = COMPOSED.index("def bind_c4_resident_composed_r11_v1(", capture_start)
    capture = COMPOSED[capture_start:capture_end]
    merge = capture.index("owner.r11_extension.merge_window_control(")
    publish = capture.index("owner.transport_extension.publish_if_all_ok(")
    assert merge < publish
    assert "persistent_outer_status" in GRAPH
    assert "all_ok(stage, batch)" in GRAPH
