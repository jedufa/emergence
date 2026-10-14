from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WDL_CONTROLLER = (
    ROOT / "engine/perf/cuda_architecture_b_flattened_wdl_controller_v2_kernels.cu"
)


def test_genome_v2_changed_loci_feed_existing_lifecycle_mutation_flag() -> None:
    source = WDL_CONTROLLER.read_text(encoding="utf-8")

    changed_loci_bridge = (
        "const int64_t genome_v2_changed_loci =\n"
        "                genome_v2_staged_children[birth].changed_loci;"
    )
    assert changed_loci_bridge in source
    assert "if (genome_v2_changed_loci > 0) {" in source
    assert "mutation_flags[birth] = true;" in source
    assert (
        "mutation_deltas[birth] +=\n"
        "                    static_cast<float>(genome_v2_changed_loci);"
    ) in source


def test_genome_v2_bridge_precedes_event_preflight_and_reuses_eed_contract() -> None:
    source = WDL_CONTROLLER.read_text(encoding="utf-8")

    bridge_index = source.index("genome_v2_staged_children[birth].changed_loci;")
    policy_status_index = source.rindex(
        "if (statuses[0].policy_status != 0) {\n"
        "        EIEL_RUNTIME_BIRTH_PREPARE_FAIL(160 + statuses[0].policy_status);",
        0,
        bridge_index,
    )
    event_preflight_index = source.index(
        "events_v2::preflight_ordinary_birth_events_external_scratch_v2(",
        bridge_index,
    )
    event_staging_index = source.index(
        "if (mutation_flags[success_ordinal]) {",
        bridge_index,
    )

    assert policy_status_index < bridge_index < event_preflight_index
    assert bridge_index < event_staging_index
