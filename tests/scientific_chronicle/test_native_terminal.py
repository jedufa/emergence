from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from native_runtime.cuda_architecture_b_runner_owner_v1 import (
    _materialize_native_terminal_chronicle as materialize_compact_chronicle,
)
from native_runtime.architecture_b_reporting import (
    _materialize_native_terminal_chronicle as materialize_boundary_chronicle,
)
from simulation.chronicle.recorder import ScientificChronicleRecorder
from simulation.chronicle.storage import ARTIFACT_NAMES, read_json, read_jsonl


def _recorder(root: Path, *, profile: str = "notable") -> ScientificChronicleRecorder:
    cfg = SimpleNamespace(
        chronicle_enable=True,
        chronicle_profile=profile,
        chronicle_run_id="native-terminal-run",
        chronicle_terminal_enable=False,
        campaign_id="native-terminal-campaign",
        branch_id="native-terminal-branch",
        generations=2,
        allow_large_output_profile=profile in {"scientific_full", "debug_trace"},
    )
    return ScientificChronicleRecorder(SimpleNamespace(sim_cfg=cfg, run_dir=str(root)))


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _summaries() -> list[dict[str, int | str]]:
    return [
        {
            "schema_version": "eiel_patch_p2_production_compact_summary_v1",
            "generation": 0,
            "world": "patch",
            "alive": 64,
            "births": 5,
            "failed_births": 7,
            "deaths": 2,
            "event_cursor": 14,
            "moved": 100,
            "gathered": 20,
        },
        {
            "schema_version": "eiel_patch_p2_production_compact_summary_v1",
            "generation": 1,
            "world": "patch",
            "alive": 63,
            "births": 3,
            "failed_births": 4,
            "deaths": 4,
            "event_cursor": 23,
            "moved": 90,
            "gathered": 18,
        },
    ]


def test_native_terminal_ingestion_buffers_until_one_final_write(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path / "run")
    initial_tree = _tree_bytes(recorder.root)

    emitted = recorder.observe_native_terminal_summaries(_summaries())

    assert len(emitted) == 2
    assert _tree_bytes(recorder.root) == initial_tree
    recorder.finalize()

    events = read_jsonl(recorder.root / ARTIFACT_NAMES["events"])
    summary = read_json(recorder.root / ARTIFACT_NAMES["summary"])
    generation_events = [
        row for row in events if row["event_type"] == "generation_summary"
    ]
    assert [row["generation"] for row in generation_events] == [0, 1]
    assert generation_events[0]["measurements"]["native_typed_lifecycle_rows"] == 14
    assert generation_events[1]["measurements"]["native_typed_lifecycle_rows"] == 9
    assert generation_events[0]["measurements"]["cohort_size"] is None
    assert (
        summary["source_completeness"] == "native_terminal_compact_generation_summaries"
    )
    assert any("per-generation lineage" in text for text in summary["limitations"])


def test_native_terminal_full_profile_does_not_invent_lineage_rarity(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path / "run", profile="scientific_full")
    lifecycle = [
        {
            "generation": 0,
            "tick": 4,
            "sequence": 0,
            "event_type": 1,
            "child_agent_id": 70,
            "lineage_founder_id": 12,
            "reason_code": 0,
            "value": 0.0,
        }
    ]

    recorder.observe_native_terminal_summaries(
        _summaries(), typed_lifecycle_rows=lifecycle
    )
    recorder.finalize()

    events = read_jsonl(recorder.root / ARTIFACT_NAMES["events"])
    event_types = {row["event_type"] for row in events}
    assert "birth_observation" in event_types
    assert "notable_birth" not in event_types


def test_native_owner_terminal_materializer_writes_manifest_receipt(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    recorder = _recorder(run_dir)
    runner = SimpleNamespace(run_dir=str(run_dir), scientific_chronicle=recorder)
    settlement = SimpleNamespace(compact_summary_rows=_summaries())

    receipt = materialize_compact_chronicle(runner, settlement)

    assert receipt is not None
    assert receipt["status"] == "complete"
    assert receipt["generation_count"] == 2
    assert receipt["path"] == "chronicle/chronicle_manifest.json"
    assert (run_dir / receipt["path"]).is_file()
    assert recorder.finalized is True


def test_run_many_boundary_materializer_preserves_native_lineage_and_ecology(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    recorder = _recorder(run_dir)
    runner = SimpleNamespace(run_dir=str(run_dir), scientific_chronicle=recorder)

    def boundary(
        generation: int,
        *,
        founders: tuple[int, ...],
        depleted: float,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            generation=generation,
            counts={
                "alive": len(founders),
                "successful_births": 0,
                "failed_births": 0,
                "deaths": 0,
                "birth_attempts": 0,
                "capacity": 8,
                "free_slots": 8 - len(founders),
            },
            cohort_size=len(founders),
            cohort_survivors=len(founders),
            lifecycle_events={},
            lifecycle_start_cursor=0,
            lifecycle_end_cursor=0,
            overlay={
                "values": {
                    "native_lineage_metrics_input": {
                        "records": [
                            {"agent_id": index, "founder_id": founder}
                            for index, founder in enumerate(founders)
                        ]
                    },
                    "patch_food_depleted_frac": depleted,
                    "patch_food_amount_mean": 1.0 - depleted,
                }
            },
        )

    receipt = materialize_boundary_chronicle(
        runner,
        [
            boundary(0, founders=(1, 2), depleted=0.10),
            boundary(1, founders=(1,), depleted=0.40),
        ],
    )

    assert receipt is not None
    assert receipt["status"] == "complete"
    events = read_jsonl(recorder.root / ARTIFACT_NAMES["events"])
    assert {row["event_type"] for row in events}.issuperset(
        {"lineage_extinction", "regional_food_depletion"}
    )
    summary = read_json(recorder.root / ARTIFACT_NAMES["summary"])
    assert summary["source_completeness"] == (
        "native_terminal_generation_boundary_projections"
    )
