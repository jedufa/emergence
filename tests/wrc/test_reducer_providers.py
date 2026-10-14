from __future__ import annotations

import csv
import json
from pathlib import Path

from engine.wrc.reducer import ReducerConfig
from engine.wrc.reducer.providers import (
    ChronicleObservationProvider,
    MetricsCsvObservationProvider,
    PathTextResource,
)


def test_metrics_csv_provider_handles_large_json_fields(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    large = json.dumps([{"x": "a" * 150_000}])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["generation", "signal", "large_json"])
        writer.writerow([0, 1.5, large])
        writer.writerow([1, 2.5, large])

    rows = list(
        MetricsCsvObservationProvider(
            PathTextResource(path),
            ticks_per_observation=10,
        )
    )

    assert [row.sequence for row in rows] == [0, 1]
    assert [row.tick for row in rows] == [10, 20]
    assert [row.metrics for row in rows] == [{"signal": 1.5}, {"signal": 2.5}]
    assert [row.source.row_number for row in rows] == [2, 3]


def test_chronicle_provider_replays_generation_summaries_only(tmp_path: Path) -> None:
    path = tmp_path / "formal_events.jsonl"
    events = [
        {
            "event_id": "evt-0",
            "event_type": "generation_summary",
            "generation": 0,
            "step": 10,
            "measurements": {"alive": 100, "food_mean": 0.5, "flag": True},
        },
        {
            "event_id": "evt-x",
            "event_type": "lineage_appearance",
            "generation": 0,
            "step": 10,
            "measurements": {"alive": 100},
        },
        {
            "event_id": "evt-1",
            "event_type": "generation_summary",
            "generation": 1,
            "step": 20,
            "measurements": {"alive": 90, "food_mean": 0.7},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in events) + "\n")

    rows = list(ChronicleObservationProvider(PathTextResource(path)))

    assert [row.sequence for row in rows] == [0, 1]
    assert rows[0].metrics == {"alive": 100.0, "food_mean": 0.5}
    assert rows[1].source.event_id == "evt-1"


def test_required_metric_overrides_default_exclusion(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    path.write_text("generation,route_id,signal\n0,7,1\n", encoding="utf-8")

    rows = list(
        MetricsCsvObservationProvider(
            PathTextResource(path),
            config=ReducerConfig(),
            required_metrics=["route_id"],
        )
    )

    assert rows[0].metrics == {"route_id": 7.0, "signal": 1.0}


def test_only_final_partial_metrics_row_may_be_ignored(tmp_path: Path) -> None:
    from engine.wrc.reducer.providers import ObservationProviderError

    path = tmp_path / "metrics.csv"
    path.write_text(
        "generation,signal\n0,1\nnot-a-generation,2\n",
        encoding="utf-8",
    )
    rows = list(
        MetricsCsvObservationProvider(
            PathTextResource(path),
            allow_partial_last_row=True,
        )
    )
    assert [row.metrics for row in rows] == [{"signal": 1.0}]

    path.write_text(
        "generation,signal\n0,1\nnot-a-generation,2\n2,3\n",
        encoding="utf-8",
    )
    try:
        list(
            MetricsCsvObservationProvider(
                PathTextResource(path),
                allow_partial_last_row=True,
            )
        )
    except ObservationProviderError as exc:
        assert "invalid metrics row" in str(exc)
    else:
        raise AssertionError("malformed middle row was not rejected")
