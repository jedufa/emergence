from __future__ import annotations

import os

from simulation.metrics_pkg.epi_av_timeseries import EPITimeSeriesWriter


def test_epi_timeseries_writer_stable_bytes(tmp_path) -> None:
    run_dir = str(tmp_path)
    w = EPITimeSeriesWriter(run_dir=run_dir)

    w.append_row(generation=1, epi_value=71.86591234, av_value=None)
    w.append_row(generation=2, epi_value=72.0, av_value=0.123456789)

    p = os.path.join(run_dir, "epi_av_timeseries.csv")
    assert os.path.exists(p)

    data = open(p, "rb").read()

    # Header + two rows. Fixed float formatting (8 decimals).
    expected = (
        b"generation,epi_value,av_value\n"
        b"1,71.86591234,\n"
        b"2,72.00000000,0.12345679\n"
    )
    assert data == expected


def test_epi_timeseries_writer_is_append_only(tmp_path) -> None:
    run_dir = str(tmp_path)
    w = EPITimeSeriesWriter(run_dir=run_dir)

    w.append_row(generation=1, epi_value=1.0, av_value=None)
    w.append_row(generation=2, epi_value=2.0, av_value=None)
    w.append_row(generation=3, epi_value=3.0, av_value=None)

    p = os.path.join(run_dir, "epi_av_timeseries.csv")
    lines = open(p, "r", encoding="utf-8").read().splitlines()
    assert lines[0] == "generation,epi_value,av_value"
    assert lines[1].startswith("1,")
    assert lines[2].startswith("2,")
    assert lines[3].startswith("3,")
