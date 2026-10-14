from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.scripts import genome_upgrade_v2_parity_gate_v1 as gate


def test_collection_count_uses_explicit_collector_counts() -> None:
    output = (
        b"tests/test_one.py: 82\n"
        b"tests/test_two.py: 489\n"
    )

    assert gate._collection_count(output) == gate.EXPECTED_COUNTS[gate.SOURCE_LANE]


def test_junit_counts_are_structured_and_include_skips(tmp_path: Path) -> None:
    report = tmp_path / "report.xml"
    report.write_text(
        (
            "<testsuites>"
            '<testsuite tests="13" failures="1" errors="0" skipped="2">'
            "</testsuite>"
            "</testsuites>"
        ),
        encoding="ascii",
    )

    assert gate._junit_counts(report) == {
        "tests": 13,
        "failures": 1,
        "errors": 0,
        "skipped": 2,
        "passed": 10,
    }


def test_run_lane_binds_exact_argv_logs_and_execution_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def run(argv: list[str], **_kwargs: object) -> SimpleNamespace:
        junit = next(
            Path(value.split("=", 1)[1])
            for value in argv
            if value.startswith("--junitxml=")
        )
        junit.write_text(
            (
                "<testsuites>"
                '<testsuite tests="15" failures="0" errors="0" skipped="0">'
                "</testsuite>"
                "</testsuites>"
            ),
            encoding="ascii",
        )
        return SimpleNamespace(
            returncode=0,
            stdout=b"full stdout\n",
            stderr=b"full stderr\n",
        )

    monkeypatch.setattr(gate.subprocess, "run", run)

    receipt = gate._run_lane(
        gate.ENV_LANE,
        output_directory=tmp_path,
        collect_only=False,
        python=Path("/venv/bin/python"),
        pythonpath="/project/engine:/project",
    )

    assert receipt["ok"] is True
    assert receipt["counts"]["passed"] == 15
    assert receipt["exact_argv"][:4] == [
        "/venv/bin/python",
        "-m",
        "pytest",
        "-q",
    ]
    assert receipt["classification"]["requires_final_artifact"] is False
    assert (tmp_path / receipt["stdout_log"]).read_bytes() == b"full stdout\n"
    assert receipt["stdout_sha256"] == hashlib.sha256(b"full stdout\n").hexdigest()


def test_run_lane_fails_closed_on_skipped_device_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def run(argv: list[str], **_kwargs: object) -> SimpleNamespace:
        junit = next(
            Path(value.split("=", 1)[1])
            for value in argv
            if value.startswith("--junitxml=")
        )
        junit.write_text(
            (
                "<testsuites>"
                '<testsuite tests="15" failures="0" errors="0" skipped="1">'
                "</testsuite>"
                "</testsuites>"
            ),
            encoding="ascii",
        )
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(gate.subprocess, "run", run)

    receipt = gate._run_lane(
        gate.ENV_LANE,
        output_directory=tmp_path,
        collect_only=False,
        python=Path("/venv/bin/python"),
        pythonpath="/project/engine:/project",
    )

    assert receipt["ok"] is False
    assert receipt["counts"]["tests"] == 15
    assert receipt["counts"]["skipped"] == 1


def test_gate_preserves_the_active_venv_python_symlink(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    venv_python = tmp_path / ".venv" / "bin" / "python"
    monkeypatch.setattr(gate.sys, "executable", str(venv_python))
    monkeypatch.setattr(
        gate,
        "_clean_source_context",
        lambda: (tmp_path, "a" * 40),
    )
    monkeypatch.setattr(gate, "_validate_selectors", lambda _root: None)
    observed = []

    def run_lane(lane: str, **kwargs: object) -> dict[str, object]:
        observed.append(kwargs["python"])
        expected = gate.EXPECTED_COUNTS[lane]
        return {
            "ok": True,
            "counts": {
                "collected": expected,
                "tests": None,
            },
        }

    monkeypatch.setattr(gate, "_run_lane", run_lane)
    output = tmp_path.parent / f"{tmp_path.name}-output"

    receipt = gate.execute_gate(
        output_directory=output,
        collect_only=True,
    )

    assert observed == [venv_python, venv_python]
    assert receipt["runtime"]["python"] == str(venv_python)
