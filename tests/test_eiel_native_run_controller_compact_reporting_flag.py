from __future__ import annotations

import pytest

from engine.native_runtime.run_controller import (
    NativeRunController,
    NativeRunControllerError,
)


def test_compact_reporting_only_requires_an_exact_boolean() -> None:
    with pytest.raises(
        NativeRunControllerError,
        match="compact_reporting_only must be boolean",
    ):
        NativeRunController.from_runner(
            object(),
            compact_reporting_only=1,  # type: ignore[arg-type]
        )


def test_compact_schedule_uses_one_terminal_checkpoint_without_exact_reporting() -> None:
    controller = object.__new__(NativeRunController)
    controller._start_generation = 0
    controller._schedule_origin_generation = 0
    controller._initial_boundary_ordinal = 0
    controller._compact_reporting_only = True

    expectations, projection_exports = controller._scheduled_boundary_expectations(
        completed=4,
        requested=4,
        schedule={
            "progress_every_generations": 1,
            "report_every_generations": 0,
            "checkpoint_every_generations": 0,
            "checkpoint_on_cancel": True,
            "explicit_next_generations": [4],
        },
    )

    assert len(expectations) == 4
    assert all(expectation["exact_due"] is False for expectation in expectations)
    assert [expectation["checkpoint_due"] for expectation in expectations] == [
        False,
        False,
        False,
        True,
    ]
    assert projection_exports == 1
