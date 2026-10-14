from __future__ import annotations

from pathlib import Path

import pytest

from eiel_composition.science_instrument_cli import _parser
from tools import eiel


_ROOT = Path(__file__).resolve().parents[1]


def test_operator_help_presents_science_run_as_the_normal_workflow() -> None:
    top_level_help = eiel.render_help()
    science_help = _parser().format_help()

    assert (
        "science run <request.json> [--human]  # normal science workflow"
        in top_level_help
    )
    assert "science {validate|plan} <request.json>  # optional diagnostics" in (
        top_level_help
    )
    assert "Normal workflow: eiel science run REQUEST --human." in science_help
    assert "{run,validate,plan,activate-authority,normalize,prepare,demo}" in (
        science_help
    )
    assert "machine administration: install" in science_help
    assert "lower-level diagnostic: normalize" in science_help
    assert "lower-level diagnostic: compile" in science_help
    assert "technical-only demonstration:" in science_help


def test_top_level_science_run_dispatches_to_the_science_front_door(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forwarded: list[str] = []

    def fake_core_tool_main(name: str):
        assert name == "science"

        def invoke(arguments: list[str]) -> int:
            forwarded.extend(arguments)
            return 0

        return invoke

    monkeypatch.setattr(eiel, "_core_tool_main", fake_core_tool_main)

    assert eiel.main(["science", "run", "campaign.json", "--human"]) == 0
    assert forwarded == ["run", "campaign.json", "--human"]


def test_science_governance_uses_the_one_command_default() -> None:
    root_guidance = (_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    science_guidance = (_ROOT / "governance/AGENTS_SCIENCE.md").read_text(
        encoding="utf-8"
    )

    canonical_command = "./tools/bin/eiel science run <request.json> --human"
    assert canonical_command in root_guidance
    assert canonical_command in science_guidance
