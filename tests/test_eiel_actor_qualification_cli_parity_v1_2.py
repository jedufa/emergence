from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from engine.campaigns.qualification_framework import (
    AxisRangeV1,
    OperatingEnvelopeV1,
    QualificationCatalogV1,
)
from tests.eiel_actor_v11_helpers import REPO_ROOT, run_eiel
from tools.actors.qualification_manager import integration_boundary


def _catalog(path: Path) -> Path:
    envelope = OperatingEnvelopeV1(
        envelope_id="env-one",
        scope="CONFIGURATION_OPERATING_ENVELOPE",
        axes=(
            AxisRangeV1("grid_width", 16, 32),
            AxisRangeV1("grid_height", 16, 32),
            AxisRangeV1("ticks_per_generation", 80, 80),
        ),
        categorical_invariants={
            "provider_id": "provider",
            "runtime_id": "runtime",
            "world_family": "patch",
            "execution_route_sha256": "route",
        },
        anchor_ids=(),
        required_corners=(
            {"grid_width": 16, "grid_height": 16, "ticks_per_generation": 80},
            {"grid_width": 16, "grid_height": 32, "ticks_per_generation": 80},
            {"grid_width": 32, "grid_height": 16, "ticks_per_generation": 80},
            {"grid_width": 32, "grid_height": 32, "ticks_per_generation": 80},
        ),
        accepted_points=(
            {"grid_width": 16, "grid_height": 16, "ticks_per_generation": 80},
        ),
        evidence=(),
    )
    path.write_text(
        json.dumps(QualificationCatalogV1(envelopes=(envelope,)).payload()),
        encoding="utf-8",
    )
    return path


def _request(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "provider_id": "provider",
                "runtime_id": "runtime",
                "world_family": "patch",
                "execution_route_sha256": "route",
                "grid_width": 24,
                "grid_height": 24,
                "ticks_per_generation": 80,
            }
        ),
        encoding="utf-8",
    )
    return path


def _lower_cli(catalog: Path, request: Path) -> dict[str, object]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.scripts.eiel_qualification",
            "status",
            "--catalog",
            str(catalog),
            "--request",
            str(request),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_manager_preserves_lower_qualification_resolution_exactly(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path / "catalog.json")
    request = _request(tmp_path / "request.json")
    lower = _lower_cli(catalog, request)
    manager = run_eiel(
        [
            "qualificationManager",
            "check",
            "--catalog",
            str(catalog),
            "--request",
            str(request),
        ],
        state_root=tmp_path / "state",
        expected=1,
    )
    assert manager["source"] == "qf_catalog"
    assert manager["full_result"] == lower
    assert manager["decision"] == lower["decision"] == "DELTA_REQUIRED"
    command = manager["next_actions"][0]["command"]
    assert f"--catalog {catalog}" in command
    assert f"--request {request}" in command
    assert "--envelope-id env-one" in command


def test_manager_boundary_delegates_and_never_executes() -> None:
    boundary = integration_boundary()
    assert boundary["shared_core"] == "engine.campaigns.qualification_framework"
    assert boundary["lower_level_qualification_cli"] == (
        "python -m tools.scripts.eiel_qualification"
    )
    assert boundary["campaign_route_qualification_cli"] == (
        "python -m tools.scripts.eiel_campaign qualify"
    )
    assert boundary["manager_executes_campaigns"] is False
    assert boundary["manager_reimplements_qualification_semantics"] is False
    assert boundary["scientific_qualification_effect"] == "none"


def test_real_register_resolution_is_source_bound_and_non_authoritative(tmp_path: Path) -> None:
    register = REPO_ROOT / "docs/100_governance-authority/110_authority-registers/EIEL-DOC-000005--eiel-qualification-authority-register.json"
    if not register.is_file():
        pytest.skip("published qualification register is not included in this core transfer")
    result = run_eiel(
        [
            "qualificationManager",
            "check",
            "--track-label",
            "b48w16p64c128gNt160gpu",
            "--store",
            "--generated-at",
            "2026-08-03T12:00:00Z",
        ],
        state_root=tmp_path / "state",
    )
    assert result["decision"] == "QUALIFIED"
    assert result["source"] == "authority_register"
    assert result["full_result"]["record_id"] == "b48w16p64c128gNt160gpu"
    retained = run_eiel(
        ["qualificationManager", "show", result["check_id"]],
        state_root=tmp_path / "state",
    )
    payload = retained["record"]["payload"]
    assert payload["authoritative_effect"] == "none_read_only_resolution"
    assert retained["record"]["source_refs"][0]["kind"] == "authority_register"


def _lower_decide(evidence: Path) -> tuple[int, dict[str, object]]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.scripts.eiel_qualification",
            "decide",
            "--evidence",
            str(evidence),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stream = result.stdout.strip() or result.stderr.strip()
    assert stream, result
    return result.returncode, json.loads(stream)


def test_manager_and_lower_cli_share_fail_closed_decision_import(tmp_path: Path) -> None:
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps({"decision": "A", "note": "looks good"}), encoding="utf-8")
    lower_rc, lower = _lower_decide(bare)
    manager = run_eiel(
        ["qualificationManager", "decide", "--evidence", str(bare)],
        state_root=tmp_path / "state-bare",
        expected=1,
    )
    assert lower_rc == 1
    assert manager["qualification_decision"] == lower
    assert manager["qualification_decision"]["accepted_for_envelope_expansion"] is False
    assert "structured qualification evidence bindings are missing" in manager["problems"][0]

    structured = tmp_path / "structured.json"
    structured.write_text(
        json.dumps(
            {
                "decision": "A",
                "plan_sha256": "a" * 64,
                "required_obligations": ["repeat_parity"],
                "satisfied_recipe_obligations": ["repeat_parity"],
                "evidence_bindings": [
                    {
                        "evidence_id": "technical-fixture",
                        "kind": "qualification_fixture",
                        "uri": "fixture://qualification",
                        "sha256": "b" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    lower_rc, lower = _lower_decide(structured)
    manager = run_eiel(
        ["qualificationManager", "decide", "--evidence", str(structured)],
        state_root=tmp_path / "state-structured",
    )
    assert lower_rc == 0
    assert manager["qualification_decision"] == lower
    assert lower["accepted_for_envelope_expansion"] is True
    assert lower["scientific_qualification_effect"] == "none"
