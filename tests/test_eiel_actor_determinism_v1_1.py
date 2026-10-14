from __future__ import annotations

import copy
import json
from pathlib import Path

from tests.evidence.cogp1_fixtures import cogp1_manifest
from tests.eiel_actor_v11_helpers import FIXED, authorize_mission, run_eiel, scientific_mission_spec, write_test_qualification_register
from tools.actors.common import file_sha256


def _authorized_mission(state: Path, root: Path) -> str:
    spec = scientific_mission_spec()
    spec["source_identity"] = {"source_commit": "abc"}
    source = root / f"{state.name}-mission.json"
    source.write_text(json.dumps(spec), encoding="utf-8")
    created = run_eiel(
        ["missionPlanner", "create", "--input", str(source), "--generated-at", FIXED],
        state_root=state,
    )
    mission_id = created["mission"]["mission_id"]
    authorize_mission(mission_id, state_root=state, ready=True)
    return mission_id


def test_repeated_deterministic_evidence_assessment_and_export_are_identical(tmp_path: Path) -> None:
    manifest = cogp1_manifest(tmp_path / "campaign", signal="supported")
    state_a = tmp_path / "state-a"
    state_b = tmp_path / "state-b"
    reports_a = tmp_path / "reports-a"
    reports_b = tmp_path / "reports-b"
    mission_a = _authorized_mission(state_a, tmp_path)
    mission_b = _authorized_mission(state_b, tmp_path)

    first = run_eiel(
        [
            "evidenceManager", "assess", str(manifest), "--out-dir", str(reports_a),
            "--deterministic", "--generated-at", FIXED, "--mission", mission_a,
        ],
        state_root=state_a,
    )
    second = run_eiel(
        [
            "evidenceManager", "assess", str(manifest), "--out-dir", str(reports_b),
            "--deterministic", "--generated-at", FIXED, "--mission", mission_b,
        ],
        state_root=state_b,
    )
    assert first["assessment_id"] == second["assessment_id"]

    # Retained receipt and output paths are provenance and therefore differ across
    # independent actor-state roots.  Their bound content hashes, authority identity,
    # decision payload, and generated artifact bytes must remain deterministic.
    first_payload = copy.deepcopy(first["assessment"])
    second_payload = copy.deepcopy(second["assessment"])
    first_payload.pop("artifacts", None)
    second_payload.pop("artifacts", None)
    first_payload["authority_context"]["preflight"].pop("receipt_path", None)
    second_payload["authority_context"]["preflight"].pop("receipt_path", None)
    assert first_payload == second_payload

    # Artifact contents, not their absolute directory names, must match.
    for name, path_a in first["assessment"]["artifacts"].items():
        path_b = second["assessment"]["artifacts"][name]
        assert file_sha256(path_a) == file_sha256(path_b), name

    # Exporting the same sealed assessment twice must be byte-identical. Different
    # output roots are intentionally recorded as provenance and therefore need not
    # produce the same actor-record bytes.
    export_a = tmp_path / "a.zip"
    export_b = tmp_path / "b.zip"
    run_eiel(
        ["evidenceManager", "export", first["assessment_id"], "--export-output", str(export_a)],
        state_root=state_a,
    )
    run_eiel(
        ["evidenceManager", "export", first["assessment_id"], "--export-output", str(export_b)],
        state_root=state_a,
    )
    assert file_sha256(export_a) == file_sha256(export_b)


def test_actor_guidance_has_stable_required_fields(tmp_path: Path) -> None:
    authority_register = write_test_qualification_register(tmp_path / "qualification-authority.json")
    result = run_eiel(["qualificationManager", "check", "--track-label", "does-not-exist", "--authority-register", str(authority_register)], state_root=tmp_path / "state", expected=1)
    for field in ("status", "reason_codes", "problems", "warnings", "next_actions", "related_records"):
        assert field in result
    assert json.dumps(result, sort_keys=True)
