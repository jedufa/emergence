from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCEDURE = ROOT / "governance/EIEL_Qualification_Preservation_and_Delta_Qualification_Procedure.md"
REGISTER = ROOT / (
    "docs/100_governance-authority/110_authority-registers/"
    "EIEL-DOC-000005--eiel-qualification-authority-register.json"
)
READING_COPY = ROOT / (
    "docs/100_governance-authority/110_authority-registers/"
    "EIEL-DOC-000004--eiel-qualification-authority-register.html"
)


def test_root_router_makes_qualification_preservation_universal() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "EIEL_Qualification_Preservation_and_Delta_Qualification_Procedure.md" in agents
    assert "existing qualification remains valid unless evidence shows" in agents
    assert "Broad requalification requires" in agents
    assert "nine-field qualification statement" in agents


def test_procedure_requires_impact_map_delta_scope_and_escalation() -> None:
    procedure = PROCEDURE.read_text(encoding="utf-8")
    for classification in (
        "PRESERVED_UNCHANGED",
        "PRESERVED_COMPATIBLE_INTERFACE",
        "DELTA_QUALIFICATION_REQUIRED",
        "BROAD_QUALIFICATION_REQUIRED",
        "IMPACT_UNCERTAIN",
    ):
        assert classification in procedure
    assert "No large qualification matrix may be planned before this map exists." in procedure
    assert "Do not multiply every existing variant" in procedure
    assert "STOP before executing it" in procedure
    assert "Broad requalification required: yes/no" in procedure


def test_existing_governance_paths_compose_the_preservation_rule() -> None:
    required_references = {
        "governance/EIEL_QualPrep_and_Qualification_Procedure.md": "qualification impact map",
        "governance/EIEL_Engineering_Validation_and_Baseline_Defect_Quarantine_Procedure.md": "prove equivalence",
        "governance/EIEL_Evidence_Sufficiency_and_Assurance_Budget_Procedure.md": "preserves unaffected qualified groups",
        "governance/EIEL_Closeout_Readiness_Procedure.md": "nine-field final qualification statement",
        "governance/EIEL_Artifact_Lineage_and_Canonical_Materialization_Procedure.md": "Artifact invalidation is not universal qualification invalidation",
    }
    for relative_path, required_text in required_references.items():
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "EIEL_Qualification_Preservation_and_Delta_Qualification_Procedure.md" in text
        assert required_text in text


def test_authority_register_and_reading_copy_carry_the_rule() -> None:
    register = json.loads(REGISTER.read_text(encoding="utf-8"))
    rule = register["policy"]["qualification_preservation_rule"]
    assert rule["procedure"] == (
        "governance/EIEL_Qualification_Preservation_and_Delta_Qualification_Procedure.md"
    )
    assert rule["default_presumption"].startswith("Existing qualification remains valid")
    assert len(rule["required_closeout_fields"]) == 9
    assert rule["required_closeout_fields"][-2] == "Broad requalification required: yes/no"

    html = READING_COPY.read_text(encoding="utf-8")
    assert "Qualification Preservation and Delta Qualification" in html
    assert rule["default_presumption"] in html
    assert rule["broad_requalification_rule"] in html

    result = subprocess.run(
        [
            sys.executable,
            "tools/scripts/render_eiel_qualification_authority_register_html.py",
            "--check",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
