from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tests.eiel_actor_v11_helpers import REPO_ROOT
from tools.evidence.librarian import wp4


SOURCE = REPO_ROOT / "tools/evidence/librarian/catalog_source/00-catalog"
TOPIC = "qualification authority register"


def test_authority_domain_filter_narrows_real_catalog_resolution() -> None:
    catalog = wp4.build_catalog(SOURCE, REPO_ROOT)
    broad = wp4.evidence_preflight(catalog, TOPIC, collection="all")
    narrow = wp4.evidence_preflight(
        catalog,
        TOPIC,
        collection="all",
        authority_domain="execution_route",
    )
    assert broad["fail_closed"] is True
    assert broad["authority_resolution_state"] == "ambiguous"
    assert len({row["domain"] for row in broad["authority_consulted"]}) > 1

    assert narrow["fail_closed"] is False
    assert narrow["authority_resolution_state"] == "resolved"
    assert narrow["authority_domain_filter"] == "execution_route"
    assert {row["domain"] for row in narrow["authority_consulted"]} == {"execution_route"}
    assert narrow["catalog_build_id"] == broad["catalog_build_id"]
    assert narrow["source_digest"] == broad["source_digest"]


def test_librarian_cli_authority_domain_matches_core() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    command = [
        sys.executable,
        "-m",
        "tools.evidence.librarian",
        "preflight",
        "--topic",
        TOPIC,
        "--collection",
        "all",
        "--authority-domain",
        "execution_route",
        "--source-root",
        str(SOURCE),
        "--json",
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    cli = json.loads(result.stdout)
    core = wp4.evidence_preflight(
        wp4.build_catalog(SOURCE, REPO_ROOT),
        TOPIC,
        collection="all",
        authority_domain="execution_route",
    )
    assert cli == core


def test_unknown_authority_domain_remains_fail_closed() -> None:
    catalog = wp4.build_catalog(SOURCE, REPO_ROOT)
    receipt = wp4.evidence_preflight(
        catalog,
        TOPIC,
        collection="all",
        authority_domain="not-a-real-authority-domain",
    )
    assert receipt["fail_closed"] is True
    assert receipt["authority_resolution_state"] == "none"
    assert receipt["authority_consulted"] == []
    assert receipt["unresolved_authority_gaps"]
