from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

from tools.evidence.librarian.cli import main
from tools.evidence.librarian.core import LibrarianError, file_sha256
from tools.evidence.librarian.ingress import (
    IngressRequest,
    execute_external_document_ingress,
    preview_external_document,
)
from tools.evidence.librarian.server import IngressServer


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _init_git_repo(repo_root: Path) -> str:
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.PIPE)
    subprocess.run(["git", "config", "user.email", "tests@example.invalid"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Ingress Tests"], cwd=repo_root, check=True)
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "synthetic ingress fixture"], cwd=repo_root, check=True, stdout=subprocess.PIPE)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()


def _existing_record(repo_root: Path, path: str, head: str) -> dict[str, object]:
    digest = file_sha256(repo_root / path)
    size = (repo_root / path).stat().st_size
    return {
        "schema_version": "eiel_document_record_v1",
        "collection": "documents",
        "document_id": "EIEL-DOC-000001",
        "title": "Existing",
        "path": path,
        "canonical_reading_location": path,
        "sha256": digest,
        "size_bytes": size,
        "git_state": "tracked",
        "document_class": "decision_report",
        "document_class_candidates": ["decision_report"],
        "refined_lifecycle_candidate": "synthetic_fixture",
        "currentness_status": "unreviewed",
        "currentness_candidate": "synthetic_review_required",
        "authored_or_generated": "candidate",
        "authorship_candidate": "authored_candidate",
        "retention_class": "candidate",
        "retention_candidate": "repo_publication",
        "review_state": "candidate",
        "review_flags": ["synthetic_fixture"],
        "supersedes_document_ids": [],
        "superseded_by_document_ids": [],
        "linked_evidence_ids": [],
        "linked_claim_ids": [],
        "linked_authority_assertion_ids": [],
        "linked_dossier_ids": [],
        "linked_raw_asset_ids": [],
        "linked_artifact_ids": [],
        "linked_issue_ids": [],
        "source_state": {"state": "synthetic_fixture", "observed_at_commit": head, "source_digest": digest},
        "identity": {
            "frozen": True,
            "path_at_freeze": "docs/40_eiel_design/original.md",
            "sha256_at_freeze": digest,
            "size_bytes_at_freeze": size,
            "freeze_commit": head,
            "freeze_source_digest": digest,
            "freeze_basis": ["synthetic_fixture"],
            "reuse_policy": "never_reuse",
            "path_is_identity": False,
            "path_history": [
                {
                    "publication_id": "synthetic-publication",
                    "from_path": "docs/40_eiel_design/original.md",
                    "to_path": path,
                    "sha256": digest,
                    "size_bytes": size,
                    "content_unchanged": True,
                    "authority_effect": "none",
                }
            ],
            "content_revisions": [],
        },
        "proposed_publication": {
            "proposal_version": "eiel_docs_publication_namespace_v1",
            "destination_class": "repo_document",
            "proposed_destination": path,
            "basis": ["synthetic_fixture"],
            "confidence": "candidate",
            "authority_effect": "none",
            "physical_move_authorized": True,
            "publication_state": "published",
        },
        "publication": {
            "publication_id": "synthetic-publication",
            "action": "repo_move",
            "path_before_publication": "docs/40_eiel_design/original.md",
            "current_path": path,
            "content_unchanged": True,
            "authority_effect": "none",
            "prepublication_head": head,
        },
        "notes": "Synthetic fixture.",
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo_root = tmp_path / "repo"
    source_root = repo_root / "tools/evidence/librarian/catalog_source/00-catalog"
    existing = "docs/500_campaigns-decisions-closeouts/520_campaign-reports/EIEL-DOC-000001--old.md"
    _write(repo_root / existing, "# Old\n")
    for directory in (
        "artifacts",
        "records",
        "claims",
        "authorities",
        "raw_assets",
        "issues",
        "documents",
        "document_relations",
        "accessions",
        "manifests",
        "policies",
        "schemas",
    ):
        (source_root / directory).mkdir(parents=True, exist_ok=True)
    _write_json(
        source_root / "schemas/document_record_v1.schema.json",
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {"document_class": {"enum": ["decision_report", "research_reference", "handoff"]}},
        },
    )
    _write_json(
        source_root / "schemas/document_relation_v1.schema.json",
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "basis": {
                    "type": "array",
                    "items": {
                        "enum": [
                            "exact_path_reference",
                            "identical_sha256",
                            "explicit_document_declaration",
                            "existing_librarian_relation",
                            "operator_governance_decision",
                            "path_history",
                            "manual_review",
                            "basename_only_unproven",
                            "unknown",
                        ]
                    },
                }
            },
        },
    )
    _write_json(
        source_root / "policies/document_identity_policy_v1.json",
        {
            "schema_version": "eiel_document_identity_policy_v1",
            "id_reuse_policy": "never_reuse",
            "path_is_identity": False,
            "allocation": {"next_sequence_after_current_freeze": 2},
        },
    )
    _write_json(
        source_root / "policies/document_publication_namespace_v1.json",
        {
            "schema_version": "eiel_document_publication_namespace_policy_v1",
            "status": "archive_publication_v1_published",
            "proposed_destination_rule": {"physical_move_authorized": True},
            "forbidden": [],
        },
    )
    _write_json(
        source_root / "manifests/DOCUMENT_LEGACY_PATH_REDIRECTS.json",
        {
            "schema_version": "eiel_document_legacy_path_redirects_v1",
            "publication_id": "synthetic-publication",
            "path_is_identity": False,
            "authority_effect": "none",
            "entry_count": 0,
            "entries": [],
        },
    )
    _write_json(
        source_root / "manifests/DOCUMENT_ARCHIVE_PUBLICATION_V1.json",
        {
            "schema_version": "eiel_document_archive_publication_manifest_v1",
            "publication_id": "synthetic-publication",
            "document_count": 1,
            "repo_document_count": 1,
            "runtime_relocation_count": 0,
            "authority_effect": "none",
            "publication_state": "published_pending_commit",
        },
    )
    head = _init_git_repo(repo_root)
    _write_json(source_root / "documents/EIEL-DOC-000001.json", _existing_record(repo_root, existing, head))
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "catalog fixture"], cwd=repo_root, check=True, stdout=subprocess.PIPE)
    external = tmp_path / "Downloads" / "new-reference.md"
    _write(external, "# New reference\n\nExternal text.\n")
    return repo_root, source_root, external


def _request(source: Path) -> IngressRequest:
    return IngressRequest(
        source=source,
        title="New reference",
        document_class="research_reference",
        destination_shelf="600_references-publications/600_research-references",
        linked_dossier_ids=("dossier-01-programme-evidence-map-methods-claim-boundaries",),
        publication_id="synthetic-human-ingress",
    )


def test_human_ingress_preview_is_deterministic_and_writes_nothing(tmp_path: Path) -> None:
    repo_root, source_root, external = _fixture(tmp_path)

    first = preview_external_document(repo_root=repo_root, source_root=source_root, request=_request(external))
    second = preview_external_document(repo_root=repo_root, source_root=source_root, request=_request(external))

    assert first == second
    assert first["ok"] is True
    assert first["document_id"] == "EIEL-DOC-000002"
    assert first["durable_writes"] is False
    assert not (repo_root / str(first["proposed_destination"])).exists()
    assert subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=repo_root, text=True) == ""


def test_human_ingress_blocks_exact_duplicates_without_allocating_id(tmp_path: Path) -> None:
    repo_root, source_root, external = _fixture(tmp_path)
    existing_path = repo_root / "docs/500_campaigns-decisions-closeouts/520_campaign-reports/EIEL-DOC-000001--old.md"
    external.write_bytes(existing_path.read_bytes())

    preview = preview_external_document(repo_root=repo_root, source_root=source_root, request=_request(external))

    assert preview["ok"] is False
    assert preview["exact_duplicates"][0]["document_id"] == "EIEL-DOC-000001"
    with pytest.raises(LibrarianError, match="exact duplicate|blockers"):
        execute_external_document_ingress(
            repo_root=repo_root,
            source_root=source_root,
            request=_request(external),
            mode="stage",
            runtime_root=tmp_path / "runtime",
        )
    assert not (repo_root / "docs/600_references-publications/600_research-references/EIEL-DOC-000002--new-reference.md").exists()


def test_human_ingress_stage_publishes_catalog_and_leaves_source_retained(tmp_path: Path) -> None:
    repo_root, source_root, external = _fixture(tmp_path)

    receipt = execute_external_document_ingress(
        repo_root=repo_root,
        source_root=source_root,
        request=_request(external),
        mode="stage",
        runtime_root=tmp_path / "runtime",
    )

    assert receipt["document_id"] == "EIEL-DOC-000002"
    assert receipt["authority_effect"] == "none"
    assert receipt["source_file_retained"] is True
    assert external.is_file()
    destination = repo_root / receipt["current_path"]
    assert destination.read_text(encoding="utf-8") == "# New reference\n\nExternal text.\n"
    record = json.loads((source_root / "documents/EIEL-DOC-000002.json").read_text())
    assert record["identity"]["path_at_freeze"] == external.resolve().as_posix()
    assert record["publication"]["action"] == "repo_copy"
    assert (repo_root / "docs/000_knowledge-catalog.html").is_file()
    status = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=repo_root, text=True)
    assert "EIEL-DOC-000002--new-reference.md" in status
    assert "docs/000_knowledge-catalog.html" in status


def test_human_ingress_rolls_back_when_catalog_publication_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root, source_root, external = _fixture(tmp_path)

    from tools.evidence.librarian import wp4

    def fail_write(*args: object, **kwargs: object) -> str:
        raise LibrarianError("synthetic catalog publication failure")

    monkeypatch.setattr(wp4, "write_catalog_html", fail_write)
    with pytest.raises(LibrarianError, match="synthetic catalog publication failure"):
        execute_external_document_ingress(
            repo_root=repo_root,
            source_root=source_root,
            request=_request(external),
            mode="stage",
            runtime_root=tmp_path / "runtime",
        )

    assert not (repo_root / "docs/600_references-publications/600_research-references/EIEL-DOC-000002--new-reference.md").exists()
    assert not (source_root / "documents/EIEL-DOC-000002.json").exists()
    assert subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=repo_root, text=True) == ""


def test_ingest_document_cli_dry_run_uses_shared_core(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo_root, source_root, external = _fixture(tmp_path)

    exit_code = main(
        [
            "ingest-document",
            str(external),
            "--repo-root",
            str(repo_root),
            "--source-root",
            str(source_root),
            "--title",
            "New reference",
            "--document-class",
            "research_reference",
            "--shelf",
            "600_references-publications/600_research-references",
            "--dry-run",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "eiel_human_document_ingress_preview_v1"
    assert payload["document_id"] == "EIEL-DOC-000002"


def test_ingress_web_rejects_post_without_token(tmp_path: Path) -> None:
    repo_root, source_root, _external = _fixture(tmp_path)
    server = IngressServer(
        ("127.0.0.1", 0),
        repo_root=repo_root,
        source_root=source_root,
        runtime_root=tmp_path / "runtime",
        token="secret-token",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/ingress/preview"
        request = Request(url, data=urlencode({"title": "No token"}).encode("utf-8"), method="POST")
        request.add_header("Content-Type", "application/x-www-form-urlencoded")
        with pytest.raises(HTTPError) as exc:
            urlopen(request, timeout=5)
        assert exc.value.code == 403
    finally:
        server.shutdown()
        server.server_close()


def test_ingress_web_preview_returns_decision_summary_with_token(tmp_path: Path) -> None:
    repo_root, source_root, external = _fixture(tmp_path)
    server = IngressServer(
        ("127.0.0.1", 0),
        repo_root=repo_root,
        source_root=source_root,
        runtime_root=tmp_path / "runtime",
        token="secret-token",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/ingress/preview"
        body = urlencode(
            {
                "token": "secret-token",
                "source_path": external.as_posix(),
                "title": "New reference",
                "document_class": "research_reference",
                "shelf": "600_references-publications/600_research-references",
                "publication_id": "synthetic-human-ingress",
            }
        ).encode("utf-8")
        request = Request(url, data=body, method="POST")
        request.add_header("Content-Type", "application/x-www-form-urlencoded")
        with urlopen(request, timeout=5) as response:
            page = response.read().decode("utf-8")
        assert "Decision Preview" in page
        assert "EIEL-DOC-000002" in page
        assert "proposed_destination" in page
    finally:
        server.shutdown()
        server.server_close()


def test_ingress_web_preview_chromium_smoke(tmp_path: Path) -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    repo_root, source_root, external = _fixture(tmp_path)
    server = IngressServer(
        ("127.0.0.1", 0),
        repo_root=repo_root,
        source_root=source_root,
        runtime_root=tmp_path / "runtime",
        token="secret-token",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=True)
            except Exception as exc:
                if "Executable doesn't exist" in str(exc):
                    pytest.skip("Playwright Chromium executable is not installed")
                raise
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            page.goto(f"http://127.0.0.1:{server.server_port}/", wait_until="networkidle")
            assert page.locator("a:text('Docs')").get_attribute("href").endswith(":8787/docs/")
            assert page.locator("a:text('Knowledge Catalog')").get_attribute("href").endswith(
                ":8787/docs/000_knowledge-catalog.html"
            )
            assert page.locator("select[name='document_class']").is_visible()
            assert page.locator("select[name='shelf_root']").is_visible()
            assert page.locator("select[name='shelf_child']").is_visible()
            page.select_option("select[name='shelf_root']", "600_references-publications")
            assert page.locator("select[name='shelf_child'] option").evaluate_all(
                "options => options.every(option => option.value === '' || option.value.startsWith('6'))"
            )
            page.fill("input[name='source_path']", external.as_posix())
            page.fill("input[name='title']", "New reference")
            page.select_option("select[name='document_class']", "research_reference")
            page.select_option("select[name='shelf_child']", "600_research-references")
            assert page.input_value("input[name='shelf']") == "600_references-publications/600_research-references"
            page.click("button:text('Preview')")
            page.wait_for_selector("text=Decision Preview")
            assert "EIEL-DOC-000002" in page.text_content("body")
            assert page.input_value("input[name='title']") == "New reference"
            assert page.input_value("select[name='document_class']") == "research_reference"
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
