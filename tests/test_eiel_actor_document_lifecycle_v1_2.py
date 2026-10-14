from __future__ import annotations

from pathlib import Path

from tests.eiel_actor_v11_helpers import REPO_ROOT
from tools.actors.authority import validate_managed_document_path


def test_codex_operator_guide_lives_with_tool_not_directly_on_published_shelf() -> None:
    assert (REPO_ROOT / "tools/actors/CODEX_OPERATOR_GUIDE.md").is_file()
    forbidden = (
        REPO_ROOT
        / "docs/000_start-here/010_operator-guides/"
        "EIEL_Actor_Managers_Codex_Operator_Guide_v1_1.md"
    )
    assert not forbidden.exists()


def test_actor_related_managed_documents_have_permanent_ids() -> None:
    violations: list[str] = []
    for path in (REPO_ROOT / "docs").rglob("*"):
        if not path.is_file():
            continue
        lowered = path.name.lower()
        if "actor-manager" not in lowered and "actor_manager" not in lowered:
            continue
        problems = validate_managed_document_path(path, repo_root=REPO_ROOT)
        violations.extend(problems)
    assert violations == []


def test_old_unregistered_actor_closeouts_are_removed() -> None:
    incoming = REPO_ROOT / "docs/900_lifecycle/940_incoming"
    assert not (incoming / "eiel-actor-managers-v1-1-codex-verification-closeout-2026-08-03.md").exists()
    assert not (incoming / "eiel-actor-managers-v1-implementation-closeout-2026-08-03.md").exists()


def test_librarian_next_document_id_respects_visible_unregistered_doc_prefix(tmp_path):
    from tools.evidence.librarian.publication import _next_document_id, _reserved_document_ids_from_docs_tree

    repo = tmp_path / "repo"
    incoming = repo / "docs/900_lifecycle/940_incoming"
    incoming.mkdir(parents=True)
    (incoming / "EIEL-DOC-000003--already-reserved.md").write_text("reserved", encoding="utf-8")
    records = [
        {"document_id": "EIEL-DOC-000001"},
        {"document_id": "EIEL-DOC-000002"},
    ]
    reserved = _reserved_document_ids_from_docs_tree(repo)
    assert reserved == {"EIEL-DOC-000003"}
    document_id, sequence = _next_document_id(records, reserved_document_ids=reserved)
    assert document_id == "EIEL-DOC-000004"
    assert sequence == 4
