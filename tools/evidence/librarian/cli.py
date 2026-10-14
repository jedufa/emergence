"""CLI for the shared EIEL Knowledge Librarian."""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

import argparse
import json
import sys
from pathlib import Path

from .core import (
    AuditConfig,
    LibrarianError,
    build_catalog,
    check_no_regression,
    inspect_zip_safety,
    load_json,
    package_audit_supplement,
    run_audit,
    write_json,
)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--evidence-root", type=Path, default=Path("evidences"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--non-deterministic", action="store_true")
    parser.add_argument("--stability-delay-seconds", type=float, default=0.0)


def _compact_preflight(result: dict[str, object]) -> dict[str, object]:
    """Return the bounded model-facing projection of a full preflight receipt."""

    return {
        "topic": result.get("topic"),
        "authority_resolution_state": result.get("authority_resolution_state"),
        "authority_assertion_ids": [
            row.get("authority_assertion_id")
            for row in result.get("authority_consulted", [])
            if isinstance(row, dict)
        ],
        "evidence_ids": list(result.get("evidence_ids", []))[:5],
        "document_ids": list(result.get("document_ids", []))[:5],
        "claim_ids": list(result.get("claims_consulted", []))[:5],
        "limitation_ids": list(result.get("limitations_reviewed", []))[:5],
        "unknown_authority_gaps": list(
            result.get("unresolved_authority_gaps", [])
        )[:2],
        "catalog_build_id": result.get("catalog_build_id"),
        "maintained_source_digest": result.get("source_digest"),
        "fail_closed": result.get("fail_closed"),
        "fallback_repository_search_allowed": result.get(
            "fallback_repository_search_allowed"
        ),
        "next_step": result.get("next_step"),
    }

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eiel-evidence-librarian", description="Shared EIEL Knowledge Librarian: evidence + documents + claims + authority")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="write a baseline inventory and tree digest")
    _common(scan)

    audit = sub.add_parser("audit", help="write the full audit-evidence supplement")
    _common(audit)

    build = sub.add_parser("build-catalog", help="build deterministic JSON and HTML views")
    build.add_argument("--repo-root", type=Path, default=Path.cwd())
    build.add_argument("--audit-output-root", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--non-deterministic", action="store_true")

    render = sub.add_parser("render", help="render the HTML view from an audit output")
    render.add_argument("--repo-root", type=Path, default=Path.cwd())
    render.add_argument("--audit-output-root", type=Path, required=True)
    render.add_argument("--output-root", type=Path, required=True)
    render.add_argument("--non-deterministic", action="store_true")

    search = sub.add_parser("search", help="search generated catalog text or WP4 source")
    search.add_argument("--catalog", type=Path)
    search.add_argument("--source-root", type=Path, default=Path("tools/evidence/librarian/catalog_source/00-catalog"))
    search.add_argument("--repo-root", type=Path, default=Path.cwd())
    search.add_argument("--class", dest="object_class")
    search.add_argument("--collection", choices=("evidence", "documents", "all"), default="all")
    search.add_argument("--status")
    search.add_argument("--limit", type=int, default=20)
    search.add_argument("--exact-id", action="store_true")
    search.add_argument("--json", action="store_true")
    search.add_argument("query")

    show = sub.add_parser("show", help="show one generated catalog object by id")
    show.add_argument("--catalog", type=Path)
    show.add_argument("--source-root", type=Path, default=Path("tools/evidence/librarian/catalog_source/00-catalog"))
    show.add_argument("--repo-root", type=Path, default=Path.cwd())
    show.add_argument("--collection", choices=("evidence", "documents", "all"), default="all")
    show.add_argument("--json", action="store_true")
    show.add_argument("identifier")

    preflight = sub.add_parser("preflight", help="mandatory EOM-1 evidence preflight before repository source inspection")
    preflight.add_argument("--repo-root", type=Path, default=Path.cwd())
    preflight.add_argument("--source-root", type=Path, default=Path("tools/evidence/librarian/catalog_source/00-catalog"))
    preflight.add_argument("--catalog", type=Path)
    preflight.add_argument("--topic", required=True)
    preflight.add_argument("--limit", type=int, default=20)
    preflight.add_argument("--collection", choices=("evidence", "documents", "all"), default="all")
    preflight.add_argument("--authority-domain", help="narrow authority resolution to one maintained authority domain")
    preflight.add_argument("--receipt", type=Path)
    preflight.add_argument("--decision-ready", action="store_true")
    preflight.add_argument("--json", action="store_true")
    preflight.add_argument(
        "--compact",
        action="store_true",
        help="print only the decision-critical preflight fields; retain full receipt with --receipt",
    )

    decision = sub.add_parser("decision", help="show a concise decision-ready preflight summary")
    decision.add_argument("--repo-root", type=Path, default=Path.cwd())
    decision.add_argument("--source-root", type=Path, default=Path("tools/evidence/librarian/catalog_source/00-catalog"))
    decision.add_argument("--catalog", type=Path)
    decision.add_argument("--limit", type=int, default=20)
    decision.add_argument("--collection", choices=("evidence", "documents", "all"), default="all")
    decision.add_argument("--authority-domain", help="narrow authority resolution to one maintained authority domain")
    decision.add_argument("--receipt", type=Path)
    decision.add_argument("--json", action="store_true")
    decision.add_argument("topic")

    graph = sub.add_parser("graph", help="emit record relation graph edges")
    graph.add_argument("--catalog", type=Path, required=True)
    graph.add_argument("--output", type=Path, required=True)

    check = sub.add_parser("check", help="run strict no-regression checks")
    check.add_argument("--baseline-issues", type=Path, required=True)
    check.add_argument("--current-issues", type=Path, required=True)
    check.add_argument("--output", type=Path)
    check.add_argument("--strict", action="store_true")

    plan = sub.add_parser("plan-migration", help="write a plan-only migration candidate")
    plan.add_argument("--issue-ledger", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)

    zipcheck = sub.add_parser("inspect-zip", help="inspect a ZIP without extracting it")
    zipcheck.add_argument("zip_path", type=Path)
    zipcheck.add_argument("--output", type=Path)

    package = sub.add_parser("package-audit", help="package an audit output deterministically")
    package.add_argument("--audit-output-root", type=Path, required=True)
    package.add_argument("--output-zip", type=Path, required=True)

    wp2r = sub.add_parser("wp2r-curate", help="run WP2R provenance reconciliation and curation")
    wp2r.add_argument("--repo-root", type=Path, default=Path.cwd())
    wp2r.add_argument("--audit-output-root", type=Path, required=True)
    wp2r.add_argument("--repeat-audit-output-root", type=Path, required=True)
    wp2r.add_argument("--audit-supplement-zip", type=Path, required=True)
    wp2r.add_argument("--output-root", type=Path, required=True)

    wp3 = sub.add_parser("wp3-freeze", help="run WP3 conservative authority adjudication and accession freeze")
    wp3.add_argument("--repo-root", type=Path, default=Path.cwd())
    wp3.add_argument("--wp2-output-root", type=Path, required=True)
    wp3.add_argument("--wp2-package", type=Path, required=True)
    wp3.add_argument("--transport-copy", type=Path)
    wp3.add_argument("--source-root", type=Path, required=True)
    wp3.add_argument("--output-root", type=Path, required=True)
    wp3.add_argument("--observed-at-repo-commit", required=True)

    wp4 = sub.add_parser("wp4-build", help="run WP4 source sharding, dossiers, and query-view build")
    wp4.add_argument("--repo-root", type=Path, default=Path.cwd())
    wp4.add_argument("--source-root", type=Path, default=Path("tools/evidence/librarian/catalog_source/00-catalog"))
    wp4.add_argument("--wp3-package", type=Path, default=Path(public_runtime_path('evidence_raw/eiel_evidence_librarian_wp3/EIEL_Evidence_Librarian_WP3_Evidence_Package_2026_08_02.zip')))
    wp4.add_argument("--wp3-output-root", type=Path, default=Path(public_runtime_path('evidence_raw/eiel_evidence_librarian_wp3/20260802T000000Z_wp3_authority_adjudication_final')))
    wp4.add_argument("--output-root", type=Path, required=True)

    authority = sub.add_parser("authority", help="show WP4 authority policy rows for a domain")
    authority.add_argument("--repo-root", type=Path, default=Path.cwd())
    authority.add_argument("--source-root", type=Path, default=Path("tools/evidence/librarian/catalog_source/00-catalog"))
    authority.add_argument("--catalog", type=Path)
    authority.add_argument("--json", action="store_true")
    authority.add_argument("domain")

    for command, collection in (
        ("claims", "claims"),
        ("records", "records"),
        ("issues", "issues"),
        ("raw-assets", "raw_assets"),
        ("documents", "documents"),
        ("document-relations", "document_relations"),
    ):
        collection_parser = sub.add_parser(command, help=f"list WP4 {collection}")
        collection_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
        collection_parser.add_argument("--source-root", type=Path, default=Path("tools/evidence/librarian/catalog_source/00-catalog"))
        collection_parser.add_argument("--catalog", type=Path)
        collection_parser.add_argument("--limit", type=int, default=50)
        collection_parser.add_argument("--status")
        collection_parser.add_argument("--collection", choices=("evidence", "documents", "all"), default="all")
        collection_parser.add_argument("--json", action="store_true")
        collection_parser.add_argument("query", nargs="?")
        collection_parser.set_defaults(collection_name=collection)

    dossiers = sub.add_parser("dossiers", help="list WP4 synthesis dossiers")
    dossiers.add_argument("--repo-root", type=Path, default=Path.cwd())
    dossiers.add_argument("--source-root", type=Path, default=Path("tools/evidence/librarian/catalog_source/00-catalog"))
    dossiers.add_argument("--catalog", type=Path)
    dossiers.add_argument("dossier_id", nargs="?")
    dossiers.add_argument("--collection", choices=("evidence", "documents", "all"), default="all")
    dossiers.add_argument("--json", action="store_true")

    status = sub.add_parser("status", help="show WP4 source status")
    status.add_argument("--repo-root", type=Path, default=Path.cwd())
    status.add_argument("--source-root", type=Path, default=Path("tools/evidence/librarian/catalog_source/00-catalog"))
    status.add_argument("--view-root", type=Path)
    status.add_argument("--collection", choices=("evidence", "documents", "all"), default="all")
    status.add_argument("--json", action="store_true")

    verify = sub.add_parser("verify-catalog", help="verify WP4 referential integrity and stale generated catalog state")
    verify.add_argument("--repo-root", type=Path, default=Path.cwd())
    verify.add_argument("--source-root", type=Path, default=Path("tools/evidence/librarian/catalog_source/00-catalog"))
    verify.add_argument("--catalog", type=Path)
    verify.add_argument("--view-root", type=Path)
    verify.add_argument("--json", action="store_true")

    rebuild = sub.add_parser("rebuild-views", help="rebuild WP4 generated catalog and dossier views")
    rebuild.add_argument("--repo-root", type=Path, default=Path.cwd())
    rebuild.add_argument("--source-root", type=Path, default=Path("tools/evidence/librarian/catalog_source/00-catalog"))
    rebuild.add_argument("--output-root", type=Path, required=True)
    rebuild.add_argument("--json", action="store_true")

    register = sub.add_parser("register-accession", help="metadata-only automatic accession registration helper")
    register.add_argument("--repo-root", type=Path, default=Path.cwd())
    register.add_argument("--source-root", type=Path, default=Path("tools/evidence/librarian/catalog_source/00-catalog"))
    register.add_argument("--object-class", required=True)
    register.add_argument("--object-json", type=Path, required=True)
    register.add_argument("--package", type=Path)
    register.add_argument("--view-root", type=Path)
    register.add_argument("--receipt", type=Path)
    register.add_argument("--json", action="store_true")

    bootstrap_docs = sub.add_parser("bootstrap-documents", help="freeze audited docs metadata into the shared Knowledge Librarian")
    bootstrap_docs.add_argument("--repo-root", type=Path, default=Path.cwd())
    bootstrap_docs.add_argument("--source-root", type=Path, default=Path("tools/evidence/librarian/catalog_source/00-catalog"))
    bootstrap_docs.add_argument("--audit-root", type=Path, required=True)
    bootstrap_docs.add_argument("--observed-commit", required=True)
    bootstrap_docs.add_argument("--receipt", type=Path)
    bootstrap_docs.add_argument("--json", action="store_true")

    register_document = sub.add_parser(
        "register-document",
        help="allocate the next stable UID for one explicit tracked document",
    )
    register_document.add_argument("--repo-root", type=Path, default=Path.cwd())
    register_document.add_argument("--source-root", type=Path, default=Path("tools/evidence/librarian/catalog_source/00-catalog"))
    register_document.add_argument("--path", required=True)
    register_document.add_argument("--title", required=True)
    register_document.add_argument("--document-class", required=True)
    register_document.add_argument("--destination-shelf", required=True)
    register_document.add_argument("--observed-commit", required=True)
    register_document.add_argument("--dossier", action="append", default=[])
    register_document.add_argument("--receipt", type=Path)
    register_document.add_argument("--json", action="store_true")

    publish_document = sub.add_parser(
        "register-and-publish-document",
        help="allocate the next stable UID and physically publish one tracked document",
    )
    publish_document.add_argument("--repo-root", type=Path, default=Path.cwd())
    publish_document.add_argument("--source-root", type=Path, default=Path("tools/evidence/librarian/catalog_source/00-catalog"))
    publish_document.add_argument("--path", required=True)
    publish_document.add_argument("--title", required=True)
    publish_document.add_argument("--document-class", required=True)
    publish_document.add_argument("--destination-shelf", required=True)
    publish_document.add_argument("--observed-commit", required=True)
    publish_document.add_argument("--dossier", action="append", default=[])
    publish_document.add_argument("--reading-copy-of")
    publish_document.add_argument("--publication-id", default="post-publication-document-ingress-v1")
    publish_document.add_argument("--receipt", type=Path)
    publish_document.add_argument("--json", action="store_true")

    ingest_document = sub.add_parser(
        "ingest-document",
        help="preview, stage, or commit one external human-supplied document",
    )
    ingest_document.add_argument("source", type=Path)
    ingest_document.add_argument("--repo-root", type=Path, default=Path.cwd())
    ingest_document.add_argument("--source-root", type=Path, default=Path("tools/evidence/librarian/catalog_source/00-catalog"))
    ingest_document.add_argument("--title", required=True)
    ingest_document.add_argument("--document-class", required=True)
    ingest_document.add_argument("--shelf", required=True)
    ingest_document.add_argument("--dossier", action="append", default=[])
    ingest_document.add_argument("--reading-copy-of")
    ingest_document.add_argument("--publication-id")
    ingest_document.add_argument("--runtime-root", type=Path, default=Path(public_runtime_path('evidence_raw/eiel_knowledge_librarian_human_ingress_v1')))
    mode = ingest_document.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--stage", action="store_true")
    mode.add_argument("--commit", action="store_true")
    ingest_document.add_argument("--trash-source", action="store_true")
    ingest_document.add_argument("--receipt", type=Path)
    ingest_document.add_argument("--json", action="store_true")

    serve = sub.add_parser(
        "serve",
        help="serve the token-protected local Knowledge Librarian ingress UI",
    )
    serve.add_argument("--repo-root", type=Path, default=Path.cwd())
    serve.add_argument("--source-root", type=Path, default=Path("tools/evidence/librarian/catalog_source/00-catalog"))
    serve.add_argument("--runtime-root", type=Path, default=Path(public_runtime_path('evidence_raw/eiel_knowledge_librarian_human_ingress_v1')))
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)
    serve.add_argument("--token")

    validate_publication = sub.add_parser(
        "validate-document-publication",
        help="fail-closed validation of a document publication plan",
    )
    validate_publication.add_argument("--repo-root", type=Path, default=Path.cwd())
    validate_publication.add_argument("--plan", type=Path, required=True)
    validate_publication.add_argument("--runtime-root", type=Path)
    validate_publication.add_argument("--expected-head")
    validate_publication.add_argument("--expected-document-count", type=int)
    validate_publication.add_argument("--receipt", type=Path)
    validate_publication.add_argument("--json", action="store_true")

    prepare_publication = sub.add_parser(
        "prepare-document-publication",
        help="copy and verify a non-destructive document publication candidate",
    )
    prepare_publication.add_argument("--repo-root", type=Path, default=Path.cwd())
    prepare_publication.add_argument("--plan", type=Path, required=True)
    prepare_publication.add_argument("--candidate-root", type=Path, required=True)
    prepare_publication.add_argument("--runtime-stage-root", type=Path, required=True)
    prepare_publication.add_argument("--runtime-root", type=Path)
    prepare_publication.add_argument("--expected-head")
    prepare_publication.add_argument("--expected-document-count", type=int)
    prepare_publication.add_argument("--receipt", type=Path)
    prepare_publication.add_argument("--json", action="store_true")

    publish_documents = sub.add_parser(
        "publish-documents",
        help="explicitly authorize and publish a verified document archive candidate",
    )
    publish_documents.add_argument("--repo-root", type=Path, default=Path.cwd())
    publish_documents.add_argument("--plan", type=Path, required=True)
    publish_documents.add_argument("--candidate-root", type=Path, required=True)
    publish_documents.add_argument("--runtime-stage-root", type=Path, required=True)
    publish_documents.add_argument("--runtime-root", type=Path)
    publish_documents.add_argument("--old-docs-root", type=Path, required=True)
    publish_documents.add_argument("--backup-zip", type=Path, required=True)
    publish_documents.add_argument("--expected-head", required=True)
    publish_documents.add_argument("--expected-document-count", type=int, required=True)
    publish_documents.add_argument("--authorization", required=True)
    publish_documents.add_argument("--receipt", type=Path)
    publish_documents.add_argument("--json", action="store_true")

    finalize_publication = sub.add_parser(
        "finalize-document-publication",
        help="record published paths, redirects, and semantic-equivalence metadata",
    )
    finalize_publication.add_argument("--repo-root", type=Path, default=Path.cwd())
    finalize_publication.add_argument("--source-root", type=Path, default=Path("tools/evidence/librarian/catalog_source/00-catalog"))
    finalize_publication.add_argument("--plan", type=Path, required=True)
    finalize_publication.add_argument("--runtime-root", type=Path)
    finalize_publication.add_argument("--publication-id", required=True)
    finalize_publication.add_argument("--prepublication-head", required=True)
    finalize_publication.add_argument("--receipt", type=Path)
    finalize_publication.add_argument("--json", action="store_true")

    resolve_document = sub.add_parser(
        "resolve-document",
        help="resolve a document ID, legacy path, or current path",
    )
    resolve_document.add_argument("--source-root", type=Path, default=Path("tools/evidence/librarian/catalog_source/00-catalog"))
    resolve_document.add_argument("--repo-root", type=Path, default=Path.cwd())
    resolve_document.add_argument("--json", action="store_true")
    resolve_document.add_argument("identifier_or_path")

    rewrite_paths = sub.add_parser(
        "rewrite-document-path-references",
        help="dry-run or apply exact legacy document path replacements in active tracked text files",
    )
    rewrite_paths.add_argument("--repo-root", type=Path, default=Path.cwd())
    rewrite_paths.add_argument("--redirect-manifest", type=Path, required=True)
    rewrite_paths.add_argument("--execute", action="store_true")
    rewrite_paths.add_argument("--authorization")
    rewrite_paths.add_argument("--receipt", type=Path)
    rewrite_paths.add_argument("--json", action="store_true")

    rewrite_documents = sub.add_parser(
        "rewrite-published-document-references",
        help="rewrite legacy paths in explicitly selected living documents",
    )
    rewrite_documents.add_argument("--repo-root", type=Path, default=Path.cwd())
    rewrite_documents.add_argument("--source-root", type=Path, default=Path("tools/evidence/librarian/catalog_source/00-catalog"))
    rewrite_documents.add_argument("--redirect-manifest", type=Path, required=True)
    rewrite_documents.add_argument("--document-id", action="append", required=True)
    rewrite_documents.add_argument("--execute", action="store_true")
    rewrite_documents.add_argument("--authorization")
    rewrite_documents.add_argument("--receipt", type=Path)
    rewrite_documents.add_argument("--json", action="store_true")

    record_revisions = sub.add_parser(
        "record-document-content-revisions",
        help="record controlled edits while preserving stable document identities",
    )
    record_revisions.add_argument("--repo-root", type=Path, default=Path.cwd())
    record_revisions.add_argument("--source-root", type=Path, default=Path("tools/evidence/librarian/catalog_source/00-catalog"))
    record_revisions.add_argument("--document-id", action="append", required=True)
    record_revisions.add_argument("--reason", required=True)
    record_revisions.add_argument("--authorization", required=True)
    record_revisions.add_argument("--receipt", type=Path)
    record_revisions.add_argument("--json", action="store_true")

    publish_site = sub.add_parser(
        "publish-knowledge-site",
        help="write the generated human Knowledge Catalog into docs/000_knowledge-catalog.html",
    )
    publish_site.add_argument("--repo-root", type=Path, default=Path.cwd())
    publish_site.add_argument("--source-root", type=Path, default=Path("tools/evidence/librarian/catalog_source/00-catalog"))
    publish_site.add_argument("--output", type=Path, default=Path("docs/000_knowledge-catalog.html"))
    publish_site.add_argument("--receipt", type=Path)
    publish_site.add_argument("--json", action="store_true")

    verify_documents = sub.add_parser(
        "verify-published-documents",
        help="verify all current repo/runtime documents against maintained records",
    )
    verify_documents.add_argument("--repo-root", type=Path, default=Path.cwd())
    verify_documents.add_argument("--source-root", type=Path, default=Path("tools/evidence/librarian/catalog_source/00-catalog"))
    verify_documents.add_argument("--runtime-root", type=Path)
    verify_documents.add_argument("--receipt", type=Path)
    verify_documents.add_argument("--json", action="store_true")

    return parser


def _audit_config(args: argparse.Namespace) -> AuditConfig:
    return AuditConfig(
        repo_root=args.repo_root,
        evidence_root=args.evidence_root,
        output_root=args.output_root,
        deterministic=not args.non_deterministic,
        stability_delay_seconds=args.stability_delay_seconds,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command in {"scan", "audit"}:
            result = run_audit(_audit_config(args))
        elif args.command in {"build-catalog", "render"}:
            result = build_catalog(
                repo_root=args.repo_root,
                audit_output_root=args.audit_output_root,
                output_root=args.output_root,
                deterministic=not args.non_deterministic,
            )
        elif args.command == "search":
            if args.source_root or not args.catalog:
                from . import wp4

                catalog = wp4._load_catalog_or_source(
                    args.catalog,
                    _wp4_source(args.repo_root, args.source_root),
                    args.repo_root,
                )
                result = wp4.search_catalog(
                    catalog,
                    args.query,
                    object_class=args.object_class,
                    collection=args.collection,
                    status=args.status,
                    limit=args.limit,
                    exact_id=args.exact_id,
                )
                print(json.dumps(result, indent=2, sort_keys=True))
                return 0 if result["match_count"] else wp4.EXIT_NO_MATCH
            result = _search(args.catalog, args.query)
        elif args.command == "show":
            if args.source_root or not args.catalog:
                from . import wp4

                catalog = wp4._load_catalog_or_source(
                    args.catalog,
                    _wp4_source(args.repo_root, args.source_root),
                    args.repo_root,
                )
                result = wp4.show_object(catalog, args.identifier, collection=args.collection)
                print(json.dumps(result, indent=2, sort_keys=True))
                return 0
            result = _show(args.catalog, args.identifier)
        elif args.command == "preflight":
            from . import wp4

            catalog = wp4._load_catalog_or_source(
                args.catalog,
                _wp4_source(args.repo_root, args.source_root),
                args.repo_root,
            )
            result = wp4.evidence_preflight(
                catalog,
                args.topic,
                limit=args.limit,
                collection=args.collection,
                authority_domain=args.authority_domain,
            )
            if args.receipt:
                write_json(args.receipt, result, pretty=True)
            if args.decision_ready:
                summary = wp4.decision_ready_preflight(
                    catalog,
                    args.topic,
                    limit=args.limit,
                    collection=args.collection,
                    authority_domain=args.authority_domain,
                )
                print(wp4.format_decision_ready_summary(summary))
                if not args.compact:
                    print("\nDetailed preflight audit:")
            if args.compact:
                print(
                    json.dumps(
                        _compact_preflight(result),
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                )
            else:
                print(json.dumps(result, indent=2, sort_keys=True))
            return wp4.EXIT_AMBIGUOUS if result["fail_closed"] else 0
        elif args.command == "decision":
            from . import wp4

            catalog = wp4._load_catalog_or_source(
                args.catalog,
                _wp4_source(args.repo_root, args.source_root),
                args.repo_root,
            )
            result = wp4.decision_ready_preflight(
                catalog,
                args.topic,
                limit=args.limit,
                collection=args.collection,
                authority_domain=args.authority_domain,
            )
            if args.receipt:
                write_json(args.receipt, result, pretty=True)
            if args.json:
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                print(wp4.format_decision_ready_summary(result))
            return wp4.EXIT_AMBIGUOUS if result["preflight_receipt"]["fail_closed"] else 0
        elif args.command == "graph":
            result = _graph(args.catalog, args.output)
        elif args.command == "check":
            result = check_no_regression(
                args.baseline_issues, args.current_issues, strict=args.strict
            )
            if args.output:
                write_json(args.output, result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 1 if result["status"] == "failed" else 0
        elif args.command == "plan-migration":
            from .core import build_migration_plan

            result = build_migration_plan(load_json(args.issue_ledger))
            write_json(args.output, result)
        elif args.command == "inspect-zip":
            result = inspect_zip_safety(args.zip_path)
            if args.output:
                write_json(args.output, result)
        elif args.command == "package-audit":
            result = package_audit_supplement(args.audit_output_root, args.output_zip)
        elif args.command == "wp2r-curate":
            from .wp2r import build_wp2r

            result = build_wp2r(
                repo_root=args.repo_root,
                audit_root=args.audit_output_root,
                repeat_audit_root=args.repeat_audit_output_root,
                supplement_zip=args.audit_supplement_zip,
                output_root=args.output_root,
            )
        elif args.command == "wp3-freeze":
            from .wp3 import run_wp3

            result = run_wp3(
                repo_root=args.repo_root,
                wp2_root=args.wp2_output_root,
                wp2_package=args.wp2_package,
                transport_copy=args.transport_copy,
                source_root=args.source_root,
                output_root=args.output_root,
                observed_at_repo_commit=args.observed_at_repo_commit,
            )
        elif args.command == "wp4-build":
            from .wp4 import run_wp4

            result = run_wp4(
                repo_root=args.repo_root,
                source_root=_wp4_source(args.repo_root, args.source_root),
                wp3_package=args.wp3_package,
                wp3_output_root=args.wp3_output_root,
                output_root=args.output_root,
            )
        elif args.command == "authority":
            from . import wp4

            catalog = wp4._load_catalog_or_source(
                args.catalog,
                _wp4_source(args.repo_root, args.source_root),
                args.repo_root,
            )
            result = wp4.authority_query(catalog, args.domain)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        elif args.command in {"claims", "records", "issues", "raw-assets", "documents", "document-relations"}:
            from . import wp4

            src = _wp4_source(args.repo_root, args.source_root)
            catalog = wp4._load_catalog_or_source(args.catalog, src, args.repo_root)
            result = wp4.collection_query(
                catalog,
                args.collection_name,
                status=args.status,
                limit=args.limit,
                logical_collection=args.collection,
            )
            if args.query:
                rows = [
                    row for row in result["rows"]
                    if args.query.casefold() in json.dumps(row, sort_keys=True, ensure_ascii=False).casefold()
                ]
                result = {**result, "count": len(rows), "rows": rows}
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        elif args.command == "dossiers":
            from . import wp4

            catalog = wp4._load_catalog_or_source(
                args.catalog,
                _wp4_source(args.repo_root, args.source_root),
                args.repo_root,
            )
            result = wp4.dossiers_query(catalog, collection=args.collection)
            if args.dossier_id:
                rows = [row for row in result["dossiers"] if row["dossier_id"] == args.dossier_id]
                if not rows:
                    raise wp4.CliExit(f"dossier not found: {args.dossier_id}", wp4.EXIT_NO_MATCH, {"dossier_id": args.dossier_id})
                result = {"schema_version": "eiel_wp4_dossier_query_v1", "dossier": rows[0]}
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        elif args.command == "status":
            from . import wp4

            src = _wp4_source(args.repo_root, args.source_root)
            result = wp4.status_query(args.repo_root, src, collection=args.collection)
            if args.view_root:
                catalog_path = args.view_root / wp4.KNOWLEDGE_CATALOG_JSON
                result = {
                    **result,
                    "html_catalog": str(args.view_root / wp4.KNOWLEDGE_CATALOG_HTML),
                    "verify_catalog": wp4.verify_catalog(src, catalog_path, args.repo_root),
                }
        elif args.command == "verify-catalog":
            from . import wp4

            src = _wp4_source(args.repo_root, args.source_root)
            catalog_path = args.catalog
            if args.view_root:
                catalog_path = args.view_root / wp4.KNOWLEDGE_CATALOG_JSON
            result = wp4.verify_catalog(src, catalog_path, args.repo_root)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["ok"] else wp4.EXIT_STALE_VIEW
        elif args.command == "rebuild-views":
            from . import wp4

            src = _wp4_source(args.repo_root, args.source_root)
            result = wp4.rebuild_views(
                args.repo_root,
                src,
                args.output_root,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        elif args.command == "register-accession":
            from . import wp4

            obj = load_json(args.object_json)
            result = wp4.register_accession_object(
                _wp4_source(args.repo_root, args.source_root),
                args.object_class,
                obj,
                repo_root=args.repo_root,
                package_path=args.package,
                view_root=args.view_root,
            )
            if args.receipt:
                write_json(args.receipt, result, pretty=True)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        elif args.command == "bootstrap-documents":
            from . import wp4
            from .documents import (
                build_document_issues,
                build_document_records,
                build_document_relations,
                validate_document_objects,
                write_bootstrap_source,
            )

            src = _wp4_source(args.repo_root, args.source_root)
            records = build_document_records(
                repo_root=args.repo_root,
                catalog_candidate_path=args.audit_root / "document_catalog_candidate_v1.json",
                refined_lifecycle_path=args.audit_root / "docs_refined_lifecycle_candidate_v1.json",
                authority_queue_path=args.audit_root / "document_authority_review_queue_candidate_v1.json",
                observed_commit=args.observed_commit,
            )
            relations = build_document_relations(records)
            document_issues = build_document_issues(records)
            result = write_bootstrap_source(
                source_root=src,
                records=records,
                relations=relations,
                document_issues=document_issues,
            )
            objects = wp4.load_sharded_accessions(src)
            validation = validate_document_objects(objects, source_root=src)
            integrity = wp4.referential_integrity(objects)
            if not validation["ok"] or not integrity["ok"]:
                raise LibrarianError("document bootstrap failed schema or referential-integrity validation")
            result = {
                **result,
                "document_validation": validation,
                "referential_integrity": integrity,
                "generated_aggregates": wp4.generate_aggregate_ledgers(src, objects),
                "source_manifest": wp4.write_source_manifest(src, source_layout="per_object_shards"),
            }
            if args.receipt:
                write_json(args.receipt, result, pretty=True)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        elif args.command in {
            "validate-document-publication",
            "prepare-document-publication",
            "publish-documents",
        }:
            from .publication import (
                DEFAULT_RUNTIME_ROOT,
                execute_publication,
                prepare_publication_candidate,
                validate_publication_plan,
            )

            runtime_root = args.runtime_root or DEFAULT_RUNTIME_ROOT
            if args.command == "validate-document-publication":
                result = validate_publication_plan(
                    args.repo_root,
                    args.plan,
                    runtime_root=runtime_root,
                    expected_head=args.expected_head,
                    expected_document_count=args.expected_document_count,
                )
            elif args.command == "prepare-document-publication":
                result = prepare_publication_candidate(
                    args.repo_root,
                    args.plan,
                    args.candidate_root,
                    args.runtime_stage_root,
                    runtime_root=runtime_root,
                    expected_head=args.expected_head,
                    expected_document_count=args.expected_document_count,
                )
            else:
                result = execute_publication(
                    args.repo_root,
                    args.plan,
                    args.candidate_root,
                    args.runtime_stage_root,
                    args.old_docs_root,
                    args.backup_zip,
                    authorization=args.authorization,
                    runtime_root=runtime_root,
                    expected_head=args.expected_head,
                    expected_document_count=args.expected_document_count,
                )
            if args.receipt:
                write_json(args.receipt, result, pretty=True)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        elif args.command == "register-document":
            from . import wp4
            from .documents import register_repo_document

            src = _wp4_source(args.repo_root, args.source_root)
            result = register_repo_document(
                repo_root=args.repo_root,
                source_root=src,
                path=args.path,
                title=args.title,
                document_class=args.document_class,
                destination_shelf=args.destination_shelf,
                observed_commit=args.observed_commit,
                linked_dossier_ids=args.dossier,
            )
            objects = wp4.load_sharded_accessions(src)
            validation = wp4.referential_integrity(objects)
            if not validation["ok"]:
                raise LibrarianError("explicit document registration failed referential-integrity validation")
            result = {
                **result,
                "referential_integrity": validation,
                "generated_aggregates": wp4.generate_aggregate_ledgers(src, objects),
                "source_manifest": wp4.write_source_manifest(src, source_layout="per_object_shards"),
            }
            if args.receipt:
                write_json(args.receipt, result, pretty=True)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        elif args.command == "register-and-publish-document":
            from . import wp4
            from .publication import register_and_publish_repo_document

            src = _wp4_source(args.repo_root, args.source_root)
            result = register_and_publish_repo_document(
                repo_root=args.repo_root,
                source_root=src,
                path=args.path,
                title=args.title,
                document_class=args.document_class,
                destination_shelf=args.destination_shelf,
                observed_commit=args.observed_commit,
                linked_dossier_ids=args.dossier,
                reading_copy_of=args.reading_copy_of,
                publication_id=args.publication_id,
            )
            objects = wp4.load_sharded_accessions(src)
            document_validation = wp4.validate_document_objects(objects, source_root=src)
            integrity = wp4.referential_integrity(objects)
            if not document_validation["ok"] or not integrity["ok"]:
                raise LibrarianError("post-publication document ingress failed catalog validation")
            result = {
                **result,
                "document_validation": document_validation,
                "referential_integrity": integrity,
                "generated_aggregates": wp4.generate_aggregate_ledgers(src, objects),
                "source_manifest": wp4.write_source_manifest(src, source_layout="per_object_shards"),
            }
            if args.receipt:
                write_json(args.receipt, result, pretty=True)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        elif args.command == "ingest-document":
            from .ingress import ingest_document_from_args

            args.source_root = _wp4_source(args.repo_root, args.source_root)
            result = ingest_document_from_args(args)
            if args.receipt:
                write_json(args.receipt, result, pretty=True)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        elif args.command == "serve":
            from .server import serve

            serve(
                host=args.host,
                port=args.port,
                repo_root=args.repo_root,
                source_root=_wp4_source(args.repo_root, args.source_root),
                runtime_root=args.runtime_root,
                token=args.token,
            )
            return 0
        elif args.command == "finalize-document-publication":
            from . import wp4
            from .publication import DEFAULT_RUNTIME_ROOT, finalize_catalog_publication

            src = _wp4_source(args.repo_root, args.source_root)
            result = finalize_catalog_publication(
                args.repo_root,
                src,
                args.plan,
                publication_id=args.publication_id,
                prepublication_head=args.prepublication_head,
                runtime_root=args.runtime_root or DEFAULT_RUNTIME_ROOT,
            )
            objects = wp4.load_sharded_accessions(src)
            integrity = wp4.referential_integrity(objects)
            if not integrity["ok"]:
                raise LibrarianError("published document catalog failed referential-integrity validation")
            result = {
                **result,
                "referential_integrity": integrity,
                "generated_aggregates": wp4.generate_aggregate_ledgers(src, objects),
                "source_manifest": wp4.write_source_manifest(src, source_layout="per_object_shards"),
            }
            if args.receipt:
                write_json(args.receipt, result, pretty=True)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        elif args.command == "resolve-document":
            from .publication import resolve_document_locator

            result = resolve_document_locator(
                _wp4_source(args.repo_root, args.source_root),
                args.identifier_or_path,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        elif args.command == "rewrite-document-path-references":
            from .publication import rewrite_repository_path_references

            if args.execute and not args.authorization:
                raise LibrarianError("--execute requires --authorization")
            if not args.execute and args.authorization:
                raise LibrarianError("--authorization is valid only with --execute")
            result = rewrite_repository_path_references(
                args.repo_root,
                args.redirect_manifest,
                authorization=args.authorization if args.execute else None,
            )
            if args.receipt:
                write_json(args.receipt, result, pretty=True)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        elif args.command == "rewrite-published-document-references":
            from .publication import rewrite_published_document_references

            if args.execute and not args.authorization:
                raise LibrarianError("--execute requires --authorization")
            if not args.execute and args.authorization:
                raise LibrarianError("--authorization is valid only with --execute")
            result = rewrite_published_document_references(
                args.repo_root,
                _wp4_source(args.repo_root, args.source_root),
                args.redirect_manifest,
                args.document_id,
                authorization=args.authorization if args.execute else None,
            )
            if args.receipt:
                write_json(args.receipt, result, pretty=True)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        elif args.command == "record-document-content-revisions":
            from . import wp4
            from .publication import record_document_content_revisions

            src = _wp4_source(args.repo_root, args.source_root)
            result = record_document_content_revisions(
                args.repo_root,
                src,
                args.document_id,
                reason=args.reason,
                authorization=args.authorization,
            )
            objects = wp4.load_sharded_accessions(src)
            document_validation = wp4.validate_document_objects(objects, source_root=src)
            integrity = wp4.referential_integrity(objects)
            if not document_validation["ok"] or not integrity["ok"]:
                raise LibrarianError("document content revision failed catalog validation")
            result = {
                **result,
                "document_validation": document_validation,
                "referential_integrity": integrity,
                "generated_aggregates": wp4.generate_aggregate_ledgers(src, objects),
                "source_manifest": wp4.write_source_manifest(src, source_layout="per_object_shards"),
            }
            if args.receipt:
                write_json(args.receipt, result, pretty=True)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        elif args.command == "publish-knowledge-site":
            from . import wp4

            src = _wp4_source(args.repo_root, args.source_root)
            output = args.output if args.output.is_absolute() else args.repo_root / args.output
            expected_output = (args.repo_root / "docs/000_knowledge-catalog.html").resolve()
            if output.resolve() != expected_output:
                raise LibrarianError(
                    "published Knowledge Catalog output must be "
                    "docs/000_knowledge-catalog.html"
                )
            catalog = wp4.build_catalog(src, args.repo_root.resolve())
            integrity = wp4.referential_integrity(catalog["accessions"])
            document_validation = wp4.validate_document_objects(
                catalog["accessions"], source_root=src
            )
            if not integrity["ok"] or not document_validation["ok"]:
                raise LibrarianError("Knowledge Catalog site publication failed catalog validation")
            dossier_manifests = wp4.build_dossier_manifests(catalog)
            output_sha256 = wp4.write_catalog_html(output, catalog, dossier_manifests)
            result = {
                "schema_version": "eiel_knowledge_catalog_site_publication_v1",
                "output": str(output.resolve()),
                "output_sha256": output_sha256,
                "output_size_bytes": output.stat().st_size,
                "build_id": catalog["build_id"],
                "maintained_source_digest": catalog["maintained_source_digest"],
                "document_count": catalog["object_counts"]["documents"],
                "evidence_record_count": catalog["object_counts"]["records"],
                "reading_surface_only": True,
                "authority_effect": "none",
            }
            if args.receipt:
                write_json(args.receipt, result, pretty=True)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        elif args.command == "verify-published-documents":
            from .publication import DEFAULT_RUNTIME_ROOT, verify_published_document_archive

            result = verify_published_document_archive(
                args.repo_root,
                _wp4_source(args.repo_root, args.source_root),
                runtime_root=args.runtime_root or DEFAULT_RUNTIME_ROOT,
            )
            if args.receipt:
                write_json(args.receipt, result, pretty=True)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        else:
            parser.error(f"unsupported command: {args.command}")
            return 2
    except LibrarianError as exc:
        from . import wp4

        if isinstance(exc, wp4.CliExit):
            payload = {"error": str(exc), **exc.payload}
            print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
            return exc.exit_code
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _wp4_source(repo_root: Path, source_root: Path | None) -> Path:
    source = source_root or Path("tools/evidence/librarian/catalog_source/00-catalog")
    return source if source.is_absolute() else repo_root / source


def _search(catalog_path: Path, query: str) -> dict[str, object]:
    catalog = load_json(catalog_path)
    needle = query.casefold()
    matches = []
    for collection in (
        "artifact_inventory",
        "issues",
        "records",
        "claims",
        "authorities",
        "raw_assets",
    ):
        for row in catalog.get(collection, []):
            if needle in json.dumps(row, sort_keys=True).casefold():
                matches.append({"collection": collection, "row": row})
    return {"query": query, "match_count": len(matches), "matches": matches[:200]}


def _show(catalog_path: Path, identifier: str) -> dict[str, object]:
    catalog = load_json(catalog_path)
    for collection in (
        "artifact_inventory",
        "issues",
        "records",
        "claims",
        "authorities",
        "raw_assets",
    ):
        for row in catalog.get(collection, []):
            ids = {
                str(row.get("artifact_id")),
                str(row.get("issue_id")),
                str(row.get("evidence_id")),
                str(row.get("claim_id")),
                str(row.get("authority_assertion_id")),
                str(row.get("raw_asset_id")),
            }
            if identifier in ids:
                return {"collection": collection, "row": row}
    raise LibrarianError(f"identifier not found: {identifier}")


def _graph(catalog_path: Path, output: Path) -> dict[str, object]:
    catalog = load_json(catalog_path)
    edges = []
    for record in catalog.get("records", []):
        source = record.get("evidence_id")
        for field in (
            "claim_ids",
            "authority_assertion_ids",
            "artifact_ids",
            "raw_asset_ids",
            "derived_from_record_ids",
            "has_part_record_ids",
            "supersedes_record_ids",
            "superseded_by_record_ids",
        ):
            for target in record.get(field, []):
                edges.append({"source": source, "relation": field, "target": target})
    write_json(output, {"schema_version": "eiel_evidence_relation_graph_v1", "edges": edges})
    return {"output": str(output), "edge_count": len(edges)}


if __name__ == "__main__":
    raise SystemExit(main())
