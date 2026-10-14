"""Retained read-only dispatch for durable operator-tooling contracts.

The registry does not migrate or normalize unknown data.  It distinguishes a
known historical reader from an unsupported future format before any current
constructor can apply defaults.  Imports are lazy to keep actor modules free of
cycles and to make deletion of an optional subsystem fail locally.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from tools.actors.common import ActorError
from tools.actors.contract_versions import WORKER_RESULT_SCHEMA_VERSION

# (family, schema version) -> "module:function".  This is executable dispatch,
# not merely an inventory of schema filenames.
READER_REGISTRY: dict[tuple[str, str], str] = {
    ("actor_record", "eiel_actor_record_envelope_v1"): "tools.actors.common:load_actor_record",
    ("work_item", "eiel_work_item_v1"): "tools.actors.work_manager:load_work_payload",
    ("codex_session", "eiel_codex_session_v1"): "tools.actors.session_manager:load_session_payload",
    ("work_package", "eiel_work_package_v1"): "tools.actors.work_package:load_work_package_value",
    ("work_package_binding", "eiel_work_package_binding_v1"): "tools.actors.work_package:load_work_package_binding",
    ("work_package_receipt", "eiel_work_package_contract_receipt_v1"): "tools.actors.work_package:load_work_package_contract_receipt",
    ("workspace", "eiel_codex_workspace_v1"): "tools.actors.workspace:load_workspace_artifact",
    ("workspace", "eiel_codex_workspace_v2"): "tools.actors.workspace:load_workspace_artifact",
    ("workspace_audit", "eiel_codex_workspace_audit_v1"): "tools.actors.workspace:load_workspace_audit_artifact",
    ("review_evidence", "eiel_review_evidence_v2"): "tools.actors.session_manager:load_review_evidence",
    ("review_evidence", "eiel_review_evidence_v3"): "tools.actors.session_manager:load_review_evidence",
    ("review_evidence", "eiel_review_evidence_v4"): "tools.actors.session_manager:load_review_evidence",
    ("review_target", "eiel_review_target_v1"): "tools.actors.session_manager:load_review_target",
    ("review_target", "eiel_review_target_v2"): "tools.actors.session_manager:load_review_target",
    ("delivery_evidence", "eiel_session_delivery_evidence_v2"): "tools.actors.session_manager:load_session_delivery_evidence",
    ("delivery_evidence", "eiel_session_delivery_evidence_v3"): "tools.actors.session_manager:load_session_delivery_evidence",
    ("delivery_evidence", "eiel_session_delivery_evidence_v4"): "tools.actors.session_manager:load_session_delivery_evidence",
    ("lifecycle_decision_snapshot", "eiel_lifecycle_decision_snapshot_v1"): "tools.actors.lifecycle_policy:load_lifecycle_decision_snapshot",
    ("lifecycle_decision_snapshot", "eiel_lifecycle_decision_snapshot_v2"): "tools.actors.lifecycle_policy:load_lifecycle_decision_snapshot",
    ("context_receipt", "eiel_context_receipt_v1"): "tools.actors.context_receipt:load_context_receipt",
    ("repository_bootstrap", "eiel_repository_bootstrap_v1"): "tools.actors.bootstrap:load_bootstrap_payload",
    ("token_snapshot", "eiel_token_snapshot_v1"): "tools.actors.resource_accounting:load_token_snapshot",
    ("session_token_usage", "eiel_session_token_usage_v1"): "tools.actors.resource_accounting:load_session_usage_artifact",
    ("work_token_usage", "eiel_token_usage_v1"): "tools.actors.resource_accounting:load_work_usage_artifact",
    ("token_report", "eiel_token_report_index_v1"): "tools.actors.resource_accounting:load_token_report_index",
    ("dispatch_request", "eiel_dispatch_request_v1"): "tools.scheduler.contracts:load_dispatch_payload",
    ("worker_assignment", "eiel_worker_assignment_v1"): "tools.scheduler.contracts:load_assignment_payload",
    ("assignment_lease", "eiel_assignment_lease_v1"): "tools.scheduler.contracts:load_lease_payload",
    ("worker_result", WORKER_RESULT_SCHEMA_VERSION): "tools.scheduler.contracts:load_worker_result_payload",
}


def registered_versions(family: str) -> tuple[str, ...]:
    return tuple(sorted(version for (name, version) in READER_REGISTRY if name == family))


def _resolve(target: str) -> Callable[..., Any]:
    import importlib

    module_name, function_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise ActorError(
            f"retained schema reader is unavailable: {target}",
            code="schema_reader_unavailable",
        )
    return function


def load_registered(family: str, raw: Any, **kwargs: Any) -> Any:
    if not isinstance(raw, Mapping):
        raise ActorError(f"{family} must be an object", code="schema_record_malformed")
    version = raw.get("schema_version")
    if not isinstance(version, str) or not version.strip():
        raise ActorError(f"{family} schema_version is missing", code="schema_version_missing")
    target = READER_REGISTRY.get((family, version.strip()))
    if target is None:
        supported = registered_versions(family)
        raise ActorError(
            f"unsupported {family} schema_version: {version}",
            code="schema_version_unsupported",
            problems=["Supported schema versions: " + ", ".join(supported)] if supported else [],
        )
    return _resolve(target)(raw, **kwargs)


def registry_snapshot() -> dict[str, Any]:
    return {
        "schema_version": "eiel_schema_reader_registry_v1",
        "readers": [
            {"family": family, "schema_version": version, "reader": target}
            for (family, version), target in sorted(READER_REGISTRY.items())
        ],
    }


__all__ = ["READER_REGISTRY", "load_registered", "registered_versions", "registry_snapshot"]
