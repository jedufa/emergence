"""Frozen persisted-contract and semantic-vocabulary identifiers.

This module is deliberately dependency-free.  A schema tag without a retained
reader or a vocabulary identifier is decorative metadata, which is a charming
human custom and a terrible recovery strategy.  Current constructors import
these identifiers; historical readers keep their own frozen tables here.
"""

from __future__ import annotations

ACTOR_RECORD_SCHEMA_VERSION = "eiel_actor_record_envelope_v1"
WORK_SCHEMA_VERSION = "eiel_work_item_v1"
SESSION_SCHEMA_VERSION = "eiel_codex_session_v1"
WORK_PACKAGE_SCHEMA_VERSION = "eiel_work_package_v1"
WORK_PACKAGE_BINDING_SCHEMA_VERSION = "eiel_work_package_binding_v1"
WORK_PACKAGE_RECEIPT_SCHEMA_VERSION = "eiel_work_package_contract_receipt_v1"
WORKSPACE_SCHEMA_VERSION = "eiel_codex_workspace_v2"
WORKSPACE_LEGACY_SCHEMA_VERSION = "eiel_codex_workspace_v1"

OPERATOR_VOCABULARY_VERSION = "eiel_operator_vocabulary_v1_18"
SCHEDULER_VOCABULARY_VERSION = "eiel_scheduler_vocabulary_v1_18"

WORK_LIFECYCLE_PROJECTION_VERSION = "eiel_work_lifecycle_projection_v2"
SESSION_ATTEMPT_PROJECTION_VERSION = "eiel_session_attempt_projection_v2"
SCHEDULER_ASSIGNMENT_PROJECTION_VERSION = "eiel_scheduler_assignment_projection_v2"
ARCHITECTURE_SIMPLIFICATION_CONTRACT_VERSION = (
    "eiel_operator_architecture_simplification_v1_18j"
)

WORK_STATUSES_V1_18 = (
    "BACKLOG", "READY", "ACTIVE", "PAUSED", "BLOCKED", "COMPLETED",
    "DISCONTINUED", "OBSOLETE", "SUPERSEDED",
)
SESSION_STATUSES_V1_18 = (
    "CREATED", "ACTIVE", "PAUSED", "INTERRUPTED", "BLOCKED", "COMPLETED",
    "DISCONTINUED", "OBSOLETE", "SUPERSEDED",
)
SESSION_OUTCOMES_V1_18 = (
    "ARCHITECTURE_COMPLETE", "ENGINEERING_COMPLETE", "IMPLEMENTATION_COMPLETE",
    "DELIVERY_RECORDED", "EXTERNAL_DELIVERY_ADOPTED", "REVIEW_COMPLETE",
    "PROMOTION_REQUESTED", "SESSION_COMPLETE",
)
AUTHORITY_ROLES_V1_18 = (
    "ARCHITECTURE_AUTHORITY", "ENGINEERING_AUTHORITY",
    "IMPLEMENTATION_AUTHORITY", "REVIEW_AUTHORITY",
)
REVIEW_VERDICTS_V1_18 = ("ACCEPT", "CHANGES_REQUIRED", "REJECT", "PROMOTE", "BLOCKED")

WORKER_RESULT_SCHEMA_VERSION = "eiel_worker_result_v1"

__all__ = [name for name in globals() if name.isupper()]
