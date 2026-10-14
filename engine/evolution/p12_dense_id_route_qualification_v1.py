"""P12 dense historical biological-ID route qualification receipt.

The receipt binds existing high-ID, checkpoint, WDL-capacity, and EED/RESPEC
cursor evidence without promoting the quarantined CUDA shadow to authority or
claiming final P12 package qualification for unrelated mechanics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final, Mapping

from engine.authoritative_state.schema import content_sha256
from engine.authoritative_state.schema import canonical_value
from engine.evolution.p12_stage2_contracts_v1 import (
    EED_CURSOR_CONTRACT_ID,
    RESPEC_CURSOR_CONTRACT_ID,
    WDL_CAPACITY_CONTRACT_ID,
    cuda_shadow_route_audit_v1,
    cursor_contract_v1,
    dense_identity_route_envelope_v1,
    wdl_capacity_contract_v1,
)


SCHEMA_VERSION: Final[str] = "p12_dense_id_route_qualification_v2"
DECISION: Final[str] = "P12-DENSE-ID-ROUTE-QUALIFIED"
STATUS: Final[str] = "dense_id_route_package_qualified"
CONTRACT_ID: Final[str] = "p12-dense-id-route-qualification-v2"


class P12DenseIDRouteQualificationError(RuntimeError):
    """The dense-ID receipt could not bind expected live source evidence."""


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_evidence(source_root: Path, relative_path: str, fragment: str) -> dict[str, Any]:
    path = source_root / relative_path
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise P12DenseIDRouteQualificationError(
            f"missing dense-ID qualification evidence: {relative_path}"
        ) from exc
    if fragment not in text:
        raise P12DenseIDRouteQualificationError(
            f"dense-ID qualification evidence fragment is absent: {relative_path}"
        )
    line = text[: text.index(fragment)].count("\n") + 1
    return {
        "path": relative_path,
        "sha256": _file_sha256(path),
        "line": line,
        "fragment_sha256": content_sha256(
            {"path": relative_path, "fragment": fragment}
        ),
    }


def _evidence_group(
    *,
    group_id: str,
    status: str,
    conclusion: str,
    evidence: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    return {
        "group_id": group_id,
        "status": status,
        "conclusion": conclusion,
        "evidence": list(evidence),
    }


def build_p12_dense_id_route_qualification_v1(
    *,
    source_root: Path | str = ".",
    source_commit: str | None = None,
) -> dict[str, Any]:
    """Return a hash-bound dense-ID qualification progress receipt."""

    root = Path(source_root).resolve()
    wdl_contract = wdl_capacity_contract_v1()
    eed_contract = cursor_contract_v1(EED_CURSOR_CONTRACT_ID)
    respec_contract = cursor_contract_v1(RESPEC_CURSOR_CONTRACT_ID)
    route_envelope = dense_identity_route_envelope_v1(
        cells=72,
        active_counts=(256, 128, 0) + ((17,) * 69),
        max_stable_biological_id=1_000_000_007,
    )
    try:
        cuda_shadow_audit = cuda_shadow_route_audit_v1(root)
    except OSError as exc:
        raise P12DenseIDRouteQualificationError(
            "dense-ID qualification evidence fragment is absent: cuda shadow audit"
        ) from exc
    groups = (
        _evidence_group(
            group_id="native_high_id_checkpoint_resume",
            status="route_qualified",
            conclusion=(
                "Native CPU compact identity has focused high-ID birth/death, "
                "checkpoint materialization, slot-world authority, and dense "
                "checkpoint rejection evidence."
            ),
            evidence=(
                _source_evidence(
                    root,
                    "tests/perf/test_eiel_cxxbackbone10_resident_state_owner.py",
                    "test_cxxbackbone10r_high_id_birth_and_death_keep_compact_storage_bounded",
                ),
                _source_evidence(
                    root,
                    "tests/perf/test_eiel_cxxbackbone10_resident_state_owner.py",
                    "test_high_id_checkpoint_materializes_slot_world_authority_without_dense_growth",
                ),
                _source_evidence(
                    root,
                    "tests/perf/test_eiel_cxxbackbone10_resident_state_owner.py",
                    "test_cxxbackbone10r_high_id_mate_combat_and_reporting_use_slot_authority",
                ),
                _source_evidence(
                    root,
                    "tests/perf/test_eiel_cxxbackbone10_resident_state_owner.py",
                    "test_cxxbackbone10r_checkpoint_v2_rejects_dense_identity",
                ),
                _source_evidence(
                    root,
                    "tests/perf/test_eiel_cxxbackbone10_resident_state_owner.py",
                    "test_cxxbackbone10r_v1_to_v2_migration_rejects_dense_corruption",
                ),
                _source_evidence(
                    root,
                    "tests/perf/test_eiel_cxxbackbone10_resident_state_owner.py",
                    "test_cxxbackbone16r_native_runtime_checkpoint_restore_matches_uninterrupted",
                ),
            ),
        ),
        _evidence_group(
            group_id="wdl_capacity_contract",
            status="route_qualified",
            conclusion=(
                "WDL compact reporting capacity is resident-capacity based for "
                "b1, b2, b12, b48, and b72 and fails closed on overflow."
            ),
            evidence=(
                _source_evidence(
                    root,
                    "engine/evolution/p12_stage2_contracts_v1.py",
                    "capacity_basis\": \"resident slot capacity, never max stable biological id",
                ),
                _source_evidence(
                    root,
                    "tests/test_p12_stage2_contracts_v1.py",
                    "test_wdl_capacity_projection_scales_by_cells_not_stable_id",
                ),
                _source_evidence(
                    root,
                    "tests/test_eiel_wdl_checkpoint_descriptor_plan.py",
                    "test_wdl_descriptor_plan_is_an_exact_151_descriptor_profile",
                ),
            ),
        ),
        _evidence_group(
            group_id="eed_respec_slot_token_cursors",
            status="route_qualified_reference_boundary",
            conclusion=(
                "EED and RESPEC cursor fixtures are slot plus slot-token owned "
                "and reject stale writes, dead rows, and non-increasing token reuse."
            ),
            evidence=(
                _source_evidence(
                    root,
                    "engine/evolution/p12_stage2_cursor_fixtures_v1.py",
                    "def write_cursor_row_v1",
                ),
                _source_evidence(
                    root,
                    "tests/test_p12_stage2_cursor_fixtures_v1.py",
                    "test_cursor_write_requires_exact_slot_token_owner",
                ),
                _source_evidence(
                    root,
                    "tests/test_p12_stage2_cursor_fixtures_v1.py",
                    "test_reused_slot_starts_empty_and_rejects_non_increasing_token",
                ),
                _source_evidence(
                    root,
                    "tests/test_eiel_eed0_resident_checkpoint_gate_v1.py",
                    "test_checkpoint_gate_covers_two_axes_and_stays_unpromoted",
                ),
            ),
        ),
        _evidence_group(
            group_id="cuda_shadow_disposition",
            status="quarantined_non_authoritative_admitted",
            conclusion=(
                "Current flattened CUDA and WDL routes have zero dense lookup "
                "reads through the compatibility shadow, but the ABI still "
                "exports/materializes a capacity-sized low-ID shadow. Legacy "
                "kernels remain excluded compatibility debt."
            ),
            evidence=(
                _source_evidence(
                    root,
                    "engine/evolution/p12_stage2_contracts_v1.py",
                    "quarantine_non_authoritative_compatibility_projection",
                ),
                _source_evidence(
                    root,
                    "tests/test_p12_stage2_contracts_v1.py",
                    "test_cuda_shadow_route_audit_quarantines_current_and_legacy_use",
                ),
                _source_evidence(
                    root,
                    "tests/test_eiel_cuda_architecture_b_flattened_controller_v1_contract.py",
                    "test_wdl_controller_high_id_identity_authority_matches_base",
                ),
            ),
        ),
        _evidence_group(
            group_id="ragged_multicell_and_sentinel_bounds",
            status="route_qualified",
            conclusion=(
                "The dense-ID route envelope binds ragged multicell active rows "
                "to resident capacity and fails closed on the reserved INT64_MAX "
                "identity sentinel."
            ),
            evidence=(
                _source_evidence(
                    root,
                    "engine/evolution/p12_stage2_contracts_v1.py",
                    "def dense_identity_route_envelope_v1",
                ),
                _source_evidence(
                    root,
                    "tests/test_p12_stage2_contracts_v1.py",
                    "test_dense_identity_route_envelope_scales_by_capacity_for_ragged_cells",
                ),
                _source_evidence(
                    root,
                    "tests/test_p12_stage2_contracts_v1.py",
                    "test_dense_identity_route_envelope_fails_closed_on_int64_sentinel",
                ),
            ),
        ),
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "decision": DECISION,
        "source_commit": source_commit,
        "status": STATUS,
        "dense_id_scaling_final_qualified": True,
        "dense_id_route_package_qualified": True,
        "route_envelope_contract_id": route_envelope["contract_id"],
        "route_envelope_contract_sha256": route_envelope["contract_sha256"],
        "route_envelope": route_envelope,
        "native_high_id_checkpoint_evidence_present": True,
        "wdl_capacity_contract_id": WDL_CAPACITY_CONTRACT_ID,
        "wdl_capacity_contract_sha256": wdl_contract["contract_sha256"],
        "eed_cursor_contract_id": EED_CURSOR_CONTRACT_ID,
        "eed_cursor_contract_sha256": eed_contract["contract_sha256"],
        "respec_cursor_contract_id": RESPEC_CURSOR_CONTRACT_ID,
        "respec_cursor_contract_sha256": respec_contract["contract_sha256"],
        "cuda_shadow_contract_id": cuda_shadow_audit["contract_id"],
        "cuda_shadow_audit_sha256": cuda_shadow_audit["contract_sha256"],
        "cuda_current_route_dense_lookup_reads": (
            cuda_shadow_audit["current_route_dense_lookup_reads"]
        ),
        "cuda_legacy_dense_lookup_debt_files": (
            cuda_shadow_audit["legacy_dense_lookup_debt_files"]
        ),
        "cuda_shadow_retired": False,
        "cuda_shadow_authoritative": False,
        "cuda_shadow_non_authority_admitted": True,
        "final_package_route_qualified": False,
        "final_package_route_qualified_scope": (
            "false means unresolved P12 structural/recurrent/CUDA receipt mechanics; "
            "dense identity route/package evidence is qualified by this receipt"
        ),
        "successor_campaign_authorized": False,
        "evidence_groups": list(groups),
        "remaining_work": [
            "final P12 package and route qualification after all mechanics are present",
        ],
    }
    payload = canonical_value(payload)
    payload["receipt_sha256"] = content_sha256(payload)
    return payload


def canonical_p12_dense_id_route_qualification_bytes_v1(
    receipt: Mapping[str, Any],
) -> bytes:
    """Serialize a dense-ID receipt with stable ASCII JSON ordering."""

    return (
        json.dumps(
            dict(receipt),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
