from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from eiel_composition.execution_identity_projection import (
    SHADOW_PROJECTION_AUTHORITY,
    SHADOW_PROJECTION_INPUT_SCHEMA_VERSION,
    ShadowProjectionError,
    diff_shadow_projections_v1,
    inspect_shadow_projection_v1,
    project_current_execution_v1,
    project_from_input_v1,
    validate_shadow_projection_v1,
    write_shadow_projection_v1,
)
from emergence_contracts import (
    ContentDigest,
    ExperimentId,
    RunId,
    canonical_sha256,
    derive_attempt_id,
    derive_run_id,
)
from engine.campaigns.compiler import compile_campaign, load_campaign_spec
from engine.campaigns.providers import (
    CampaignProviderError,
    default_execution_provider_registry,
)
from engine.qrr.execution_request_v1 import (
    ExecutionRequestDerivationV1,
    build_execution_request_v1,
)
from tools.eiel import main as eiel_main
from eiel_composition.plan_identity_cli import main as plan_main


ROOT = Path(__file__).resolve().parents[2]
SPEC_ROOT = ROOT / "tests/fixtures/public_campaigns"
STALE_BASELINE_ACS_MESSAGE = (
    "native ACS runtime artifact mapping is stale: acs_native_runtime_extension"
)


def _input(
    spec_name: str = "patch_fixed_g14_irregular.yaml",
    *,
    request_kind: str = "ordinary",
) -> dict[str, Any]:
    plan = compile_campaign(load_campaign_spec(SPEC_ROOT / spec_name))
    provider = default_execution_provider_registry().get(plan.profile_id)
    batch = plan.batches[0]
    derivation = None
    if request_kind == "qrr":
        derivation = ExecutionRequestDerivationV1(
            batch_id="qrr-shadow-batch",
            cell_ids=(batch.cell_ids[0],),
            generations=max(1, plan.generations // 2),
            checkpoint_milestones=(0, max(1, plan.generations // 2)),
            reporting_enabled=False,
        )
    bound_value = build_execution_request_v1(
        provider, plan, batch.batch_id, derivation=derivation
    )
    if request_kind == "continuation":
        bound_value = replace(
            bound_value,
            resident_request=replace(
                bound_value.resident_request,
                deliberate_restore_tick=2500,
                deliberate_restore_ticks=(2500,),
            ),
        )
    bound = bound_value.payload()
    return {
        "schema_version": SHADOW_PROJECTION_INPUT_SCHEMA_VERSION,
        "campaign_plan": plan.payload(),
        "bound_execution_request": bound,
        "baseline": {
            "schema_version": "crca_test_baseline_v1",
            "baseline_id": "CURRENT_AUTHORITY_UNCHANGED",
            "qualification_status": "SHADOW_ONLY",
        },
        "source_identity": {
            "source_commit": "1" * 40,
            "source_tree_identity": {"git_tree": "2" * 40},
            "dirty_patch_sha256": None,
            "toolchain": {"python": "fixture"},
            "environment": {"mode": "test"},
            "device_conformance_envelope": {"device": "not-executed"},
        },
        "assurance": {
            "intended_claim_id": "TECHNICAL_ONLY_SHADOW",
            "qualified_envelope": None,
            "obligations": {"no_execution": {"required": True}},
            "authority_references": {"current_compiler": {"unchanged": True}},
            "review_references": {},
            "initial_holds": {"shadow": {"disposition": "HOLD_DIAGNOSTIC"}},
        },
        "cell_or_repeat_ordinal": 0,
        "attempt_ordinal": 1,
        "attempt_nonce": "deterministic-fixture",
        "proposed_reason": "deterministic test projection",
    }


def _known_stale_baseline_acs_error(error: BaseException) -> bool:
    return type(error) is CampaignProviderError and str(error) == (
        STALE_BASELINE_ACS_MESSAGE
    )


def _input_or_xfail_on_known_stale_acs(*args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        return _input(*args, **kwargs)
    except CampaignProviderError as error:
        if not _known_stale_baseline_acs_error(error):
            raise
        pytest.xfail(
            "baseline continuation fixture references an unavailable native ACS "
            "runtime artifact; no semantic assertion was reached"
        )


@pytest.mark.parametrize(
    ("spec_name", "request_kind"),
    [
        ("patch_fixed_g14_irregular.yaml", "ordinary"),
        ("wdl_gru_g37_irregular.yaml", "qrr"),
        (
            "tw_rngv2/science_representative_5000tick_v1.yaml",
            "continuation",
        ),
    ],
)
def test_current_ordinary_continuation_and_qrr_fixtures_reconstruct_exactly(
    spec_name: str, request_kind: str
) -> None:
    source = _input_or_xfail_on_known_stale_acs(
        spec_name, request_kind=request_kind
    )

    first = project_from_input_v1(source)
    second = project_from_input_v1(copy.deepcopy(source))

    assert first == second == validate_shadow_projection_v1(first)
    assert first["authority"] == SHADOW_PROJECTION_AUTHORITY
    assert first["shadow_run_projection"]["occurrence_created"] is False
    assert first["shadow_attempt_projection"]["attempt_created"] is False
    assert first["shadow_attempt_projection"]["execution_side_effect_observed"] is False
    assert diff_shadow_projections_v1(first, second)["decision"] == (
        "IDENTICAL_RECONSTRUCTION"
    )
    restore = first["layers"]["attempt_admission_request"]["payload"][
        "requested_restore_checkpoint_digest"
    ]
    assert (restore is not None) is (request_kind == "continuation")


def test_child_run_parent_occurrence_is_exact_restore_admission_identity() -> None:
    source = _input()
    parent = {
        "schema_version": "crca_workhorse_semantic_checkpoint_occurrence_v2",
        "occurrence_digest_seed": "a" * 64,
    }
    receipt = project_current_execution_v1(
        source["campaign_plan"],
        source["bound_execution_request"],
        source["baseline"],
        source["source_identity"],
        source["assurance"],
        attempt_ordinal=1,
        attempt_nonce="fresh-child-restore",
        parent_checkpoint=parent,
        branch_spec={"branch_law": "closed-workhorse-semantic-child-v2"},
    )
    expected = canonical_sha256(parent).to_dict()

    assert receipt["shadow_run_projection"]["candidate"][
        "parent_checkpoint_digest"
    ] == expected
    assert receipt["layers"]["attempt_admission_request"]["payload"][
        "requested_restore_checkpoint_digest"
    ] == expected
    assert receipt["shadow_attempt_projection"]["monotonic_ordinal"] == 1

    continuation = copy.deepcopy(source)
    resident = continuation["bound_execution_request"]["resident_request"]
    resident["deliberate_restore_tick"] = 1
    resident["deliberate_restore_ticks"] = [1]
    bound = continuation["bound_execution_request"]
    bound["bound_request_sha256"] = canonical_sha256(
        {
            key: item
            for key, item in bound.items()
            if key not in {"bound_request_sha256", "execution_route_contract"}
        }
    ).hex
    with pytest.raises(ShadowProjectionError, match="child checkpoint identity"):
        project_current_execution_v1(
            continuation["campaign_plan"],
            continuation["bound_execution_request"],
            continuation["baseline"],
            continuation["source_identity"],
            continuation["assurance"],
            parent_checkpoint=parent,
        )


def test_stale_acs_quarantine_matches_only_the_exact_provider_error() -> None:
    assert _known_stale_baseline_acs_error(
        CampaignProviderError(STALE_BASELINE_ACS_MESSAGE)
    )
    assert not _known_stale_baseline_acs_error(
        CampaignProviderError("native ACS runtime artifact mapping is invalid")
    )
    assert not _known_stale_baseline_acs_error(
        ValueError(STALE_BASELINE_ACS_MESSAGE)
    )


def _keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [str(key) for key in value] + [
            child for item in value.values() for child in _keys(item)
        ]
    if isinstance(value, list):
        return [child for item in value for child in _keys(item)]
    return []


def test_tick_native_shadow_contract_has_no_generation_shaped_authority() -> None:
    receipt = project_from_input_v1(
        _input_or_xfail_on_known_stale_acs(
            "tw_rngv2/science_representative_5000tick_v1.yaml"
        )
    )
    semantic = receipt["layers"]["semantic"]["payload"]
    prohibited = {
        "generation",
        "generations",
        "ticks_per_generation",
        "physical_window_generations",
    }

    assert prohibited.isdisjoint(_keys(semantic))
    assert semantic["absolute_start_next_tick"] == 0
    assert semantic["absolute_target_next_tick"] == 5000
    assert semantic["trajectory_checkpoint_ticks"] == [0, 2500, 5000]


def test_legacy_fixture_is_privately_projected_to_absolute_ticks() -> None:
    receipt = project_from_input_v1(_input())
    plan = _input()["campaign_plan"]
    semantic = receipt["layers"]["semantic"]["payload"]

    assert semantic["absolute_target_next_tick"] == (
        plan["execution"]["generations"] * plan["execution"]["ticks_per_generation"]
    )
    assert all(type(tick) is int for tick in semantic["trajectory_checkpoint_ticks"])


def test_diff_classifies_semantic_implementation_and_baseline_changes() -> None:
    ordinary = project_from_input_v1(_input())
    continuation = project_from_input_v1(_input("wdl_gru_g37_irregular.yaml"))
    semantic_diff = diff_shadow_projections_v1(ordinary, continuation)
    assert semantic_diff["decision"] == "SUCCESSOR_OR_SIBLING_RUN_REQUIRED"
    assert semantic_diff["semantic_run_id_preserved"] is False

    implementation_input = _input()
    implementation_input["source_identity"]["source_commit"] = "3" * 40
    implementation = project_from_input_v1(implementation_input)
    implementation_diff = diff_shadow_projections_v1(ordinary, implementation)
    assert implementation_diff["decision"] == (
        "SAME_RUN_NEW_ATTEMPT_REQUIRES_EXACT_CONFORMANCE"
    )
    assert implementation_diff["semantic_run_id_preserved"] is True

    baseline_input = _input()
    baseline_input["baseline"]["baseline_id"] = "CHANGED_LOCK"
    baseline = project_from_input_v1(baseline_input)
    assert diff_shadow_projections_v1(ordinary, baseline)["decision"] == (
        "PREADMISSION_REEVALUATION_NO_ATTEMPT"
    )


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("authority",), "EXECUTE", "authority marker"),
        (("shadow_run_projection", "occurrence_created"), True, "cannot create"),
        (("shadow_attempt_projection", "attempt_created"), True, "runtime Attempt"),
        (("layers", "semantic", "payload", "absolute_target_next_tick"), 99, "invalid"),
    ],
)
def test_tampered_or_false_authority_receipt_fails_closed(
    path: tuple[str, ...], value: Any, match: str
) -> None:
    receipt = project_from_input_v1(_input())
    target: dict[str, Any] = receipt
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    if path != ("authority",):
        receipt["projection_sha256"] = canonical_sha256(
            {key: item for key, item in receipt.items() if key != "projection_sha256"}
        ).hex

    with pytest.raises(ShadowProjectionError, match=match):
        validate_shadow_projection_v1(receipt)


def test_fully_rehashed_generation_shaped_semantic_tamper_fails_closed() -> None:
    receipt = project_from_input_v1(
        _input_or_xfail_on_known_stale_acs(
            "tw_rngv2/science_representative_5000tick_v1.yaml"
        )
    )
    semantic_row = receipt["layers"]["semantic"]
    semantic_row["payload"]["generations"] = 1
    semantic_digest = canonical_sha256(semantic_row["payload"])
    semantic_row["digest"] = semantic_digest.to_dict()

    concretized_row = receipt["layers"]["concretized_execution"]
    concretized_row["payload"]["semantic_spec_digest"] = semantic_digest.to_dict()
    concretized_digest = canonical_sha256(concretized_row["payload"])
    concretized_row["digest"] = concretized_digest.to_dict()

    run = receipt["shadow_run_projection"]["candidate"]
    run["semantic_spec_digest"] = semantic_digest.to_dict()
    expected_run = derive_run_id(
        experiment_id=ExperimentId(run["experiment_id"]),
        cell_or_repeat_ordinal=run["cell_or_repeat_ordinal"],
        semantic_spec_digest=semantic_digest,
        parent_checkpoint_digest=(
            None
            if run["parent_checkpoint_digest"] is None
            else ContentDigest(**run["parent_checkpoint_digest"])
        ),
        branch_spec_digest=(
            None
            if run["branch_spec_digest"] is None
            else ContentDigest(**run["branch_spec_digest"])
        ),
    )
    run["run_id"] = str(expected_run)
    attempt = receipt["shadow_attempt_projection"]
    attempt["run_id"] = str(expected_run)
    attempt["candidate_attempt_id"] = str(
        derive_attempt_id(
            run_id=RunId(str(expected_run)),
            monotonic_ordinal=attempt["monotonic_ordinal"],
            nonce=attempt["nonce"],
        )
    )
    attempt["concretized_execution_digest"] = concretized_digest.to_dict()
    receipt["projection_sha256"] = canonical_sha256(
        {key: item for key, item in receipt.items() if key != "projection_sha256"}
    ).hex

    with pytest.raises(ShadowProjectionError, match="unsupported fields"):
        validate_shadow_projection_v1(receipt)


def test_unknown_non_null_execution_route_is_not_treated_as_legacy() -> None:
    source = _input()
    plan = source["campaign_plan"]
    plan["execution"]["execution_route_identity"] = "unknown_route_v1"
    plan["plan_sha256"] = canonical_sha256(
        {key: item for key, item in plan.items() if key != "plan_sha256"}
    ).hex
    bound = source["bound_execution_request"]
    bound["plan_sha256"] = plan["plan_sha256"]
    bound["bound_request_sha256"] = canonical_sha256(
        {
            key: item
            for key, item in bound.items()
            if key not in {"bound_request_sha256", "execution_route_contract"}
        }
    ).hex

    with pytest.raises(
        ShadowProjectionError, match="unsupported campaign execution route"
    ):
        project_from_input_v1(source)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("plan_sha256", "0" * 64, "plan_sha256"),
        ("qualification_key_sha256", "0" * 64, "qualification key"),
        ("execution_route_sha256", "0" * 64, "route digest"),
    ],
)
def test_current_lock_mismatch_fails_closed(field: str, value: str, match: str) -> None:
    source = _input()
    if field == "plan_sha256":
        source["campaign_plan"][field] = value
    else:
        bound = source["bound_execution_request"]
        bound[field] = value
        bound["bound_request_sha256"] = canonical_sha256(
            {
                key: item
                for key, item in bound.items()
                if key not in {"bound_request_sha256", "execution_route_contract"}
            }
        ).hex

    with pytest.raises(ShadowProjectionError, match=match):
        project_from_input_v1(source)


def test_writer_inspection_and_cli_round_trip(tmp_path: Path, capsys: Any) -> None:
    input_path = tmp_path / "input.json"
    receipt_path = tmp_path / "receipt.json"
    input_path.write_text(json.dumps(_input()), encoding="utf-8")

    assert plan_main(["project", str(input_path), str(receipt_path)]) == 0
    projected = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert inspect_shadow_projection_v1(projected)["status"] == (
        "VALID_SHADOW_PROJECTION"
    )
    assert write_shadow_projection_v1(tmp_path / "copy.json", projected).exists()
    assert plan_main(["inspect", str(receipt_path), "--human"]) == 0
    assert "Authority: shadow only" in capsys.readouterr().out
    assert eiel_main(["plan", "diff", str(receipt_path), str(receipt_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["decision"] == "IDENTICAL_RECONSTRUCTION"


def test_cli_returns_two_for_invalid_receipt(tmp_path: Path, capsys: Any) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("{}", encoding="utf-8")

    assert plan_main(["inspect", str(path)]) == 2
    assert "unsupported shadow projection schema" in capsys.readouterr().err
