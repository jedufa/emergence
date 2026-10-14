from __future__ import annotations

from dataclasses import replace

import pytest

from engine.campaigns.mixed_mode_resident_isolation_gate import (
    RUN_KIND_ALONE,
    RUN_KIND_CHECKPOINT_BASELINE,
    RUN_KIND_CHECKPOINT_FRESH_OWNER,
    RUN_KIND_DUPLICATE,
    RUN_KIND_MIXED,
    RUN_KIND_OVERFLOW,
    RUN_KIND_PERMUTED,
    CheckpointCaseV1,
    FrozenMinimumInheritanceExposureV1,
    InheritanceAttemptV1,
    IsolationRunCaptureV1,
    MixedModeIsolationContractError,
    MixedModeIsolationPlanV1,
    OverflowCaseV1,
    ParentSelectionContractV1,
    ReproductionPathReceiptV1,
    ResidentCellCaptureV1,
    ResidentReproductionCellConfigV1,
    verify_mixed_mode_resident_isolation_v1,
)


REFILL = "generation_boundary_refill_v1"
LIVE = "ordinary_live_birth_v1"
ZERO = "0" * 64
ONE = "1" * 64
TWO = "2" * 64
THREE = "3" * 64
FOUR = "4" * 64
FIVE = "5" * 64


def _selection() -> ParentSelectionContractV1:
    return ParentSelectionContractV1(
        selection_mode="uniform_eligible_parent_v1",
        eligible_parent_source="authoritative_eligible_parent_set_v1",
        draw_distribution="uniform_v1",
        deterministic=True,
        blinded_dimensions=("genotype", "lineage", "policy", "strategy"),
    )


def _attempt(success: bool = True) -> InheritanceAttemptV1:
    return InheritanceAttemptV1(
        primary_eligible_parent_ids=("p0", "p1", "p2"),
        mate_eligible_parent_ids=("p1", "p2"),
        selected_parent_ids=("p0", "p1"),
        success=success,
        contributing_parent_ids=("p0", "p1") if success else (),
        failure_code=None if success else "capacity_v1",
    )


def _path_receipts(
    active: str, *, births: int = 2
) -> tuple[ReproductionPathReceiptV1, ...]:
    values = []
    for path in (REFILL, LIVE):
        enabled = path == active
        attempts = tuple(_attempt() for _ in range(births)) if enabled else ()
        values.append(
            ReproductionPathReceiptV1(
                path_id=path,
                enabled=enabled,
                rng_counters=(("parent_selection", births * 2),) if enabled else (),
                donor_draw_ordinals=tuple(range(births * 7)) if enabled else (),
                counter_writes=births if enabled else 0,
                sidecar_writes=births if enabled else 0,
                event_row_writes=births if enabled else 0,
                rng_state_writes=births * 2 if enabled else 0,
                attempts=attempts,
            )
        )
    return tuple(values)


def _cell(
    logical_id: str,
    path: str,
    position: int,
    *,
    occurrence_id: str | None = None,
    equivalence_key: str | None = None,
    neighbors: tuple[str, ...] = (),
    digest: str = ONE,
    births: int = 2,
) -> ResidentCellCaptureV1:
    return ResidentCellCaptureV1(
        occurrence_id=occurrence_id or logical_id,
        config=ResidentReproductionCellConfigV1(
            logical_cell_id=logical_id,
            equivalence_key=equivalence_key or logical_id,
            reproduction_path=path,
            selection=_selection(),
            parent_arity=2,
            donor_draws_per_successful_birth=7,
            authorized_rng_counter_ids=("parent_selection",),
        ),
        batch_position=position,
        neighbor_path_ids=neighbors,
        final_state_sha256=digest,
        pre_fault_state_sha256=digest,
        checkpoint_sidecar_sha256=TWO,
        lifecycle_state_sha256=THREE,
        final_authority_sha256=digest,
        pre_fault_authority_sha256=None,
        ok=True,
        fail_closed=False,
        error_code=None,
        path_receipts=_path_receipts(path, births=births),
    )


def _run(
    run_id: str,
    kind: str,
    cells: tuple[ResidentCellCaptureV1, ...],
    plan_digest: str,
    *,
    fresh_owner: bool = False,
) -> IsolationRunCaptureV1:
    return IsolationRunCaptureV1(
        run_id=run_id,
        run_kind=kind,
        plan_sha256=plan_digest,
        captured_at_utc="2026-07-23T23:00:00+00:00",
        fresh_owner=fresh_owner,
        cells=cells,
    )


def _plan() -> MixedModeIsolationPlanV1:
    return MixedModeIsolationPlanV1(
        path_ids=(REFILL, LIVE),
        mixed_reference_run_id="mixed",
        permutation_run_ids=("permuted",),
        duplicate_run_ids=("duplicates",),
        duplicate_required_path_ids=(REFILL, LIVE),
        overflow_cases=(
            OverflowCaseV1(
                reference_run_id="overflow-reference",
                fault_run_id="overflow",
                target_occurrence_id="live",
                overflow_error_code="event_capacity_overflow_v1",
            ),
        ),
        checkpoint_cases=(
            CheckpointCaseV1(
                uninterrupted_run_id="checkpoint-baseline",
                fresh_owner_run_id="checkpoint-resumed",
            ),
        ),
        minimum_live_birth_exposure=FrozenMinimumInheritanceExposureV1(
            threshold_id="usable-inheritance-events-v1",
            metric="successful_births",
            minimum=2,
            frozen_at_utc="2026-07-23T22:00:00+00:00",
        ),
        live_birth_path_ids=(LIVE,),
    )


def _passing_fixture():
    plan = _plan()
    digest = plan.digest_sha256
    refill = _cell("refill", REFILL, 0, neighbors=(LIVE,))
    live = _cell("live", LIVE, 1, neighbors=(REFILL,))
    mixed = _run("mixed", RUN_KIND_MIXED, (refill, live), digest)
    alone_refill = _run(
        "alone-refill",
        RUN_KIND_ALONE,
        (replace(refill, batch_position=0, neighbor_path_ids=()),),
        digest,
    )
    alone_live = _run(
        "alone-live",
        RUN_KIND_ALONE,
        (replace(live, batch_position=0, neighbor_path_ids=()),),
        digest,
    )
    permuted = _run(
        "permuted",
        RUN_KIND_PERMUTED,
        (
            replace(live, batch_position=0, neighbor_path_ids=(REFILL,)),
            replace(refill, batch_position=1, neighbor_path_ids=(LIVE,)),
        ),
        digest,
    )
    dup_refill_a = _cell(
        "dup-r-a",
        REFILL,
        0,
        occurrence_id="dup-r-a",
        equivalence_key="same-r",
        neighbors=(LIVE,),
        digest=FOUR,
    )
    dup_refill_b = _cell(
        "dup-r-b",
        REFILL,
        2,
        occurrence_id="dup-r-b",
        equivalence_key="same-r",
        neighbors=(LIVE, LIVE),
        digest=FOUR,
    )
    dup_live_a = _cell(
        "dup-l-a",
        LIVE,
        1,
        occurrence_id="dup-l-a",
        equivalence_key="same-l",
        neighbors=(REFILL, REFILL),
        digest=FIVE,
    )
    dup_live_b = _cell(
        "dup-l-b",
        LIVE,
        3,
        occurrence_id="dup-l-b",
        equivalence_key="same-l",
        neighbors=(REFILL,),
        digest=FIVE,
    )
    duplicates = _run(
        "duplicates",
        RUN_KIND_DUPLICATE,
        (dup_refill_a, dup_live_a, dup_refill_b, dup_live_b),
        digest,
    )
    overflow_reference = _run(
        "overflow-reference", RUN_KIND_MIXED, (refill, live), digest
    )
    failed_live = replace(
        live,
        final_state_sha256=live.pre_fault_state_sha256,
        pre_fault_authority_sha256=live.final_authority_sha256,
        ok=False,
        fail_closed=True,
        error_code="event_capacity_overflow_v1",
    )
    overflow = _run("overflow", RUN_KIND_OVERFLOW, (refill, failed_live), digest)
    checkpoint_baseline = _run(
        "checkpoint-baseline",
        RUN_KIND_CHECKPOINT_BASELINE,
        (refill, live),
        digest,
    )
    checkpoint_resumed = _run(
        "checkpoint-resumed",
        RUN_KIND_CHECKPOINT_FRESH_OWNER,
        (refill, live),
        digest,
        fresh_owner=True,
    )
    return plan, [
        mixed,
        alone_refill,
        alone_live,
        permuted,
        duplicates,
        overflow_reference,
        overflow,
        checkpoint_baseline,
        checkpoint_resumed,
    ]


def _check(result, check_id: str):
    return next(check for check in result.checks if check.check_id == check_id)


def _replace_run(runs, run_id, replacement):
    return [replacement if run.run_id == run_id else run for run in runs]


def test_complete_nine_obligation_gate_passes_and_exposes_versioned_payload() -> None:
    plan, runs = _passing_fixture()
    result = verify_mixed_mode_resident_isolation_v1(plan, runs)

    assert result.ok is True
    assert len(result.checks) == 9
    assert result.payload()["schema_version"] == (
        "eiel_mixed_mode_resident_isolation_gate_v2"
    )
    assert runs[0].payload()["schema_version"] == (
        "eiel_mixed_mode_resident_isolation_run_v2"
    )
    assert runs[0].payload()["cells"][0]["schema_version"] == (
        "eiel_mixed_mode_resident_cell_capture_v2"
    )
    assert result.live_birth_evaluability["live"] == {
        "path_id": LIVE,
        "metric": "successful_births",
        "observed": 2,
        "minimum": 2,
        "classification": "evaluable",
        "negative_scientific_evidence_allowed": True,
    }


def test_alone_mismatch_fails_closed_with_digest_diagnostic() -> None:
    plan, runs = _passing_fixture()
    alone = next(run for run in runs if run.run_id == "alone-live")
    bad_cell = replace(alone.cells[0], lifecycle_state_sha256=ZERO)
    runs = _replace_run(runs, alone.run_id, replace(alone, cells=(bad_cell,)))

    check = _check(
        verify_mixed_mode_resident_isolation_v1(plan, runs),
        "cell_alone_equivalence",
    )
    assert check.ok is False
    assert check.diagnostics[0].code == "alone.cell_mismatch"


def test_transient_observer_identity_is_not_mechanics_equivalence() -> None:
    plan, runs = _passing_fixture()
    alone = next(run for run in runs if run.run_id == "alone-live")
    transient = replace(
        alone.cells[0],
        final_state_sha256=ZERO,
        final_authority_sha256=FOUR,
    )
    runs = _replace_run(
        runs,
        alone.run_id,
        replace(alone, cells=(transient,)),
    )

    check = _check(
        verify_mixed_mode_resident_isolation_v1(plan, runs),
        "cell_alone_equivalence",
    )
    assert check.ok is True


def test_permutation_must_reorder_and_remain_exact() -> None:
    plan, runs = _passing_fixture()
    permuted = next(run for run in runs if run.run_id == "permuted")
    unchanged_order = tuple(
        replace(cell, batch_position=index)
        for index, cell in enumerate(
            next(run for run in runs if run.run_id == "mixed").cells
        )
    )
    runs = _replace_run(runs, permuted.run_id, replace(permuted, cells=unchanged_order))

    check = _check(
        verify_mixed_mode_resident_isolation_v1(plan, runs),
        "batch_permutation_invariance",
    )
    assert check.ok is False
    assert check.diagnostics[0].code == "permutation.not_permuted"


def test_duplicate_cells_require_different_neighbors_and_exact_receipts() -> None:
    plan, runs = _passing_fixture()
    duplicate = next(run for run in runs if run.run_id == "duplicates")
    cells = list(duplicate.cells)
    cells[2] = replace(
        cells[2],
        batch_position=3,
        neighbor_path_ids=(LIVE,),
    )
    cells[3] = replace(
        cells[3],
        batch_position=2,
        neighbor_path_ids=(LIVE, REFILL),
    )
    runs = _replace_run(runs, duplicate.run_id, replace(duplicate, cells=tuple(cells)))

    check = _check(
        verify_mixed_mode_resident_isolation_v1(plan, runs),
        "duplicate_cell_isolation",
    )
    assert check.ok is False
    assert any(
        diagnostic.code == "duplicate.neighborhood_not_varied"
        for diagnostic in check.diagnostics
    )


def test_duplicate_path_coverage_unions_two_b2_runs_and_mixed_reference() -> None:
    plan, runs = _passing_fixture()
    mixed = next(run for run in runs if run.run_id == "mixed")
    refill, live = mixed.cells
    plan = replace(
        plan,
        duplicate_run_ids=("duplicates-refill", "duplicates-live"),
    )
    digest = plan.digest_sha256
    runs = [
        replace(run, plan_sha256=digest)
        for run in runs
        if run.run_id != "duplicates"
    ]
    runs.extend(
        (
            _run(
                "duplicates-refill",
                RUN_KIND_DUPLICATE,
                (
                    replace(
                        refill,
                        occurrence_id="refill-duplicate-1",
                        batch_position=0,
                        neighbor_path_ids=(REFILL,),
                    ),
                    replace(
                        refill,
                        occurrence_id="refill-duplicate-2",
                        batch_position=1,
                        neighbor_path_ids=(REFILL,),
                    ),
                ),
                digest,
            ),
            _run(
                "duplicates-live",
                RUN_KIND_DUPLICATE,
                (
                    replace(
                        live,
                        occurrence_id="live-duplicate-1",
                        batch_position=0,
                        neighbor_path_ids=(LIVE,),
                    ),
                    replace(
                        live,
                        occurrence_id="live-duplicate-2",
                        batch_position=1,
                        neighbor_path_ids=(LIVE,),
                    ),
                ),
                digest,
            ),
        )
    )

    check = _check(
        verify_mixed_mode_resident_isolation_v1(plan, runs),
        "duplicate_cell_isolation",
    )
    assert check.ok is True

    missing_live = [
        run for run in runs if run.run_id != "duplicates-live"
    ]
    check = _check(
        verify_mixed_mode_resident_isolation_v1(plan, missing_live),
        "duplicate_cell_isolation",
    )
    assert {diagnostic.code for diagnostic in check.diagnostics} == {
        "duplicate.missing",
        "duplicate.path_coverage",
    }


def test_rng_namespaces_and_donor_budget_are_path_local() -> None:
    plan, runs = _passing_fixture()
    mixed = next(run for run in runs if run.run_id == "mixed")
    live = mixed.cells[1]
    receipts = list(live.path_receipts)
    active_index = next(
        index for index, receipt in enumerate(receipts) if receipt.enabled
    )
    receipts[active_index] = replace(
        receipts[active_index],
        rng_counters=(
            ("parent_selection", 4),
            ("neighbor_global_cursor", 1),
        ),
        donor_draw_ordinals=receipts[active_index].donor_draw_ordinals[:-1],
    )
    bad = replace(live, path_receipts=tuple(receipts))
    runs = _replace_run(runs, mixed.run_id, replace(mixed, cells=(mixed.cells[0], bad)))

    check = _check(
        verify_mixed_mode_resident_isolation_v1(plan, runs),
        "rng_and_donor_draw_isolation",
    )
    codes = {diagnostic.code for diagnostic in check.diagnostics}
    assert {"rng.unauthorized_counter", "rng.donor_draw_budget"} <= codes


def test_overflow_target_must_fail_closed_and_peers_must_not_change() -> None:
    plan, runs = _passing_fixture()
    overflow = next(run for run in runs if run.run_id == "overflow")
    changed_peer = replace(overflow.cells[0], lifecycle_state_sha256=ZERO)
    authority_leak_target = replace(
        overflow.cells[1],
        final_authority_sha256=FOUR,
    )
    runs = _replace_run(
        runs,
        overflow.run_id,
        replace(overflow, cells=(changed_peer, authority_leak_target)),
    )

    check = _check(
        verify_mixed_mode_resident_isolation_v1(plan, runs),
        "overflow_isolation",
    )
    codes = {diagnostic.code for diagnostic in check.diagnostics}
    assert {
        "overflow.target_not_fail_closed",
        "overflow.peer_changed",
    } <= codes


def test_checkpoint_comparison_requires_real_fresh_owner_and_exact_cells() -> None:
    plan, runs = _passing_fixture()
    resumed = next(run for run in runs if run.run_id == "checkpoint-resumed")
    changed = replace(resumed.cells[1], lifecycle_state_sha256=ZERO)
    runs = _replace_run(
        runs,
        resumed.run_id,
        replace(resumed, fresh_owner=False, cells=(resumed.cells[0], changed)),
    )

    check = _check(
        verify_mixed_mode_resident_isolation_v1(plan, runs),
        "fresh_owner_checkpoint_isolation",
    )
    codes = {diagnostic.code for diagnostic in check.diagnostics}
    assert {"checkpoint.owner_boundary", "checkpoint.cell_mismatch"} <= codes


def test_disabled_path_cannot_write_or_consume_rng() -> None:
    plan, runs = _passing_fixture()
    mixed = next(run for run in runs if run.run_id == "mixed")
    live = mixed.cells[1]
    receipts = list(live.path_receipts)
    disabled_index = next(
        index for index, receipt in enumerate(receipts) if not receipt.enabled
    )
    receipts[disabled_index] = replace(
        receipts[disabled_index],
        rng_counters=(("parent_selection", 1),),
        sidecar_writes=1,
    )
    runs = _replace_run(
        runs,
        mixed.run_id,
        replace(
            mixed,
            cells=(
                mixed.cells[0],
                replace(live, path_receipts=tuple(receipts)),
            ),
        ),
    )

    check = _check(
        verify_mixed_mode_resident_isolation_v1(plan, runs),
        "disabled_path_write_isolation",
    )
    assert check.ok is False
    assert check.diagnostics[0].code == "disabled.path_wrote"


def test_event_exposure_requires_each_registered_path_in_reference() -> None:
    plan, runs = _passing_fixture()
    mixed = next(run for run in runs if run.run_id == "mixed")
    only_live = replace(mixed, cells=(replace(mixed.cells[1], batch_position=0),))
    runs = _replace_run(runs, mixed.run_id, only_live)

    check = _check(
        verify_mixed_mode_resident_isolation_v1(plan, runs),
        "inheritance_event_exposure",
    )
    assert check.ok is False
    assert check.diagnostics[0].code == "exposure.path_coverage"
    assert check.diagnostics[0].context["missing_paths"] == [REFILL]


def test_frozen_minimum_classifies_low_exposure_as_not_evaluable_not_negative() -> None:
    plan, runs = _passing_fixture()
    mixed = next(run for run in runs if run.run_id == "mixed")
    low_live = _cell(
        "live",
        LIVE,
        1,
        neighbors=(REFILL,),
        births=1,
    )
    runs = _replace_run(
        runs, mixed.run_id, replace(mixed, cells=(mixed.cells[0], low_live))
    )

    result = verify_mixed_mode_resident_isolation_v1(plan, runs)
    check = _check(result, "frozen_minimum_live_birth_exposure")
    assert check.ok is True
    assert result.live_birth_evaluability["live"]["classification"] == (
        "not_evaluable_low_exposure"
    )
    assert (
        result.live_birth_evaluability["live"]["negative_scientific_evidence_allowed"]
        is False
    )


def test_plan_binding_and_freeze_order_are_mandatory() -> None:
    plan, runs = _passing_fixture()
    mixed = next(run for run in runs if run.run_id == "mixed")
    runs = _replace_run(
        runs,
        mixed.run_id,
        replace(
            mixed,
            plan_sha256=ZERO,
            captured_at_utc="2026-07-23T21:00:00+00:00",
        ),
    )

    check = _check(
        verify_mixed_mode_resident_isolation_v1(plan, runs),
        "frozen_minimum_live_birth_exposure",
    )
    codes = {diagnostic.code for diagnostic in check.diagnostics}
    assert {
        "minimum.plan_binding",
        "minimum.not_frozen_before_capture",
    } <= codes


def test_contract_rejects_noncanonical_or_ambiguous_capture_data() -> None:
    with pytest.raises(
        MixedModeIsolationContractError,
        match="selected primary must belong",
    ):
        InheritanceAttemptV1(
            primary_eligible_parent_ids=("p0",),
            mate_eligible_parent_ids=("p1",),
            selected_parent_ids=("p1", "p1"),
            success=True,
            contributing_parent_ids=("p1",),
        )
