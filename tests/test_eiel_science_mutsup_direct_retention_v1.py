from __future__ import annotations

from dataclasses import replace

import pytest

import emergence_evidence.science_mutsup_direct_retention_v1 as mutsup_reducer
from emergence_contracts import (
    CanonicalPayloadV1,
    ContentDigest,
    SchemaVersion,
    ScienceEvidencePartitionV1,
    ScienceObservationDraftV1,
    ScienceObservationProfileV1,
    ScienceRecordKindV1,
    canonical_sha256,
    canonical_science_record_stream_sha256_v1,
    finalize_science_observation_drafts_v1,
)
from emergence_contracts.science_instrument import _RECORD_REQUIRED_PAYLOAD_FIELDS
from emergence_evidence import (
    MUTSUP_DIRECT_RETENTION_CONFIGURATION_SCHEMA_VERSION,
    MUTSUP_DIRECT_RETENTION_REQUIRED_CAPABILITY_IDS,
    ScienceReducerError,
    aggregate_mutsup_direct_retention_v1,
    default_science_reducer_registry_v1,
    execute_mutsup_direct_retention_v1,
)


def _digest(character: str) -> ContentDigest:
    return ContentDigest("sha256", character * 64)


def _payload(kind: ScienceRecordKindV1, **overrides: object) -> CanonicalPayloadV1:
    value = {field: None for field in _RECORD_REQUIRED_PAYLOAD_FIELDS[kind]}
    value.update(overrides)
    return CanonicalPayloadV1.from_value(
        SchemaVersion("mutsup_fixture_payload_v1"), value
    )


def _draft(
    kind: ScienceRecordKindV1,
    tick: int,
    address: str,
    payload: CanonicalPayloadV1,
    *,
    phase: int = 10,
) -> ScienceObservationDraftV1:
    return ScienceObservationDraftV1(
        producer_component_id="mutsup-fixture-producer",
        producer_contract_version=SchemaVersion("mutsup_fixture_producer_v1"),
        producer_phase_registry_ordinal=phase,
        absolute_tick=tick,
        semantic_source_address=address,
        record_kind=kind,
        record_kind_version=1,
        control_binding_sha256=_digest("a"),
        payload=payload,
    )


def _control(cell_id: str) -> ScienceObservationDraftV1:
    return _draft(
        ScienceRecordKindV1.CONTROL_BINDING,
        0,
        f"world/{cell_id}/control",
        _payload(
            ScienceRecordKindV1.CONTROL_BINDING,
            binding_id=f"binding-{cell_id}",
            semantic_cell_id=cell_id,
            seed_bindings={"fixture": 7},
            control_values={"mutation": "enabled"},
            control_schedules={"mutation": {"kind": "static"}},
            registered_profile_ids=["SCIENCE_STANDARD_V1"],
            semantic_envelope_resolution_sha256=_digest("b").to_dict(),
            semantic_applied_readback_sha256=_digest("c").to_dict(),
            continuation_parent_execution_identity=None,
        ),
        phase=1,
    )


def _birth(
    child_id: int,
    tick: int,
    parent_ids: list[int],
    parent_genomes: list[str],
    child_pre: str,
    child_post: str,
) -> tuple[ScienceObservationDraftV1, ScienceObservationDraftV1]:
    attempt_id = f"birth-attempt-{child_id}"
    opportunity_id = f"birth-opportunity-{child_id}"
    common = {
        "birth_attempt_id": attempt_id,
        "opportunity_id": opportunity_id,
        "parent_agent_ids": parent_ids,
        "parent_genome_state_ids": parent_genomes,
        "birth_mode_id": "fixture_birth",
        "inheritance_operator_id": "fixture_inheritance",
        "recombination_operator_id": "fixture_recombination",
    }
    attempt = _draft(
        ScienceRecordKindV1.BIRTH_ATTEMPT,
        tick,
        f"birth/{child_id}/attempt",
        _payload(
            ScienceRecordKindV1.BIRTH_ATTEMPT,
            **common,
            attempt_result_class="COMMITTED",
            reason_code="fixture_committed",
        ),
        phase=20,
    )
    committed = _draft(
        ScienceRecordKindV1.BIRTH_COMMITTED,
        tick,
        f"birth/{child_id}/committed",
        _payload(
            ScienceRecordKindV1.BIRTH_COMMITTED,
            **common,
            child_agent_id=child_id,
            parent_hereditary_package_state_ids=[],
            child_pre_mutation_genome_state_id=child_pre,
            child_post_birth_genome_state_id=child_post,
            child_hereditary_package_state_ids=[],
            child_linkage_state_id=f"link-{child_id}",
            founder_id="fixture-founder",
            lineage_id=f"lineage-{child_id}",
            lineage_depth=1,
        ),
        phase=21,
    )
    return attempt, committed


def _mutation(
    child_id: int,
    tick: int,
    pre_genome: str,
    post_genome: str,
    *,
    result_class: str = "MUTATION_APPLIED",
    address_suffix: str = "",
    native_synthetic: bool = False,
    native_exact: bool = False,
    native_saturated: bool = False,
) -> ScienceObservationDraftV1:
    origin_agent_id = child_id if native_exact else 10
    mutation_class_id = (
        "native_genome_v2_point_mutation"
        if native_exact
        else ("native_mutation_count" if native_synthetic else "fixture_point_mutation")
    )
    if native_exact:
        native_pre_value = -127 if native_saturated else 0
        native_requested_delta = -1 if native_saturated else 1
        details = [
            {
                "genome_region": 0,
                "haplotype": 0,
                "linked_locus": 0,
                "pre_value": native_pre_value,
                "post_value": native_pre_value if native_saturated else 1,
                "requested_delta": native_requested_delta,
                "effective_delta": 0 if native_saturated else 1,
                "saturated_noop": native_saturated,
            }
        ]
        affected_coordinates = {
            "accepted_draws": 1,
            "accepted_draw_details": details,
            "changed_alleles": [] if native_saturated else [
                {
                    "genome_region": 0,
                    "haplotype": 0,
                    "linked_locus": 0,
                    "pre_value": 0,
                    "post_value": 1,
                }
            ],
            "native_mutation_count": 1,
            "saturated_noop_count": 1 if native_saturated else 0,
        }
    else:
        affected_coordinates = {"site": child_id}
    return _draft(
        ScienceRecordKindV1.MUTATION_OUTCOME,
        tick,
        f"mutation/{child_id}/{result_class.lower()}{address_suffix}",
        _payload(
            ScienceRecordKindV1.MUTATION_OUTCOME,
            mutation_opportunity_id=f"mutation-opportunity-{child_id}",
            mutation_attempt_id=f"mutation-attempt-{child_id}",
            subject_agent_id=child_id,
            origin_agent_id=origin_agent_id,
            origin_absolute_tick=tick,
            attempted=True,
            result_class=(
                "MUTATION_SATURATED_NOOP"
                if native_saturated
                else result_class
            ),
            operator_id=(
                "native_genome_v2_mutation" if native_exact else "fixture_mutation"
            ),
            mutation_class_id=mutation_class_id,
            affected_coordinates=affected_coordinates,
            pre_value_or_state_digest=(
                canonical_sha256({"genome_state_id": pre_genome}).to_dict()
                if native_exact
                else _digest("d").to_dict()
            ),
            post_value_or_state_digest=(
                canonical_sha256({"genome_state_id": post_genome}).to_dict()
                if native_exact
                else _digest("e").to_dict()
            ),
            bounded_delta=(
                {
                    "mutation_count": 1,
                    "accepted_draws": 1,
                    "changed_allele_count": 0 if native_saturated else 1,
                    "saturated_noop_count": 1 if native_saturated else 0,
                }
                if native_exact
                else {"bounded": True}
            ),
            pre_genome_state_id=pre_genome,
            post_genome_state_id=post_genome,
            reason_code=(
                "native_exact_genome_delta"
                if native_exact
                else ("native_mutation_count" if native_synthetic else "fixture_mutation")
            ),
        ),
        phase=22,
    )


def _completion(cell_id: str, tick: int, rows) -> ScienceObservationDraftV1:
    prefix = canonical_science_record_stream_sha256_v1(tuple(rows))
    return _draft(
        ScienceRecordKindV1.WORLD_COMPLETION,
        tick,
        f"world/{cell_id}/completion",
        _payload(
            ScienceRecordKindV1.WORLD_COMPLETION,
            completion_id=f"completion-{cell_id}",
            completion_class="TERMINAL_HORIZON",
            terminal_state_sha256=_digest("f").to_dict(),
            terminal_population_count=4,
            terminal_absolute_tick=tick,
            # Native publication seals this marker after the prefix is sealed.
            next_absolute_tick=tick,
            terminal_record_count=len(rows),
            canonical_record_stream_sha256=prefix.to_dict(),
            checkpoint_occurrence_ids=[],
            reason_code="fixture_terminal",
        ),
        phase=99,
    )


def _world(
    name: str,
    *,
    arm_id: str = "on",
    replicate: int = 1,
    pair_id: str | None = "pair-1",
    mutation_mode: str = "retained",
    acquisition_character: str = "3",
    native_synthetic: bool = False,
    native_exact: bool = False,
    native_saturated: bool = False,
):
    cell_id = f"cell-{name}"
    drafts: list[ScienceObservationDraftV1] = [_control(cell_id)]
    if mutation_mode == "retained":
        first_pre = "g0"
        first_post = "g0" if native_saturated else "g1"
        later_parent = "g0" if native_saturated else "g1"
        attempt, first = _birth(
            20, 1, [10, 11], ["p0", "p1"], first_pre, first_post
        )
        later_attempt, later = _birth(
            40, 3, [20, 30], [later_parent, "p2"], later_parent, later_parent
        )
        third_attempt, third = _birth(
            50, 5, [40, 31], [later_parent, "p3"], later_parent, later_parent
        )
        drafts.extend(
            (
                attempt,
                first,
                _mutation(
                    20,
                    1,
                    "g0",
                    first_post,
                    native_synthetic=native_synthetic,
                    native_exact=native_exact,
                    native_saturated=native_saturated,
                ),
            )
        )
        drafts.extend((later_attempt, later, third_attempt, third))
    elif mutation_mode == "duplicate":
        attempt, first = _birth(20, 1, [10, 11], ["p0", "p1"], "g0", "g1")
        drafts.extend(
            (
                attempt,
                first,
                _mutation(20, 1, "g0", "g1"),
                _mutation(20, 1, "g0", "g1", address_suffix="-duplicate"),
            )
        )
    elif mutation_mode == "death":
        attempt, first = _birth(21, 1, [10, 11], ["p0", "p1"], "g0", "g2")
        drafts.extend((attempt, first, _mutation(21, 1, "g0", "g2")))
        drafts.append(
            _draft(
                ScienceRecordKindV1.DEATH,
                4,
                "death/21",
                _payload(
                    ScienceRecordKindV1.DEATH,
                    agent_id=21,
                    terminal_agent_state_id="state-21",
                    genome_state_id="g2",
                    lineage_id="lineage-21",
                    cause_class="fixture_death",
                    reason_code="fixture_terminal",
                    causing_event_ids=[],
                ),
                phase=30,
            )
        )
    elif mutation_mode == "no_op":
        drafts.append(_mutation(20, 1, "g0", "g0", result_class="NO_OP"))
    elif mutation_mode == "identical_applied":
        drafts.append(
            _mutation(20, 1, "g0", "g0", result_class="MUTATION_APPLIED")
        )
    elif mutation_mode != "none":
        raise AssertionError(mutation_mode)

    base_rows = finalize_science_observation_drafts_v1(
        tuple(drafts),
        instrument_manifest_sha256=_digest("1"),
        crca_semantic_run_digest=_digest("2"),
        science_semantic_acquisition_sha256=_digest(acquisition_character),
        cell_id=cell_id,
        logical_world_id=0,
        semantic_world_key=CanonicalPayloadV1.from_value(
            SchemaVersion("mutsup_fixture_world_key_v1"), {"world": name}
        ),
    )
    all_rows = finalize_science_observation_drafts_v1(
        tuple(drafts) + (_completion(cell_id, 10, base_rows),),
        instrument_manifest_sha256=_digest("1"),
        crca_semantic_run_digest=_digest("2"),
        science_semantic_acquisition_sha256=_digest(acquisition_character),
        cell_id=cell_id,
        logical_world_id=0,
        semantic_world_key=CanonicalPayloadV1.from_value(
            SchemaVersion("mutsup_fixture_world_key_v1"), {"world": name}
        ),
    )
    world_uid = all_rows[0].world_uid
    partitions: list[ScienceEvidencePartitionV1] = []
    for kind in ScienceRecordKindV1:
        kind_rows = tuple(row for row in all_rows if row.record_kind is kind)
        partitions.append(
            ScienceEvidencePartitionV1(
                instrument_manifest_sha256=_digest("1"),
                science_semantic_acquisition_sha256=_digest(acquisition_character),
                observation_abi_version=SchemaVersion(
                    "eiel_science_observation_abi_v1"
                ),
                profile=ScienceObservationProfileV1.STANDARD,
                world_uid=world_uid,
                record_kind=kind,
                absolute_tick_start=0,
                absolute_tick_end=11,
                rows=kind_rows,
            )
        )
    configuration = CanonicalPayloadV1.from_value(
        MUTSUP_DIRECT_RETENTION_CONFIGURATION_SCHEMA_VERSION,
        {
            "world_ownership": [
                {
                    "world_uid": world_uid.to_dict(),
                    "study_id": "fixture-study",
                    "arm_id": arm_id,
                    "replicate": replicate,
                    "pair_id": pair_id,
                }
            ],
            "milestone_offsets_ticks": [1, 2, 5],
        },
    )
    return {
        "rows": all_rows,
        "partitions": tuple(partitions),
        "configuration": configuration,
        "world_uid": world_uid,
        "acquisition": _digest(acquisition_character),
    }


def _execute(fixture, *, stop_after_partitions=None, resume=None, capabilities=None):
    return execute_mutsup_direct_retention_v1(
        fixture["configuration"],
        fixture["partitions"],
        evidence_capability_ids=(
            MUTSUP_DIRECT_RETENTION_REQUIRED_CAPABILITY_IDS
            if capabilities is None
            else capabilities
        ),
        campaign_spec_sha256=_digest("4"),
        instrument_manifest_sha256=_digest("1"),
        route_binding_sha256=_digest("5"),
        science_semantic_acquisition_sha256=fixture["acquisition"],
        physical_execution_plan_sha256=_digest("6"),
        science_execution_plan_sha256=_digest("7"),
        reduction_plan_sha256=_digest("8"),
        stop_after_partitions=stop_after_partitions,
        resume=resume,
    )


def test_three_generation_direct_retention_and_death_milestones() -> None:
    retained = _execute(_world("retained"))
    assert retained.complete
    world = retained.result_payload.to_value()["worlds"][0]
    assert world["counts"]["applied_mutation_count"] == 1
    assert world["direct_reproductive_retention_count"] == 1
    assert world["direct_reproductive_retention_fraction"] == 1.0
    assert [
        item["mutations_with_later_carrier"] for item in world["carrier_milestones"]
    ] == [
        0,
        1,
        1,
    ]

    death = _execute(_world("death", mutation_mode="death"))
    death_world = death.result_payload.to_value()["worlds"][0]
    assert death_world["direct_reproductive_retention_count"] == 0
    assert death_world["deaths_before_retention"] == {
        "evidence_supported": True,
        "count": 1,
        "denominator": 1,
        "fraction": 1.0,
    }


def test_partition_permutation_and_physical_packing_are_invariant() -> None:
    fixture = _world("permutation")
    uninterrupted = _execute(fixture)
    partitions = list(fixture["partitions"])
    birth_partition = next(
        item
        for item in partitions
        if item.record_kind is ScienceRecordKindV1.BIRTH_COMMITTED
    )
    split = birth_partition.rows[:1], birth_partition.rows[1:]
    repacked = [
        item
        for item in partitions
        if item.record_kind is not ScienceRecordKindV1.BIRTH_COMMITTED
    ]
    repacked.extend(
        ScienceEvidencePartitionV1(
            instrument_manifest_sha256=birth_partition.instrument_manifest_sha256,
            science_semantic_acquisition_sha256=birth_partition.science_semantic_acquisition_sha256,
            observation_abi_version=birth_partition.observation_abi_version,
            profile=birth_partition.profile,
            world_uid=birth_partition.world_uid,
            record_kind=birth_partition.record_kind,
            absolute_tick_start=0,
            absolute_tick_end=11,
            rows=rows,
        )
        for rows in split
    )
    permuted = tuple(reversed(repacked))
    permuted_fixture = {**fixture, "partitions": permuted}
    permuted_result = _execute(permuted_fixture)
    assert permuted_result.result_payload == uninterrupted.result_payload


def test_checkpoint_restart_is_byte_identical() -> None:
    fixture = _world("restart")
    uninterrupted = _execute(fixture)
    interrupted = _execute(fixture, stop_after_partitions=3)
    assert not interrupted.complete and interrupted.checkpoint is not None
    resumed = _execute(fixture, resume=interrupted.checkpoint)
    assert resumed.complete
    assert resumed.result_payload == uninterrupted.result_payload
    assert (
        resumed.result_receipt.result_sha256
        == uninterrupted.result_receipt.result_sha256
    )
    assert (
        resumed.result_receipt.resume_chain_sha256
        == interrupted.checkpoint.checkpoint_sha256()
    )


def test_missing_capability_or_required_record_kind_fails_closed() -> None:
    fixture = _world("missing")
    with pytest.raises(ScienceReducerError, match="capabilities are missing"):
        _execute(fixture, capabilities=())
    missing_kind = {
        **fixture,
        "partitions": tuple(
            item
            for item in fixture["partitions"]
            if item.record_kind is not ScienceRecordKindV1.MUTATION_OUTCOME
        ),
    }
    with pytest.raises(ScienceReducerError, match="missing required record kind"):
        _execute(missing_kind)


def test_off_arm_mutation_leak_duplicate_and_missing_join_fail_closed() -> None:
    with pytest.raises(ScienceReducerError, match="mutation in OFF arm"):
        _execute(_world("off-leak", arm_id="off"))

    duplicate_fixture = _world("duplicate", mutation_mode="duplicate")
    with pytest.raises(ScienceReducerError, match="duplicate mutation_attempt_id"):
        _execute(duplicate_fixture)

    missing_join = _world("missing-join")
    birth_partition = next(
        item
        for item in missing_join["partitions"]
        if item.record_kind is ScienceRecordKindV1.BIRTH_COMMITTED
    )
    birth_row = birth_partition.rows[0]
    bad_payload = birth_row.payload.to_value()
    bad_payload["child_post_birth_genome_state_id"] = "not-the-mutation-genome"
    bad_birth = replace(
        birth_row,
        payload=CanonicalPayloadV1.from_value(
            birth_row.payload.schema_version, bad_payload
        ),
    )
    bad_birth_partition = replace(
        birth_partition,
        rows=(bad_birth,) + birth_partition.rows[1:],
    )
    missing_join["partitions"] = tuple(
        bad_birth_partition if item is birth_partition else item
        for item in missing_join["partitions"]
    )
    with pytest.raises(
        ScienceReducerError,
        match="dropped evidence|missing or ambiguous mutation-to-child",
    ):
        _execute(missing_join)


def test_native_aggregate_mutation_provenance_is_rejected() -> None:
    with pytest.raises(ScienceReducerError, match="aggregate-only provenance"):
        _execute(_world("native-aggregate", native_synthetic=True))


def test_native_exact_child_origin_mutation_is_accepted() -> None:
    result = _execute(_world("native-exact", native_exact=True))
    assert result.complete
    world = result.result_payload.to_value()["worlds"][0]
    assert world["counts"]["applied_mutation_count"] == 1
    assert world["direct_reproductive_retention_count"] == 1


def test_native_saturated_mutation_is_explicitly_non_applied() -> None:
    result = _execute(
        _world("native-saturated", native_exact=True, native_saturated=True)
    )
    assert result.complete
    world = result.result_payload.to_value()["worlds"][0]
    assert world["counts"]["mutation_outcome_count"] == 1
    assert world["counts"]["applied_mutation_count"] == 0
    assert world["mutation_opportunity"]["status"] == "NO_OBSERVED_MUTATIONS"


def test_generic_identical_pre_post_applied_claim_is_rejected() -> None:
    with pytest.raises(ScienceReducerError, match="identical pre/post"):
        _execute(_world("identical-applied", mutation_mode="identical_applied"))


def test_native_exact_non_saturated_requested_direction_tamper_is_rejected() -> None:
    fixture = _world("native-direction-tamper", native_exact=True)
    mutation_partition = next(
        item
        for item in fixture["partitions"]
        if item.record_kind is ScienceRecordKindV1.MUTATION_OUTCOME
    )
    row = mutation_partition.rows[0].semantic_dict()
    detail = row["payload"]["payload"]["affected_coordinates"][
        "accepted_draw_details"
    ][0]
    # The native detail says 0 -> 1, so changing only the requested direction
    # must not remain a valid non-saturated event.
    detail["requested_delta"] = -1
    with pytest.raises(ScienceReducerError, match="requested delta"):
        mutsup_reducer._mutation(row)


def test_native_exact_saturated_wrong_boundary_direction_is_rejected() -> None:
    fixture = _world("native-boundary-tamper", native_exact=True)
    mutation_partition = next(
        item
        for item in fixture["partitions"]
        if item.record_kind is ScienceRecordKindV1.MUTATION_OUTCOME
    )
    row = mutation_partition.rows[0].semantic_dict()
    payload = row["payload"]["payload"]
    detail = payload["affected_coordinates"]["accepted_draw_details"][0]
    # -127 is the native negative policy boundary.  A requested +1 draw there
    # is not a saturated no-op; accepting this would permit false saturation.
    detail.update(
        pre_value=-127,
        post_value=-127,
        requested_delta=1,
        effective_delta=0,
        saturated_noop=True,
    )
    coordinates = payload["affected_coordinates"]
    coordinates["changed_alleles"] = []
    coordinates["saturated_noop_count"] = 1
    payload["result_class"] = "MUTATION_SATURATED_NOOP"
    payload["post_genome_state_id"] = payload["pre_genome_state_id"]
    digest = canonical_sha256(
        {"genome_state_id": payload["pre_genome_state_id"]}
    ).to_dict()
    payload["pre_value_or_state_digest"] = digest
    payload["post_value_or_state_digest"] = digest
    payload["bounded_delta"]["changed_allele_count"] = 0
    payload["bounded_delta"]["saturated_noop_count"] = 1
    with pytest.raises(ScienceReducerError, match="native boundary"):
        mutsup_reducer._mutation(row)


def test_native_exact_delta_tamper_is_rejected() -> None:
    fixture = _world("native-exact-tamper", native_exact=True)
    mutation_partition = next(
        item
        for item in fixture["partitions"]
        if item.record_kind is ScienceRecordKindV1.MUTATION_OUTCOME
    )
    mutation_row = mutation_partition.rows[0]
    payload = mutation_row.payload.to_value()
    payload["affected_coordinates"] = {
        "changed_alleles": [],
        "native_mutation_count": 1,
    }
    tampered_row = replace(
        mutation_row,
        payload=CanonicalPayloadV1.from_value(
            mutation_row.payload.schema_version, payload
        ),
    )
    tampered_partition = replace(
        mutation_partition,
        rows=(tampered_row,) + mutation_partition.rows[1:],
    )
    fixture["partitions"] = tuple(
        tampered_partition if item is mutation_partition else item
        for item in fixture["partitions"]
    )
    with pytest.raises(
        ScienceReducerError,
        match="native exact mutation allele delta|dropped evidence",
    ):
        _execute(fixture)


def test_gap_and_dropped_evidence_fail_closed() -> None:
    fixture = _world("integrity")
    birth = next(
        item
        for item in fixture["partitions"]
        if item.record_kind is ScienceRecordKindV1.BIRTH_COMMITTED
    )
    gapped = replace(
        birth.rows[-1], world_local_sequence=birth.rows[-1].world_local_sequence + 2
    )
    gapped_partition = replace(birth, rows=tuple(birth.rows[:-1]) + (gapped,))
    gapped_fixture = {
        **fixture,
        "partitions": tuple(
            gapped_partition if item is birth else item
            for item in fixture["partitions"]
        ),
    }
    with pytest.raises(ScienceReducerError, match="sequence gap"):
        _execute(gapped_fixture)

    dropped = _world("dropped")
    dropped_configuration = dropped["configuration"].to_value()
    dropped_configuration["dropped_records"] = 1
    dropped["configuration"] = CanonicalPayloadV1.from_value(
        dropped["configuration"].schema_version, dropped_configuration
    )
    with pytest.raises(ScienceReducerError, match="dropped"):
        _execute(dropped)


def test_no_observed_mutations_is_distinct_from_insufficient_opportunity_evidence() -> (
    None
):
    no_observed = _execute(_world("no-op", mutation_mode="no_op"))
    no_observed_world = no_observed.result_payload.to_value()["worlds"][0]
    assert (
        no_observed_world["mutation_opportunity"]["status"] == "NO_OBSERVED_MUTATIONS"
    )
    insufficient = _execute(_world("no-opportunity", mutation_mode="none"))
    insufficient_world = insufficient.result_payload.to_value()["worlds"][0]
    assert (
        insufficient_world["mutation_opportunity"]["status"]
        == "INSUFFICIENT_OPPORTUNITY_EVIDENCE"
    )


def test_complete_applied_outcome_stream_distinguishes_disabled_control_from_zero_events() -> (
    None
):
    off = _world("off-complete", arm_id="off", mutation_mode="none")
    on = _world("on-complete", arm_id="on", mutation_mode="none")
    for fixture in (off, on):
        configuration = fixture["configuration"].to_value()
        configuration["mutation_observation_semantics"] = (
            "applied_outcomes_only_complete"
        )
        fixture["configuration"] = CanonicalPayloadV1.from_value(
            MUTSUP_DIRECT_RETENTION_CONFIGURATION_SCHEMA_VERSION,
            configuration,
        )

    off_world = _execute(off).result_payload.to_value()["worlds"][0]
    on_world = _execute(on).result_payload.to_value()["worlds"][0]
    assert off_world["mutation_opportunity"]["status"] == ("MUTATION_DISABLED_CONTROL")
    assert on_world["mutation_opportunity"]["status"] == "NO_OBSERVED_MUTATIONS"


def test_study_aggregator_uses_explicit_pair_map_and_preserves_acquisition_ids() -> (
    None
):
    off = _execute(
        _world(
            "off-empty", arm_id="off", mutation_mode="none", acquisition_character="3"
        )
    )
    on = _execute(_world("on-retained", arm_id="on", acquisition_character="4"))
    off_world = off.result_payload.to_value()["worlds"][0]
    on_world = on.result_payload.to_value()["worlds"][0]
    pairing = {
        "pairs": [
            {
                "pair_id": "pair-1",
                "off_world_uid": off_world["world_uid"],
                "on_world_uid": on_world["world_uid"],
            }
        ],
        "off_arm_ids": ["off"],
        "on_arm_ids": ["on"],
    }
    result = aggregate_mutsup_direct_retention_v1(
        [on.result_payload, off.result_payload], pairing=pairing
    )
    value = result.to_value()
    assert value["decision"]["classification"] == "INSUFFICIENT_OPPORTUNITY_EVIDENCE"
    assert len(value["paired_contrasts"]) == 1
    assert value["paired_contrasts"][0]["metrics"][
        "direct_reproductive_retention_count"
    ] == {
        "off": 0,
        "on": 1,
        "on_minus_off": 1,
    }
    expected_acquisitions = [
        world["science_semantic_acquisition_sha256"]
        for world in sorted(
            (off_world, on_world), key=lambda item: item["world_uid"]["hex"]
        )
    ]
    assert value["science_semantic_acquisition_identities"] == expected_acquisitions
    assert value["physical_partition_invariant"] is True


def test_registry_exposes_the_protocol_bound_reducer() -> None:
    registry = default_science_reducer_registry_v1()
    registration = registry.get("science.mutsup.direct_retention.v1")
    assert registry.reducer_ids() == ("science.mutsup.direct_retention.v1",)
    assert registration.descriptor.reducer_id == registration.implementation.reducer_id
