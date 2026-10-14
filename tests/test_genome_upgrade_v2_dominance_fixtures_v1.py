from __future__ import annotations

from evolution.genome_upgrade_v2 import (
    PRODUCT_COUNT,
    Q10_SCALE,
    GenomeUpgradeV2V1,
    diploid_dominance_series_v1,
    dominance_fixture_v1,
)


def _genome(
    *,
    product: int,
    activity_a: int,
    activity_b: int,
    modifier_a: int = 0,
    modifier_b: int = 0,
) -> GenomeUpgradeV2V1:
    neutral = GenomeUpgradeV2V1.neutral()
    alleles = [
        [list(parameters) for parameters in homolog]
        for homolog in neutral.alleles
    ]
    modifiers = [list(homolog) for homolog in neutral.modifiers]
    alleles[0][product] = [activity_a, Q10_SCALE, Q10_SCALE, 0]
    alleles[1][product] = [activity_b, Q10_SCALE, Q10_SCALE, 0]
    modifiers[0][product] = modifier_a
    modifiers[1][product] = modifier_b
    return GenomeUpgradeV2V1(
        alleles=tuple(
            tuple(tuple(parameters) for parameters in homolog)
            for homolog in alleles
        ),
        modifiers=tuple(tuple(homolog) for homolog in modifiers),
    )


def test_dosage_insufficiency_and_partial_full_loss_of_function() -> None:
    dosage = _genome(product=6, activity_a=400, activity_b=400)
    fixture = dominance_fixture_v1(
        dosage, (0,) * 8, product_index=6, threshold_q10=600
    )
    assert fixture["homolog_0_expressed"] is False
    assert fixture["homolog_1_expressed"] is False
    assert fixture["heterozygote_expressed"] is True

    partial = dominance_fixture_v1(
        _genome(product=6, activity_a=512, activity_b=0),
        (0,) * 8,
        product_index=6,
        threshold_q10=256,
    )
    full = dominance_fixture_v1(
        _genome(product=6, activity_a=0, activity_b=0),
        (0,) * 8,
        product_index=6,
        threshold_q10=256,
    )
    assert partial["heterozygote_output_q10"] > 0
    assert full["heterozygote_output_q10"] == 0


def test_incomplete_and_codominant_contributions_are_derived() -> None:
    series = diploid_dominance_series_v1(
        _genome(product=6, activity_a=128, activity_b=256),
        (0,) * 8,
        product_index=6,
    )
    assert series["incomplete_dominance"] is True
    assert series["codominant_contribution"] is True
    assert series["free_dominance_scalar"] is False


def test_modifier_background_produces_bounded_over_and_underdominance() -> None:
    over = diploid_dominance_series_v1(
        _genome(
            product=2,
            activity_a=0,
            activity_b=Q10_SCALE,
            modifier_a=0,
            modifier_b=-Q10_SCALE,
        ),
        (0,) * 8,
        product_index=2,
    )
    under = diploid_dominance_series_v1(
        _genome(
            product=2,
            activity_a=-Q10_SCALE,
            activity_b=-Q10_SCALE // 2,
            modifier_a=-Q10_SCALE,
            modifier_b=Q10_SCALE,
        ),
        (0,) * 8,
        product_index=2,
    )
    assert over["overdominance"] is True
    assert under["underdominance"] is True
    assert all(
        -Q10_SCALE <= value <= Q10_SCALE
        for value in (
            over["heterozygote_q10"],
            under["heterozygote_q10"],
        )
    )


def test_fixture_rejects_out_of_envelope_product() -> None:
    try:
        diploid_dominance_series_v1(
            GenomeUpgradeV2V1.neutral(),
            (0,) * 8,
            product_index=PRODUCT_COUNT,
        )
    except ValueError as exc:
        assert "outside the fixed envelope" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("out-of-envelope product was admitted")
