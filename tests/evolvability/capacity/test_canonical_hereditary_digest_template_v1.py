from __future__ import annotations

import hashlib
import random

import pytest

from agents.genome import Genome
from evolution.canonical_hereditary_digest_template_v1 import (
    CanonicalHereditaryDigestTemplateError,
    build_canonical_hereditary_digest_template_v1,
    common_canonical_hereditary_digest_template_v1,
    render_canonical_hereditary_digest_template_v1,
)
from evolution.diploid_inheritance_v1 import (
    attach_diploid_genome,
    diploid_genome_from_genome,
    homozygous_genome_from_phenotype,
)


def _genome(seed: int) -> Genome:
    genome = Genome.random(random.Random(seed))
    attach_diploid_genome(genome, diploid_genome_from_genome(genome))
    return genome


@pytest.mark.parametrize(
    "seed",
    (101, 202, 303, 404, 505, 606, 707, 808, 909, 1001, 6062, 7007),
)
def test_template_reconstructs_whole_canonical_payload(seed: int) -> None:
    genome = _genome(seed)
    template = build_canonical_hereditary_digest_template_v1(genome)
    rendered = render_canonical_hereditary_digest_template_v1(
        template,
        genome,
    )

    assert rendered == genome.canonical_bytes()
    assert hashlib.sha256(rendered).hexdigest() == genome.canonical_hash()


def test_template_identity_ignores_only_typed_resident_fields() -> None:
    left = _genome(1)
    right = _genome(2)

    assert common_canonical_hereditary_digest_template_v1((left, right))

    right.c1_operator_applied = "bounded_rewire"
    with pytest.raises(
        CanonicalHereditaryDigestTemplateError,
        match="do not share",
    ):
        common_canonical_hereditary_digest_template_v1((left, right))


def test_template_admits_resource_handling_axis_variation() -> None:
    left = _genome(11)
    right = _genome(12)
    left.resource_handling_axis = -1.0
    right.resource_handling_axis = 1.0
    attach_diploid_genome(left, homozygous_genome_from_phenotype(left))
    attach_diploid_genome(right, homozygous_genome_from_phenotype(right))

    template = common_canonical_hereditary_digest_template_v1((left, right))

    assert "resource_handling_axis" in template.gene_fields
    assert render_canonical_hereditary_digest_template_v1(
        template,
        left,
    ) == left.canonical_bytes()
    assert render_canonical_hereditary_digest_template_v1(
        template,
        right,
    ) == right.canonical_bytes()
