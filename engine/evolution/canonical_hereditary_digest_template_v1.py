"""Exact canonical-JSON templates for resident hereditary SHA-256."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Final, Mapping, Sequence

from evolution.diploid_inheritance_v1 import DIPLOID_SCALAR_LOCI


SCHEMA_VERSION: Final[str] = "canonical_hereditary_digest_template_v1"
TOKEN_PREFIX: Final[bytes] = b"\x1e"
TOKEN_SUFFIX: Final[bytes] = b"\x1f"
TOKEN_WIDTH: Final[int] = 3
MAX_TOKEN_COUNT: Final[int] = 256

GENE_FIELDS: Final[tuple[str, ...]] = (
    "action_temperature",
    "aggression_bias",
    "directed_inheritance_channel_mask",
    "drive_birth_noise",
    "drive_inertia",
    "drive_logit_gain",
    "drive_noise",
    "drive_sp_fatigue",
    "drive_sp_fear",
    "drive_sp_hunger",
    "drive_sp_pain",
    "exp_prior_noise",
    "exp_prior_strength",
    "exploration_bias",
    "hidden_size",
    "introspection_channel_mask",
    "memory_dim",
    "mutation_rate",
    "pred_lr",
    "reproduction_threshold",
    "self_modification_target_mask",
    "self_regulation_control_mask",
    "sensor_channel_mask",
    "sensor_radius",
    "use_gru",
)
OPTIONAL_GENE_FIELDS: Final[tuple[str, ...]] = (
    "resource_handling_axis",
)


class CanonicalHereditaryDigestTemplateError(ValueError):
    """Raised when a genome cannot enter the exact resident template envelope."""


@dataclass(frozen=True, slots=True)
class CanonicalHereditaryDigestTemplateV1:
    canonical_template: bytes
    diploid_payload_template: bytes
    genotype_hash_template: bytes
    gene_fields: tuple[str, ...] = GENE_FIELDS
    loci: tuple[str, ...] = DIPLOID_SCALAR_LOCI
    schema_version: str = SCHEMA_VERSION

    def static_identity_sha256(self) -> str:
        return hashlib.sha256(
            b"\0".join(
                (
                    self.schema_version.encode("ascii"),
                    self.canonical_template,
                    self.diploid_payload_template,
                    self.genotype_hash_template,
                    ",".join(self.gene_fields).encode("ascii"),
                    ",".join(self.loci).encode("ascii"),
                )
            )
        ).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _marker(token: int) -> str:
    return f"__EIEL_CANONICAL_TOKEN_{token:03d}__"


def _compile_template(value: Any, token_count: int) -> bytes:
    encoded = _json_bytes(value)
    for token in range(token_count):
        marker = _json_bytes(_marker(token))
        if encoded.count(marker) != 1:
            raise CanonicalHereditaryDigestTemplateError(
                f"canonical token {token} is absent or ambiguous"
            )
        encoded = encoded.replace(
            marker,
            TOKEN_PREFIX + bytes((token,)) + TOKEN_SUFFIX,
        )
    if b"__EIEL_CANONICAL_TOKEN_" in encoded:
        raise CanonicalHereditaryDigestTemplateError(
            "uncompiled canonical token remains"
        )
    return encoded


def _payload(genome: Any) -> dict[str, Any]:
    canonical_payload = getattr(genome, "canonical_payload", None)
    if not callable(canonical_payload):
        raise CanonicalHereditaryDigestTemplateError(
            "genome has no canonical payload"
        )
    payload = canonical_payload()
    if not isinstance(payload, Mapping):
        raise CanonicalHereditaryDigestTemplateError(
            "canonical genome payload is not a mapping"
        )
    return deepcopy(dict(payload))


def build_canonical_hereditary_digest_template_v1(
    genome: Any,
) -> CanonicalHereditaryDigestTemplateV1:
    payload = _payload(genome)
    genes = payload.get("genes")
    if not isinstance(genes, dict):
        raise CanonicalHereditaryDigestTemplateError(
            "canonical genome genes are missing"
        )
    diploid = genes.get("diploid_scalar_genome_v1")
    if not isinstance(diploid, dict):
        raise CanonicalHereditaryDigestTemplateError(
            "canonical diploid payload is missing"
        )
    for field in GENE_FIELDS:
        if field not in genes:
            raise CanonicalHereditaryDigestTemplateError(
                f"canonical resident gene is missing: {field}"
            )
    for homolog in ("allele_0", "allele_1"):
        values = diploid.get(homolog)
        if (
            not isinstance(values, list)
            or len(values) != len(DIPLOID_SCALAR_LOCI)
        ):
            raise CanonicalHereditaryDigestTemplateError(
                f"canonical diploid {homolog} is incompatible"
            )
    if not isinstance(diploid.get("genotype_sha256"), str):
        raise CanonicalHereditaryDigestTemplateError(
            "canonical diploid genotype digest is missing"
        )

    tokenized_gene_fields = (
        *GENE_FIELDS,
        *(field for field in OPTIONAL_GENE_FIELDS if field in genes),
    )
    canonical_value = deepcopy(payload)
    canonical_genes = canonical_value["genes"]
    canonical_genes["diploid_scalar_genome_v1"] = _marker(0)
    for token, field in enumerate(tokenized_gene_fields, start=1):
        canonical_genes[field] = _marker(token)
    canonical_template = _compile_template(
        canonical_value,
        1 + len(tokenized_gene_fields),
    )

    diploid_value = deepcopy(diploid)
    token = 0
    for homolog in ("allele_0", "allele_1"):
        for index in range(len(DIPLOID_SCALAR_LOCI)):
            diploid_value[homolog][index] = _marker(token)
            token += 1
    diploid_value["genotype_sha256"] = _marker(token)
    diploid_payload_template = _compile_template(diploid_value, token + 1)

    genotype_value = deepcopy(diploid)
    genotype_value.pop("genotype_sha256")
    token = 0
    for homolog in ("allele_0", "allele_1"):
        for index in range(len(DIPLOID_SCALAR_LOCI)):
            genotype_value[homolog][index] = _marker(token)
            token += 1
    genotype_hash_template = _compile_template(genotype_value, token)

    result = CanonicalHereditaryDigestTemplateV1(
        canonical_template=canonical_template,
        diploid_payload_template=diploid_payload_template,
        genotype_hash_template=genotype_hash_template,
        gene_fields=tuple(tokenized_gene_fields),
    )
    rendered = render_canonical_hereditary_digest_template_v1(result, genome)
    expected = getattr(genome, "canonical_bytes")()
    if rendered != expected:
        raise CanonicalHereditaryDigestTemplateError(
            "canonical template does not reproduce the genome payload"
        )
    return result


def _render(template: bytes, values: Sequence[bytes]) -> bytes:
    output = bytearray()
    cursor = 0
    while cursor < len(template):
        if (
            cursor + TOKEN_WIDTH <= len(template)
            and template[cursor : cursor + 1] == TOKEN_PREFIX
            and template[cursor + 2 : cursor + 3] == TOKEN_SUFFIX
        ):
            token = template[cursor + 1]
            if token >= len(values):
                raise CanonicalHereditaryDigestTemplateError(
                    f"canonical token is out of range: {token}"
                )
            output.extend(values[token])
            cursor += TOKEN_WIDTH
            continue
        output.append(template[cursor])
        cursor += 1
    return bytes(output)


def render_canonical_hereditary_digest_template_v1(
    template: CanonicalHereditaryDigestTemplateV1,
    genome: Any,
) -> bytes:
    payload = _payload(genome)
    genes = payload["genes"]
    diploid = genes["diploid_scalar_genome_v1"]
    alleles = tuple(
        _json_bytes(value)
        for homolog in ("allele_0", "allele_1")
        for value in diploid[homolog]
    )
    genotype_bytes = _render(template.genotype_hash_template, alleles)
    genotype_sha256 = hashlib.sha256(genotype_bytes).hexdigest()
    diploid_bytes = _render(
        template.diploid_payload_template,
        (*alleles, _json_bytes(genotype_sha256)),
    )
    gene_values = tuple(_json_bytes(genes[field]) for field in template.gene_fields)
    return _render(
        template.canonical_template,
        (diploid_bytes, *gene_values),
    )


def common_canonical_hereditary_digest_template_v1(
    genomes: Sequence[Any],
) -> CanonicalHereditaryDigestTemplateV1:
    if not genomes:
        raise CanonicalHereditaryDigestTemplateError(
            "at least one genome is required"
        )
    templates = [
        build_canonical_hereditary_digest_template_v1(genome)
        for genome in genomes
    ]
    first = templates[0]
    if any(template != first for template in templates[1:]):
        raise CanonicalHereditaryDigestTemplateError(
            "resident genomes do not share one exact canonical template"
        )
    return first


__all__ = [
    "CanonicalHereditaryDigestTemplateError",
    "CanonicalHereditaryDigestTemplateV1",
    "GENE_FIELDS",
    "OPTIONAL_GENE_FIELDS",
    "SCHEMA_VERSION",
    "TOKEN_PREFIX",
    "TOKEN_SUFFIX",
    "build_canonical_hereditary_digest_template_v1",
    "common_canonical_hereditary_digest_template_v1",
    "render_canonical_hereditary_digest_template_v1",
]
