"""Content-bound exact natural hereditary source-state contracts.

The contract is intentionally route neutral.  A source-state reference carries
the exact genome bytes needed by a later materializer, but those bytes are
always bound to the authenticated historical source and replay provenance.
There is no operation here that edits, synthesizes, or infers a genome from a
tracked projection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any, Final

from .canonical import canonical_json_bytes, canonical_sha256


EXACT_NATURAL_SOURCE_STATE_REF_SCHEMA_V1: Final[str] = (
    "eiel_exact_natural_source_state_ref_v1"
)
EXACT_NATURAL_SOURCE_STATE_BANK_SCHEMA_V1: Final[str] = (
    "eiel_exact_natural_source_state_bank_v1"
)
EXACT_NATURAL_SOURCE_STATE_MATERIALIZATION_METHOD_V1: Final[str] = (
    "inline_exact_genome_payload_v1"
)
EXACT_NATURAL_SOURCE_STATE_GENOME_ENCODING_V1: Final[str] = (
    "native_policy_trait_hex_v1"
)
EXACT_NATURAL_SOURCE_STATE_REF_CLASSES_V1: Final[frozenset[str]] = frozenset(
    {"BRANCHABLE_EXACT", "REPLAY_EXACT"}
)


class ExactNaturalSourceStateError(ValueError):
    """A source-state reference or bank is not exact and authenticated."""


def _text(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ExactNaturalSourceStateError(f"{field} must be nonempty text")
    return value


def _sha256_id(value: object, *, field: str) -> str:
    value = _text(value, field=field)
    if len(value) != 71 or not value.startswith("sha256:"):
        raise ExactNaturalSourceStateError(f"{field} is not a sha256 identity")
    if any(character not in "0123456789abcdef" for character in value[7:]):
        raise ExactNaturalSourceStateError(f"{field} is not a lowercase sha256 identity")
    return value


def _digest(value: object, *, field: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"algorithm", "hex"}:
        raise ExactNaturalSourceStateError(f"{field} must be a sha256 digest object")
    if value.get("algorithm") != "sha256":
        raise ExactNaturalSourceStateError(f"{field} algorithm is not sha256")
    digest = value.get("hex")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ExactNaturalSourceStateError(f"{field} is not a lowercase sha256 digest")
    return {"algorithm": "sha256", "hex": digest}


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ExactNaturalSourceStateError(f"{field} must be an integer >= {minimum}")
    return int(value)


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExactNaturalSourceStateError(f"{field} must be an object")
    return value


def _canonical_ref_core(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in value if key != "ref_id"}


def _artifact_rows(value: object, *, field: str) -> tuple[dict[str, str], ...]:
    if not isinstance(value, list) or not value:
        raise ExactNaturalSourceStateError(f"{field} must be a nonempty array")
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(value):
        row = _mapping(raw, field=f"{field}[{index}]")
        if set(row) != {"kind", "path", "sha256"}:
            raise ExactNaturalSourceStateError(
                f"{field}[{index}] fields are not exact"
            )
        kind = _text(row["kind"], field=f"{field}[{index}].kind")
        path = _text(row["path"], field=f"{field}[{index}].path")
        # Artifact rows use the ordinary lowercase hex form because the
        # digest is also used directly for byte authentication.  Accept a
        # ``sha256:`` input spelling, but normalize it away at the contract
        # boundary so the stored representation is unambiguous.
        digest = _digest_hex(
            row["sha256"], field=f"{field}[{index}].sha256"
        )
        key = (kind, path)
        if key in seen:
            raise ExactNaturalSourceStateError(f"{field} contains duplicate artifacts")
        seen.add(key)
        rows.append({"kind": kind, "path": path, "sha256": digest})
    ordered = tuple(sorted(rows, key=lambda row: (row["kind"], row["path"])))
    if tuple(rows) != ordered:
        raise ExactNaturalSourceStateError(f"{field} must be deterministically ordered")
    return ordered


def _digest_hex(value: object, *, field: str) -> str:
    if isinstance(value, Mapping):
        return _digest(value, field=field)["hex"]
    if isinstance(value, str) and value.startswith("sha256:"):
        value = value[7:]
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExactNaturalSourceStateError(f"{field} is not a lowercase sha256 digest")
    return value


def _genome_payload(value: object, *, field: str) -> dict[str, str]:
    row = _mapping(value, field=field)
    if set(row) != {"encoding", "policy_hex", "trait_hex"}:
        raise ExactNaturalSourceStateError(f"{field} fields are not exact")
    if row.get("encoding") != EXACT_NATURAL_SOURCE_STATE_GENOME_ENCODING_V1:
        raise ExactNaturalSourceStateError(f"{field}.encoding is unsupported")
    policy_hex = _hex(row.get("policy_hex"), expected_bytes=224, field=f"{field}.policy_hex")
    trait_hex = _hex(row.get("trait_hex"), expected_bytes=64, field=f"{field}.trait_hex")
    return {
        "encoding": EXACT_NATURAL_SOURCE_STATE_GENOME_ENCODING_V1,
        "policy_hex": policy_hex,
        "trait_hex": trait_hex,
    }


def _hex(value: object, *, expected_bytes: int, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != expected_bytes * 2
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExactNaturalSourceStateError(
            f"{field} must be lowercase hexadecimal for {expected_bytes} bytes"
        )
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ExactNaturalSourceStateError(f"{field} is not hexadecimal") from error
    return value


def full_genome_bytes_v1(value: Mapping[str, Any]) -> bytes:
    """Return the exact native genome bytes carried by a validated reference."""

    genome = _genome_payload(value.get("genome"), field="genome")
    return bytes.fromhex(genome["policy_hex"] + genome["trait_hex"])


def full_genome_state_id_v1(genome_bytes: bytes) -> str:
    if not isinstance(genome_bytes, bytes) or len(genome_bytes) != 288:
        raise ExactNaturalSourceStateError("native genome bytes must be exactly 288 bytes")
    return "sha256:" + canonical_sha256(
        {"policy": genome_bytes[:224].hex(), "trait": genome_bytes[224:].hex()}
    ).hex


def tracked_genotype_state_id_v1(
    genome_bytes: bytes,
    *,
    policy_loci: Sequence[int],
    trait_loci: Sequence[int],
) -> str:
    """Derive the carrier-compatible tracked identity without full-state collapse."""

    if not isinstance(genome_bytes, bytes) or len(genome_bytes) != 288:
        raise ExactNaturalSourceStateError("native genome bytes must be exactly 288 bytes")
    policy = genome_bytes[:224]
    trait = genome_bytes[224:]
    if not policy_loci and not trait_loci:
        return full_genome_state_id_v1(genome_bytes)
    encoded = bytearray(b"tg1\x00")
    encoded.extend(len(tuple(policy_loci)).to_bytes(2, "little"))
    for locus in policy_loci:
        _integer(locus, field="policy_locus")
        if locus >= 112:
            raise ExactNaturalSourceStateError("policy locus is outside the native range")
        encoded.extend(int(locus).to_bytes(2, "little"))
        encoded.extend(int.from_bytes(policy[locus:locus + 1], "little", signed=True).to_bytes(2, "little", signed=True))
        encoded.extend(int.from_bytes(policy[112 + locus:113 + locus], "little", signed=True).to_bytes(2, "little", signed=True))
    encoded.extend(len(tuple(trait_loci)).to_bytes(2, "little"))
    for locus in trait_loci:
        _integer(locus, field="trait_locus")
        if locus >= 16:
            raise ExactNaturalSourceStateError("trait locus is outside the native range")
        encoded.extend(int(locus).to_bytes(2, "little"))
        # The native carrier addresses the byte-level genome ABI: its
        # configured trait locus is a byte offset in the 32-byte trait
        # region, with the second haplotype at ``16 + locus``.  Preserve that
        # exact unsigned-byte projection rather than replacing it with a
        # signed int16 interpretation.
        encoded.extend(int(trait[locus]).to_bytes(2, "little", signed=False))
        encoded.extend(int(trait[16 + locus]).to_bytes(2, "little", signed=False))
    return "sha256:" + hashlib.sha256(bytes(encoded)).hexdigest()


def validate_exact_natural_source_state_ref_v1(value: object) -> dict[str, Any]:
    """Validate one immutable exact source reference and return a defensive copy."""

    row = dict(_mapping(value, field="exact_natural_source_state_ref"))
    required = {
        "schema_version",
        "ref_id",
        "source",
        "world",
        "absolute_tick",
        "agent",
        "genome",
        "genome_state_id",
        "tracked_genotype_state_id",
        "state_class",
        "birth_provenance",
        "materialization",
    }
    if set(row) != required or row.get("schema_version") != EXACT_NATURAL_SOURCE_STATE_REF_SCHEMA_V1:
        raise ExactNaturalSourceStateError("source-state reference fields or schema differ")
    _sha256_id(row["ref_id"], field="ref_id")

    source = _mapping(row["source"], field="source")
    if set(source) != {"campaign_id", "run_id", "capture_index_path", "capture_index_sha256", "artifact_digests", "artifact_set_sha256"}:
        raise ExactNaturalSourceStateError("source provenance fields are not exact")
    for field in ("campaign_id", "run_id", "capture_index_path"):
        _text(source[field], field=f"source.{field}")
    _digest(source["capture_index_sha256"], field="source.capture_index_sha256")
    artifacts = _artifact_rows(source["artifact_digests"], field="source.artifact_digests")
    artifact_set_core = {"schema_version": "eiel_exact_natural_source_artifact_set_v1", "artifacts": list(artifacts)}
    if source["artifact_set_sha256"] != canonical_sha256(artifact_set_core).to_dict():
        raise ExactNaturalSourceStateError("source artifact-set digest differs")

    world = _mapping(row["world"], field="world")
    if set(world) != {"cell_id", "world_id", "source_seed"}:
        raise ExactNaturalSourceStateError("source world fields are not exact")
    _text(world["cell_id"], field="world.cell_id")
    _text(world["world_id"], field="world.world_id")
    _integer(world["source_seed"], field="world.source_seed")
    absolute_tick = _integer(row["absolute_tick"], field="absolute_tick")

    agent = _mapping(row["agent"], field="agent")
    if set(agent) != {"agent_id", "agent_state_id", "lineage_id", "lineage_depth", "parent_a_id", "parent_b_id"}:
        raise ExactNaturalSourceStateError("agent identity fields are not exact")
    _integer(agent["agent_id"], field="agent.agent_id", minimum=1)
    _sha256_id(agent["agent_state_id"], field="agent.agent_state_id")
    _sha256_id(agent["lineage_id"], field="agent.lineage_id")
    _integer(agent["lineage_depth"], field="agent.lineage_depth")
    _integer(agent["parent_a_id"], field="agent.parent_a_id")
    _integer(agent["parent_b_id"], field="agent.parent_b_id")

    genome = _genome_payload(row["genome"], field="genome")
    genome_bytes = bytes.fromhex(genome["policy_hex"] + genome["trait_hex"])
    expected_genome = full_genome_state_id_v1(genome_bytes)
    if row.get("genome_state_id") != expected_genome:
        raise ExactNaturalSourceStateError("genome state identity is absent or differs")
    tracked = row["tracked_genotype_state_id"]
    if tracked is not None:
        _sha256_id(tracked, field="tracked_genotype_state_id")

    state_class = _text(row["state_class"], field="state_class")
    if state_class not in EXACT_NATURAL_SOURCE_STATE_REF_CLASSES_V1:
        raise ExactNaturalSourceStateError("state_class is unsupported")
    provenance = _mapping(row["birth_provenance"], field="birth_provenance")
    if set(provenance) != {"kind", "birth_tick", "parent_agent_ids", "packed_hex", "mutation_step", "provenance_sha256"}:
        raise ExactNaturalSourceStateError("birth provenance fields are not exact")
    kind = _text(provenance["kind"], field="birth_provenance.kind")
    birth_tick = _integer(provenance["birth_tick"], field="birth_provenance.birth_tick")
    if birth_tick > absolute_tick:
        raise ExactNaturalSourceStateError("birth provenance occurs after the referenced state")
    parents = provenance["parent_agent_ids"]
    if not isinstance(parents, list) or len(parents) != 2 or any(type(item) is not int or item < 0 for item in parents):
        raise ExactNaturalSourceStateError("birth provenance parent IDs are invalid")
    packed_hex = provenance["packed_hex"]
    if not isinstance(packed_hex, str) or len(packed_hex) % 2 or any(character not in "0123456789abcdef" for character in packed_hex):
        raise ExactNaturalSourceStateError("birth provenance packed bytes are invalid")
    _integer(provenance["mutation_step"], field="birth_provenance.mutation_step", minimum=1)
    provenance_core = {key: provenance[key] for key in provenance if key != "provenance_sha256"}
    if provenance["provenance_sha256"] != canonical_sha256(provenance_core).to_dict():
        raise ExactNaturalSourceStateError("birth provenance digest differs")
    if kind == "tick_zero_genome_catalog_v1" and (birth_tick != 0 or parents != [0, 0]):
        raise ExactNaturalSourceStateError("tick-zero provenance is inconsistent")
    if kind == "authenticated_birth_replay_v1" and birth_tick < 1:
        raise ExactNaturalSourceStateError("replay provenance must be later-born")

    materialization = _mapping(row["materialization"], field="materialization")
    if set(materialization) != {"method", "materialization_profile_id"} or materialization.get("method") != EXACT_NATURAL_SOURCE_STATE_MATERIALIZATION_METHOD_V1:
        raise ExactNaturalSourceStateError("materialization fields are not exact")
    _text(materialization["materialization_profile_id"], field="materialization.materialization_profile_id")
    if materialization["materialization_profile_id"] != materialization_profile_id_v1(
        row["genome_state_id"]
    ):
        raise ExactNaturalSourceStateError(
            "materialization profile identity is not genome-content bound"
        )

    core = _canonical_ref_core(row)
    if row["ref_id"] != "sha256:" + canonical_sha256(core).hex:
        raise ExactNaturalSourceStateError("source-state reference digest differs")
    row["source"] = {**dict(source), "artifact_digests": [dict(item) for item in artifacts]}
    row["world"] = dict(world)
    row["agent"] = dict(agent)
    row["genome"] = genome
    row["birth_provenance"] = dict(provenance)
    row["materialization"] = dict(materialization)
    return row


def build_exact_natural_source_state_ref_v1(
    *,
    source: Mapping[str, Any],
    world: Mapping[str, Any],
    absolute_tick: int,
    agent: Mapping[str, Any],
    policy_bytes: bytes,
    trait_bytes: bytes,
    tracked_genotype_state_id: str | None,
    state_class: str,
    birth_provenance: Mapping[str, Any],
    materialization_profile_id: str | None = None,
) -> dict[str, Any]:
    """Build one reference from exact bytes; no genome transformation is done."""

    if not isinstance(policy_bytes, bytes) or len(policy_bytes) != 224:
        raise ExactNaturalSourceStateError("policy bytes must be exactly 224 bytes")
    if not isinstance(trait_bytes, bytes) or len(trait_bytes) != 64:
        raise ExactNaturalSourceStateError("trait bytes must be exactly 64 bytes")
    genome_bytes = policy_bytes + trait_bytes
    source_value = dict(source)
    artifacts = _artifact_rows(
        source_value.get("artifact_digests"), field="source.artifact_digests"
    )
    source_value["artifact_digests"] = [dict(item) for item in artifacts]
    source_value["artifact_set_sha256"] = canonical_sha256(
        {
            "schema_version": "eiel_exact_natural_source_artifact_set_v1",
            "artifacts": list(artifacts),
        }
    ).to_dict()
    provenance_core = dict(birth_provenance)
    provenance = {
        **provenance_core,
        "provenance_sha256": canonical_sha256(provenance_core).to_dict(),
    }
    genome_state_id = full_genome_state_id_v1(genome_bytes)
    if materialization_profile_id in (None, "auto"):
        materialization_profile_id = materialization_profile_id_v1(genome_state_id)
    core: dict[str, Any] = {
        "schema_version": EXACT_NATURAL_SOURCE_STATE_REF_SCHEMA_V1,
        "source": source_value,
        "world": dict(world),
        "absolute_tick": absolute_tick,
        "agent": dict(agent),
        "genome": {
            "encoding": EXACT_NATURAL_SOURCE_STATE_GENOME_ENCODING_V1,
            "policy_hex": policy_bytes.hex(),
            "trait_hex": trait_bytes.hex(),
        },
        "genome_state_id": genome_state_id,
        "tracked_genotype_state_id": tracked_genotype_state_id,
        "state_class": state_class,
        "birth_provenance": provenance,
        "materialization": {
            "method": EXACT_NATURAL_SOURCE_STATE_MATERIALIZATION_METHOD_V1,
            "materialization_profile_id": materialization_profile_id,
        },
    }
    core["ref_id"] = "sha256:" + canonical_sha256(core).hex
    return validate_exact_natural_source_state_ref_v1(core)


def materialization_profile_id_v1(identity_id: str) -> str:
    """Return a stable materializer ID bound to a full genome identity."""

    _sha256_id(identity_id, field="identity_id")
    return "exact-natural-" + identity_id[7:23]


def validate_exact_natural_source_state_bank_v1(value: object) -> dict[str, Any]:
    row = dict(_mapping(value, field="exact_natural_source_state_bank"))
    required = {"schema_version", "bank_id", "refs", "panels", "bank_sha256"}
    if set(row) != required or row.get("schema_version") != EXACT_NATURAL_SOURCE_STATE_BANK_SCHEMA_V1:
        raise ExactNaturalSourceStateError("source-state bank fields or schema differ")
    _text(row["bank_id"], field="bank_id")
    refs_raw = row["refs"]
    if not isinstance(refs_raw, list) or not refs_raw:
        raise ExactNaturalSourceStateError("source-state bank refs must be nonempty")
    refs = [validate_exact_natural_source_state_ref_v1(item) for item in refs_raw]
    ref_ids = [item["ref_id"] for item in refs]
    if ref_ids != sorted(set(ref_ids)):
        raise ExactNaturalSourceStateError("source-state bank refs must be sorted and unique")
    panels = row["panels"]
    if not isinstance(panels, Mapping):
        raise ExactNaturalSourceStateError("source-state bank panels must be an object")
    normalized_panels: dict[str, dict[str, str]] = {}
    for panel_id, raw_panel in panels.items():
        _text(panel_id, field="panels.<id>")
        panel = _mapping(raw_panel, field=f"panels.{panel_id}")
        if not panel or any(not isinstance(role, str) or not role for role in panel):
            raise ExactNaturalSourceStateError("source-state panel roles are invalid")
        normalized_panels[panel_id] = {
            role: _sha256_id(ref_id, field=f"panels.{panel_id}.{role}")
            for role, ref_id in sorted(panel.items())
        }
    ref_set = set(ref_ids)
    if any(ref_id not in ref_set for panel in normalized_panels.values() for ref_id in panel.values()):
        raise ExactNaturalSourceStateError("source-state panel references an unknown ref")
    core = {
        "schema_version": EXACT_NATURAL_SOURCE_STATE_BANK_SCHEMA_V1,
        "bank_id": row["bank_id"],
        "refs": refs,
        "panels": normalized_panels,
    }
    if row["bank_sha256"] != canonical_sha256(core).to_dict():
        raise ExactNaturalSourceStateError("source-state bank digest differs")
    row.update(core)
    return row


def build_exact_natural_source_state_bank_v1(
    refs: Sequence[Mapping[str, Any]],
    *,
    bank_id: str,
    panels: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    validated = [validate_exact_natural_source_state_ref_v1(ref) for ref in refs]
    validated.sort(key=lambda item: item["ref_id"])
    core = {
        "schema_version": EXACT_NATURAL_SOURCE_STATE_BANK_SCHEMA_V1,
        "bank_id": _text(bank_id, field="bank_id"),
        "refs": validated,
        "panels": {
            panel_id: dict(sorted(panel.items()))
            for panel_id, panel in sorted((panels or {}).items())
        },
    }
    return validate_exact_natural_source_state_bank_v1(
        {**core, "bank_sha256": canonical_sha256(core).to_dict()}
    )


def load_exact_natural_source_state_bank_v1(
    path: str | Path,
    *,
    expected_file_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    source = Path(path).expanduser()
    if source.is_symlink() or not source.resolve().is_file():
        raise ExactNaturalSourceStateError(f"source-state bank is not a regular file: {source}")
    source = source.resolve()
    raw_bytes = source.read_bytes()
    file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if expected_file_sha256 is not None and file_sha256 != expected_file_sha256:
        raise ExactNaturalSourceStateError("source-state bank file digest differs")
    try:
        value = json.loads(raw_bytes.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ExactNaturalSourceStateError("source-state bank is not canonical JSON") from error
    if canonical_json_bytes(value) != raw_bytes:
        raise ExactNaturalSourceStateError("source-state bank is not canonical JSON")
    return validate_exact_natural_source_state_bank_v1(value), file_sha256


def resolve_exact_natural_source_state_v1(
    bank: Mapping[str, Any],
    source_state_ref: str,
    *,
    role: str | None = None,
) -> dict[str, Any]:
    validated = validate_exact_natural_source_state_bank_v1(bank)
    refs = {row["ref_id"]: row for row in validated["refs"]}
    candidate = source_state_ref
    if candidate in validated["panels"]:
        if role is None or role not in validated["panels"][candidate]:
            raise ExactNaturalSourceStateError(
                "source-state panel requires an explicit existing role"
            )
        candidate = validated["panels"][candidate][role]
    elif role is not None:
        raise ExactNaturalSourceStateError(
            "a direct exact source reference cannot carry a role selector"
        )
    ref = refs.get(candidate)
    if ref is None:
        raise ExactNaturalSourceStateError("source-state reference is absent from its bank")
    return dict(ref)


def authenticate_exact_natural_source_state_ref_v1(
    ref: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate listed source files; path alone never grants identity."""

    validated = validate_exact_natural_source_state_ref_v1(ref)
    for index, artifact in enumerate(validated["source"]["artifact_digests"]):
        path = Path(artifact["path"]).expanduser()
        if path.is_symlink() or not path.resolve().is_file():
            raise ExactNaturalSourceStateError(
                f"source artifact {index} is not a regular file"
            )
        observed = hashlib.sha256(path.resolve().read_bytes()).hexdigest()
        if observed != artifact["sha256"]:
            raise ExactNaturalSourceStateError(
                f"source artifact {index} digest differs"
            )
    capture_path = Path(validated["source"]["capture_index_path"]).expanduser()
    if capture_path.is_symlink() or not capture_path.resolve().is_file():
        raise ExactNaturalSourceStateError("source capture index is not a regular file")
    if hashlib.sha256(capture_path.resolve().read_bytes()).hexdigest() != validated["source"]["capture_index_sha256"]["hex"]:
        raise ExactNaturalSourceStateError("source capture index digest differs")
    return validated


__all__ = (
    "EXACT_NATURAL_SOURCE_STATE_BANK_SCHEMA_V1",
    "EXACT_NATURAL_SOURCE_STATE_GENOME_ENCODING_V1",
    "EXACT_NATURAL_SOURCE_STATE_MATERIALIZATION_METHOD_V1",
    "EXACT_NATURAL_SOURCE_STATE_REF_CLASSES_V1",
    "EXACT_NATURAL_SOURCE_STATE_REF_SCHEMA_V1",
    "ExactNaturalSourceStateError",
    "authenticate_exact_natural_source_state_ref_v1",
    "build_exact_natural_source_state_bank_v1",
    "build_exact_natural_source_state_ref_v1",
    "full_genome_bytes_v1",
    "full_genome_state_id_v1",
    "load_exact_natural_source_state_bank_v1",
    "materialization_profile_id_v1",
    "resolve_exact_natural_source_state_v1",
    "tracked_genotype_state_id_v1",
    "validate_exact_natural_source_state_bank_v1",
    "validate_exact_natural_source_state_ref_v1",
)
