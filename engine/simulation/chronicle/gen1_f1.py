from __future__ import annotations

"""Passive GEN1-F1 canonical birth and carrier capture for Python births."""

import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping

from agents.modular_genome_v1 import (
    canonical_hash_from_genome_dict,
    canonical_payload_from_genome_dict,
)
from simulation.chronicle.genome_attribution import (
    GenomeAttributionAccumulator,
    write_attribution_outputs,
)
from simulation.chronicle.schema import canonical_json, canonical_payload, stable_hash
from simulation.chronicle.storage import read_json, write_json, write_jsonl


SCHEMA_VERSION = "eiel_gen1_f1_capture_v1"
STATE_SCHEMA_VERSION = "eiel_gen1_f1_capture_state_v1"
BIRTH_SCHEMA_VERSION = "eiel_gen1_f1_birth_record_v1"
CARRIER_SCHEMA_VERSION = "eiel_gen1_f1_carrier_observation_v1"


def _source_commit() -> str:
    for candidate in Path(__file__).resolve().parents:
        if not (candidate / ".git").exists():
            continue
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=candidate,
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except Exception:
            break
        if result.stdout.strip():
            return result.stdout.strip()
    return "unknown"


def _genome_payload(genome: Any) -> dict[str, Any]:
    if isinstance(genome, Mapping):
        raw = dict(genome)
    else:
        canonical = getattr(genome, "canonical_payload", None)
        if callable(canonical):
            raw = canonical()
        else:
            to_dict = getattr(genome, "to_dict", None)
            if not callable(to_dict):
                raise ValueError("canonical genome payload unavailable")
            raw = to_dict()
    return canonical_payload_from_genome_dict(dict(raw))


def _lineage_id(agent: Any) -> str:
    stats = getattr(agent, "stats", None)
    value = str(getattr(stats, "lineage_id", "") or "")
    if value:
        return value
    return f"L{int(getattr(agent, 'agent_id', 0)):08d}"


class Gen1F1Capture:
    """RNG-free, duplicate-safe run-local GEN1-F1 collector."""

    def __init__(self, sim: Any) -> None:
        sim_cfg = getattr(sim, "sim_cfg", None)
        self.run_dir = Path(str(getattr(sim, "run_dir", "")))
        self.root = self.run_dir / "chronicle" / "gen1_f1"
        self.objects_root = self.root / "objects" / "genomes"
        self.run_id = str(getattr(sim_cfg, "chronicle_run_id", "") or self.run_dir.name or "run")
        self.campaign_id = str(getattr(sim_cfg, "campaign_id", "") or "") or None
        self.branch_id = str(getattr(sim_cfg, "branch_id", "") or "") or None
        self.source_commit = str(
            getattr(sim_cfg, "code_commit_hash", "")
            or os.environ.get("EIEL_GIT_COMMIT", "")
            or _source_commit()
        )
        self.seed = getattr(sim_cfg, "seed", "unknown")
        self.scenario = str(getattr(sim_cfg, "scenario", "") or getattr(sim_cfg, "world", "") or "unknown")
        self.route_hash = str(
            getattr(sim_cfg, "execution_route_hash", "")
            or getattr(sim_cfg, "route_hash", "")
            or ""
        )
        self._births: dict[str, dict[str, Any]] = {}
        self._carriers: dict[str, dict[str, Any]] = {}
        self._accumulator = GenomeAttributionAccumulator()
        self._cursor = 0
        self.capture_errors = 0
        self._persist(include_attribution=False)

    @classmethod
    def maybe_create(cls, sim: Any) -> "Gen1F1Capture | None":
        cfg = getattr(sim, "sim_cfg", None)
        return cls(sim) if bool(getattr(cfg, "gen1_f1_capture_enable", False)) else None

    def _store_genome(self, payload: Mapping[str, Any]) -> tuple[str, str]:
        normalized = canonical_payload_from_genome_dict(dict(payload))
        digest = canonical_hash_from_genome_dict(normalized)
        relative = f"objects/genomes/{digest}.json"
        path = self.root / relative
        if path.exists():
            if canonical_json(normalized) != path.read_text(encoding="utf-8").strip():
                raise ValueError(f"GEN1-F1 genome object collision: {digest}")
        else:
            write_json(path, normalized)
        return digest, relative

    def record_birth(
        self,
        *,
        parent_a: Any,
        parent_b: Any,
        child: Any,
        generation: int,
        step: int,
        population_size: int | None = None,
    ) -> str | None:
        """Capture one authoritative birth, or fail closed if stage truth is absent."""

        stages = getattr(getattr(child, "genome", None), "_gen1_f1_canonical_stages", None)
        if not isinstance(stages, Mapping):
            self.capture_errors += 1
            return None
        try:
            parent_payloads = [_genome_payload(parent_a.genome), _genome_payload(parent_b.genome)]
            pre_payload = _genome_payload(stages["child_pre_mutation_genome"])
            post_payload = _genome_payload(stages["child_post_mutation_genome"])
            final_payload = _genome_payload(child.genome)
            parent_objects = [self._store_genome(value) for value in parent_payloads]
            pre_object = self._store_genome(pre_payload)
            post_object = self._store_genome(post_payload)
            final_object = self._store_genome(final_payload)
            parent_ids = [int(parent_a.agent_id), int(parent_b.agent_id)]
            child_id = int(child.agent_id)
            identity = {
                "run_id": self.run_id,
                "campaign_id": self.campaign_id,
                "branch_id": self.branch_id,
                "generation": int(generation),
                "step": int(step),
                "parent_agent_ids": sorted(parent_ids),
                "child_agent_id": child_id,
                "parent_genome_hashes": sorted(value[0] for value in parent_objects),
                "child_pre_mutation_genome_hash": pre_object[0],
                "child_post_mutation_genome_hash": post_object[0],
                "final_attached_child_genome_hash": final_object[0],
                "stage_boundary": str(stages.get("stage_boundary") or ""),
            }
            birth_id = "gen1f1_birth_" + stable_hash("eiel-gen1-f1-birth-v1", identity)
            source_ref = f"gen1_f1_birth_records.jsonl#{birth_id}"
            raw_source = {
                **identity,
                "source_commit": self.source_commit,
                "reproduction_phase": "ordinary_python_birth",
                "lineage_id": _lineage_id(child),
                "source_record_refs": [source_ref],
                "parent_genomes": parent_payloads,
                "child_pre_mutation_genome": pre_payload,
                "child_post_mutation_genome": post_payload,
                "final_attached_child_genome": final_payload,
                "mutation_operator": str(stages.get("mutation_operator") or "unknown_unavailable"),
                "source_completeness": (
                    "canonical_parent_and_post_c2_pre_c1_post_c1_final_stages;"
                    "scalar_recombination_mutation_preimage_not_separately_exposed"
                ),
                "population_size": population_size,
                "scenario": self.scenario,
                "seed": self.seed,
            }
            source_record_id = self._accumulator.add(raw_source)
            record = canonical_payload(
                {
                    "schema_version": BIRTH_SCHEMA_VERSION,
                    "birth_event_id": birth_id,
                    "source_record_id": source_record_id,
                    **identity,
                    "source_commit": self.source_commit,
                    "execution_route_hash": self.route_hash or None,
                    "reproduction_phase": "ordinary_python_birth",
                    "lineage_id": _lineage_id(child),
                    "source_record_refs": [source_ref],
                    "mutation_operator": raw_source["mutation_operator"],
                    "stage_boundary": stages.get("stage_boundary"),
                    "parent_genome_object_refs": [value[1] for value in parent_objects],
                    "child_pre_mutation_genome_object_ref": pre_object[1],
                    "child_post_mutation_genome_object_ref": post_object[1],
                    "final_attached_child_genome_object_ref": final_object[1],
                    "source_completeness": raw_source["source_completeness"],
                    "causal_status": "not_assessed",
                }
            )
            encoded = canonical_json(record)
            if birth_id in self._births:
                if canonical_json(self._births[birth_id]) != encoded:
                    raise ValueError(f"GEN1-F1 birth identity collision: {birth_id}")
                return birth_id
            self._births[birth_id] = record
            self._cursor += 1
            return birth_id
        except Exception:
            self.capture_errors += 1
            return None

    def observe_carriers(self, *, generation: int, agents: Iterable[Any]) -> None:
        """Record canonical genome identities for living agents at a boundary."""

        attribution = self._accumulator.build()
        source_ids_by_final_hash: dict[str, list[str]] = {}
        for source in attribution["source_records"]:
            final_hash = str(dict(source.get("child_final", {}) or {}).get("hash") or "")
            if final_hash:
                source_ids_by_final_hash.setdefault(final_hash, []).append(
                    str(source["source_record_id"])
                )
        mutation_ids_by_source: dict[str, list[str]] = {}
        for event in attribution["mutation_events"]:
            mutation_ids_by_source.setdefault(str(event["source_record_id"]), []).append(
                str(event["mutation_event_id"])
            )
        for agent in sorted(agents, key=lambda value: int(getattr(value, "agent_id", -1))):
            try:
                payload = _genome_payload(agent.genome)
                genome_hash, object_ref = self._store_genome(payload)
                source_record_ids = sorted(source_ids_by_final_hash.get(genome_hash, []))
                mutation_event_ids = sorted(
                    {
                        event_id
                        for source_id in source_record_ids
                        for event_id in mutation_ids_by_source.get(source_id, [])
                    }
                )
                identity = {
                    "run_id": self.run_id,
                    "generation": int(generation),
                    "agent_id": int(agent.agent_id),
                    "genome_hash": genome_hash,
                }
                observation_id = "gen1f1_carrier_" + stable_hash(
                    "eiel-gen1-f1-carrier-v1", identity
                )
                self._carriers.setdefault(
                    observation_id,
                    canonical_payload(
                        {
                            "schema_version": CARRIER_SCHEMA_VERSION,
                            "carrier_observation_id": observation_id,
                            **identity,
                            "lineage_id": _lineage_id(agent),
                            "genome_object_ref": object_ref,
                            "source_record_ids": source_record_ids,
                            "mutation_event_ids": mutation_event_ids,
                            "carrier_status": (
                                "exact_final_genome_hash_carrier"
                                if mutation_event_ids
                                else "no_exact_recorded_mutation_join"
                            ),
                            "evidence_strength": (
                                "persisted" if mutation_event_ids else "observed"
                            ),
                            "causal_status": "not_assessed",
                        }
                    ),
                )
            except Exception:
                self.capture_errors += 1
        self._persist(include_attribution=True)

    def flush(self) -> None:
        """Publish the current bounded generation snapshot and attribution view."""

        self._persist(include_attribution=True)

    def _persist(self, *, include_attribution: bool) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        write_jsonl(self.root / "gen1_f1_birth_records.jsonl", self._births.values())
        write_jsonl(self.root / "gen1_f1_carrier_records.jsonl", self._carriers.values())
        if include_attribution:
            write_attribution_outputs(self._accumulator, self.root / "attribution")
        write_json(
            self.root / "gen1_f1_manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "campaign_id": self.campaign_id,
                "birth_record_count": len(self._births),
                "carrier_observation_count": len(self._carriers),
                "cursor": self._cursor,
                "capture_errors": self.capture_errors,
                "canonical_genome_object_count": (
                    len(list(self.objects_root.glob("*.json"))) if self.objects_root.exists() else 0
                ),
                "limitations": [
                    "The Python slice captures the exact post-C2/pre-C1 to post-C1 canonical mutation boundary.",
                    "Scalar recombination and mutation occur before that boundary and do not yet have a separate preimage.",
                    "Native-authority births are not captured by this Python slice.",
                    "Carrier observations and passive attribution do not establish causal benefit.",
                ],
            },
        )

    def export_state(self) -> dict[str, Any]:
        genome_objects = []
        if self.objects_root.exists():
            for path in sorted(self.objects_root.glob("*.json")):
                payload = read_json(path)
                digest = canonical_hash_from_genome_dict(payload)
                if path.stem != digest:
                    raise ValueError(f"GEN1-F1 genome object filename/hash mismatch: {path}")
                genome_objects.append({"sha256": digest, "payload": payload})
        return canonical_payload(
            {
                "schema_version": STATE_SCHEMA_VERSION,
                "cursor": self._cursor,
                "birth_records": list(self._births.values()),
                "carrier_records": list(self._carriers.values()),
                "attribution_state": self._accumulator.export_state(),
                "genome_objects": genome_objects,
                "capture_errors": self.capture_errors,
            }
        )

    def import_state(self, state: Mapping[str, Any]) -> None:
        if str(state.get("schema_version")) != STATE_SCHEMA_VERSION:
            raise ValueError("unsupported GEN1-F1 capture state")
        births = {
            str(value["birth_event_id"]): canonical_payload(dict(value))
            for value in state.get("birth_records", []) or []
            if isinstance(value, Mapping) and value.get("birth_event_id")
        }
        carriers = {
            str(value["carrier_observation_id"]): canonical_payload(dict(value))
            for value in state.get("carrier_records", []) or []
            if isinstance(value, Mapping) and value.get("carrier_observation_id")
        }
        accumulator = GenomeAttributionAccumulator()
        accumulator.import_state(dict(state.get("attribution_state", {}) or {}))
        for value in state.get("genome_objects", []) or []:
            if not isinstance(value, Mapping) or not isinstance(value.get("payload"), Mapping):
                raise ValueError("invalid GEN1-F1 checkpoint genome object")
            digest, _relative = self._store_genome(dict(value["payload"]))
            if digest != str(value.get("sha256") or ""):
                raise ValueError("GEN1-F1 checkpoint genome object hash mismatch")
        expected_cursor = int(state.get("cursor", len(births)))
        if expected_cursor != len(births):
            raise ValueError("GEN1-F1 checkpoint cursor does not match birth records")
        self._births = births
        self._carriers = carriers
        self._accumulator = accumulator
        self._cursor = expected_cursor
        self.capture_errors = int(state.get("capture_errors", 0))
        self._persist(include_attribution=True)
