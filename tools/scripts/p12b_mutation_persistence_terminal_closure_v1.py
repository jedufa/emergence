#!/usr/bin/env python3
"""Reduce P12B ordinary-birth mutation persistence at G30.

The reducer is post-hoc over authoritative checkpoint projections.  It treats
mutation origin as a paired-control inference: a low-mutation cell state must
be absent from its same-seed/profile mutation-off control and absent from G0
founder states, then persist in a G30 descendant chain of the event child.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.campaigns import p12a_gv2_persistence_reducer as p12a  # noqa: E402
from engine.campaigns import p12b_compact_descendant_oracle as oracle  # noqa: E402
from engine.campaigns.canonical import canonical_sha256, pretty_json  # noqa: E402
from tools.scripts import p12b_compact_terminal_closure_v1 as p12b_terminal  # noqa: E402


SCHEMA_VERSION = "p12b_mutation_persistence_terminal_closure_reduction_v1"
SCIENTIFIC_MILESTONE = 30
TECHNICAL_SEGMENTS = (5, 10, 15, 20, 25, 30)
POSITIVE_IDENTITY_TYPES = (
    "allele_genotype_state_identity",
    "preregistered_package_identity",
)


class P12BMutationPersistenceError(RuntimeError):
    """Campaign root cannot be reduced to a mutation persistence decision."""


def _fail(message: str) -> None:
    raise P12BMutationPersistenceError(message)


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise P12BMutationPersistenceError(f"cannot read {name}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"{name} must be a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cell_rows(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    cells = plan.get("cells")
    if not isinstance(cells, list):
        _fail("execution plan lacks cells")
    rows = {
        str(cell["cell_id"]): cell
        for cell in cells
        if isinstance(cell, Mapping) and type(cell.get("cell_id")) is str
    }
    if len(rows) != 24:
        _fail(f"expected 24 logical cells, got {len(rows)}")
    return rows


def _state_for_slot(
    projection: Mapping[str, Any],
    carrier: Mapping[str, int],
    *,
    identity_type: str,
    capacity: int,
) -> dict[str, Any]:
    try:
        return p12a._state_definition(
            projection,
            carrier,
            identity_type=identity_type,
            capacity=capacity,
        )
    except p12a.P12APersistenceReducerError as exc:
        raise P12BMutationPersistenceError(str(exc)) from exc


def _state_hash(identity_type: str, fields: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "schema_version": "p12b_mutation_persistence_state_identity_v1",
            "identity_type": identity_type,
            "matched_fields": fields,
        }
    )


def _event_state_hashes(event_rows: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    result = {identity_type: set() for identity_type in POSITIVE_IDENTITY_TYPES}
    for event in event_rows:
        fields_by_identity = event.get("child_identity_fields")
        if not isinstance(fields_by_identity, Mapping):
            _fail("event row lacks child_identity_fields")
        for identity_type in POSITIVE_IDENTITY_TYPES:
            fields = fields_by_identity.get(identity_type)
            if not isinstance(fields, Mapping):
                _fail(f"event row lacks child {identity_type}")
            result[identity_type].add(_state_hash(identity_type, fields))
    return result


def _terminal_state_hashes(
    target: Mapping[str, Any],
    *,
    capacity: int,
) -> dict[str, set[str]]:
    carriers = _terminal_carrier_index(target, capacity=capacity)
    return {
        identity_type: set(rows)
        for identity_type, rows in carriers.items()
    }


def _terminal_carrier_index(
    target: Mapping[str, Any],
    *,
    capacity: int,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result: dict[str, dict[str, list[dict[str, Any]]]] = {
        identity_type: {} for identity_type in POSITIVE_IDENTITY_TYPES
    }
    for carrier in p12a._live_slots(target):
        for identity_type in POSITIVE_IDENTITY_TYPES:
            state = _state_for_slot(
                target, carrier, identity_type=identity_type, capacity=capacity
            )
            state_hash = _state_hash(identity_type, state["matched_fields"])
            result[identity_type].setdefault(state_hash, []).append(
                {
                    "carrier_stable_id": int(carrier["agent_id"]),
                    "carrier_slot": int(carrier["slot"]),
                    "lineage_depth": int(carrier["lineage_depth"]),
                    "lineage_founder_id": int(carrier["lineage_founder_id"]),
                    "selected_values": state["matched_fields"],
                    "exact_values": {
                        "exact_hereditary_digest": state["exact_hereditary_digest"]
                    },
                }
            )
    return result


def _ancestor_ids(
    *,
    agent_id: int,
    birth_by_child: Mapping[int, Mapping[str, Any]],
) -> set[int]:
    ancestors: set[int] = set()
    seen: set[int] = set()
    stack = [agent_id]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        event = birth_by_child.get(current)
        if event is None or event.get("event_kind") != "ordinary_birth":
            continue
        parents = (
            int(event["parent_stable_id"]),
            int(event["mate_stable_id"]),
        )
        ancestors.update(parents)
        stack.extend(parents)
    ancestors.discard(agent_id)
    return ancestors


def _ancestor_index(
    carriers_by_identity: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    *,
    birth_by_child: Mapping[int, Mapping[str, Any]],
) -> dict[int, set[int]]:
    agents = {
        int(carrier["carrier_stable_id"])
        for states in carriers_by_identity.values()
        for carriers in states.values()
        for carrier in carriers
    }
    return {
        agent_id: _ancestor_ids(agent_id=agent_id, birth_by_child=birth_by_child)
        for agent_id in agents
    }


def _terminal_descendant_carriers(
    *,
    carriers_by_hash: Mapping[str, Sequence[Mapping[str, Any]]],
    ancestor_by_agent: Mapping[int, set[int]],
    identity_type: str,
    fields: Mapping[str, Any],
    event_child_id: int,
) -> list[dict[str, Any]]:
    state_hash = _state_hash(identity_type, fields)
    carriers: list[dict[str, Any]] = []
    for carrier in carriers_by_hash.get(state_hash, []):
        agent_id = int(carrier["carrier_stable_id"])
        if event_child_id not in ancestor_by_agent.get(agent_id, set()):
            continue
        carriers.append(dict(carrier))
    return carriers


def _reduce_cell(
    *,
    run_root: Path,
    batch_manifest_path: Path,
    plan: Mapping[str, Any],
    authority: Mapping[str, Any],
    source_commit: str,
    output_root: Path,
) -> dict[str, Any]:
    batch_manifest = _read_json(batch_manifest_path, name="batch manifest")
    cell_ids = batch_manifest.get("cell_ids")
    if not isinstance(cell_ids, list) or len(cell_ids) != 1 or type(cell_ids[0]) is not str:
        _fail("P12B mutation persistence expects physical b1 batch manifests")
    cell_id = str(cell_ids[0])
    cell = _cell_rows(plan)[cell_id]
    dimensions = cell.get("dimensions")
    parameters = cell.get("parameters")
    if not isinstance(dimensions, Mapping) or not isinstance(parameters, Mapping):
        _fail(f"cell {cell_id} lacks dimensions or parameters")
    seed = int(dimensions["ecology_seed"])
    profile = str(dimensions["founder_profile"])
    rate_q10 = int(parameters["genome_upgrade_v2.mutation_rate_q10"])
    step_q10 = int(parameters["genome_upgrade_v2.mutation_step_q10"])
    mutation_regime = "low_q10" if rate_q10 > 0 and step_q10 > 0 else "off"
    sidecars = p12b_terminal._checkpoint_sidecars(batch_manifest, cell_id=cell_id)
    result = oracle.reduce_compact_descendant_oracle(
        g0_path=sidecars[0],
        target_path=sidecars[SCIENTIFIC_MILESTONE],
        segment_paths=tuple(sidecars[generation] for generation in TECHNICAL_SEGMENTS),
        target_milestone=SCIENTIFIC_MILESTONE,
        authority=authority,
        source_commit=source_commit,
        campaign_id=str(plan["campaign_id"]),
        cell_id=cell_id,
        seed=seed,
        evidence_root=run_root,
    )
    written = oracle.write_oracle_outputs(
        result, output_root / "cells" / cell_id / "g30"
    )
    g0 = oracle._projection(sidecars[0], milestone=0)
    target = oracle._projection(sidecars[SCIENTIFIC_MILESTONE], milestone=30)
    capacity = len(p12a._tensor(g0, "alive_mask")[0])
    founder_hashes: dict[str, set[str]] = {
        identity_type: set() for identity_type in POSITIVE_IDENTITY_TYPES
    }
    for founder in p12a._live_slots(g0):
        for identity_type in POSITIVE_IDENTITY_TYPES:
            state = _state_for_slot(
                g0, founder, identity_type=identity_type, capacity=capacity
            )
            founder_hashes[identity_type].add(
                _state_hash(identity_type, state["matched_fields"])
            )
    event_rows = result["source_event_rows"]
    birth_by_child = {
        int(row["child_stable_id"]): row
        for row in event_rows
        if row.get("event_kind") == "ordinary_birth"
    }
    terminal_carriers = _terminal_carrier_index(target, capacity=capacity)
    ancestor_by_agent = _ancestor_index(
        terminal_carriers, birth_by_child=birth_by_child
    )
    candidate_rows: list[dict[str, Any]] = []
    for event in event_rows:
        if event.get("event_kind") != "ordinary_birth":
            continue
        fields_by_identity = event["child_identity_fields"]
        for identity_type in POSITIVE_IDENTITY_TYPES:
            fields = fields_by_identity[identity_type]
            state_hash = _state_hash(identity_type, fields)
            descendant_carriers = _terminal_descendant_carriers(
                carriers_by_hash=terminal_carriers[identity_type],
                ancestor_by_agent=ancestor_by_agent,
                identity_type=identity_type,
                fields=fields,
                event_child_id=int(event["child_stable_id"]),
            )
            candidate_rows.append(
                {
                    "cell_id": cell_id,
                    "seed": seed,
                    "founder_profile": profile,
                    "mutation_regime": mutation_regime,
                    "mutation_rate_q10": rate_q10,
                    "mutation_step_q10": step_q10,
                    "identity_type": identity_type,
                    "state_hash": state_hash,
                    "event_transaction_id": event["transaction_id"],
                    "event_generation": event["generation"],
                    "event_child_stable_id": event["child_stable_id"],
                    "novel_vs_g0_founders": state_hash not in founder_hashes[identity_type],
                    "terminal_descendant_carrier_count": len(descendant_carriers),
                    "terminal_descendant_carriers": descendant_carriers[:8],
                }
            )
    return {
        "cell_id": cell_id,
        "batch_manifest_path": str(batch_manifest_path),
        "profile": profile,
        "seed": seed,
        "mutation_regime": mutation_regime,
        "mutation_rate_q10": rate_q10,
        "mutation_step_q10": step_q10,
        "checkpoint_sidecars": {
            str(generation): {"path": str(path), "sha256": _sha256(path)}
            for generation, path in sorted(sidecars.items())
            if generation in {0, *TECHNICAL_SEGMENTS}
        },
        "oracle": {
            "result_sha256": result["result_sha256"],
            "summary": result["summary"],
            "output_paths": written,
        },
        "founder_state_hashes": {
            identity_type: sorted(values)
            for identity_type, values in founder_hashes.items()
        },
        "event_state_hashes": {
            identity_type: sorted(values)
            for identity_type, values in _event_state_hashes(event_rows).items()
        },
        "terminal_state_hashes": {
            identity_type: sorted(values)
            for identity_type, values in (
                (identity_type, set(rows))
                for identity_type, rows in terminal_carriers.items()
            )
        },
        "candidate_rows": candidate_rows,
    }


def _paired_off_index(
    cell_results: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str], Mapping[str, Any]]:
    result: dict[tuple[int, str], Mapping[str, Any]] = {}
    for cell in cell_results:
        if cell["mutation_regime"] != "off":
            continue
        key = (int(cell["seed"]), str(cell["profile"]))
        if key in result:
            _fail(f"duplicate off-control cell for seed/profile {key!r}")
        result[key] = cell
    return result


def _classify_candidates(
    cell_results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    off_index = _paired_off_index(cell_results)
    classified: list[dict[str, Any]] = []
    for cell in cell_results:
        if cell["mutation_regime"] != "low_q10":
            continue
        pair = off_index.get((int(cell["seed"]), str(cell["profile"])))
        if pair is None:
            _fail(f"missing paired off-control for {cell['cell_id']}")
        for row in cell["candidate_rows"]:
            identity_type = str(row["identity_type"])
            state_hash = str(row["state_hash"])
            off_event = set(pair["event_state_hashes"][identity_type])
            off_terminal = set(pair["terminal_state_hashes"][identity_type])
            mutation_unique = (
                row["novel_vs_g0_founders"]
                and state_hash not in off_event
                and state_hash not in off_terminal
            )
            positive = (
                mutation_unique
                and int(row["terminal_descendant_carrier_count"]) > 0
            )
            classified.append(
                {
                    **row,
                    "paired_off_cell_id": pair["cell_id"],
                    "absent_from_paired_off_events": state_hash not in off_event,
                    "absent_from_paired_off_terminal": state_hash not in off_terminal,
                    "mutation_unique_against_pair": mutation_unique,
                    "positive_mutation_persistence": positive,
                }
            )
    return classified


def _decision(
    *,
    cell_results: Sequence[Mapping[str, Any]],
    comparisons: Mapping[str, Any],
    classified_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    failures: list[str] = []
    if len(cell_results) != 24:
        failures.append("primary_cells_not_24")
    regimes = sorted({str(cell["mutation_regime"]) for cell in cell_results})
    profiles = sorted({str(cell["profile"]) for cell in cell_results})
    seeds = sorted({int(cell["seed"]) for cell in cell_results})
    if regimes != ["low_q10", "off"]:
        failures.append(f"mutation_regimes_not_off_and_low:{regimes}")
    if len(profiles) != 2:
        failures.append(f"profile_count_not_2:{len(profiles)}")
    if len(seeds) != 6:
        failures.append(f"seed_count_not_6:{len(seeds)}")
    for cell in cell_results:
        if cell["oracle"]["summary"].get("evaluable") is not True:
            failures.append(f"{cell['cell_id']}:oracle_not_evaluable")
    if comparisons.get("ok") is False:
        failures.append("comparison_integrity_failed")
    positives = [
        row for row in classified_rows if row.get("positive_mutation_persistence") is True
    ]
    positive_profiles = sorted({str(row["founder_profile"]) for row in positives})
    positive_seeds = sorted({int(row["seed"]) for row in positives})
    positive_identity_types = sorted({str(row["identity_type"]) for row in positives})
    if failures:
        decision = "P12B-OB-MUTATION-PERSIST-B24-G30-CONFIRMATION-E"
        label = "INTEGRITY_FAILURE"
    elif len(positive_profiles) == 2:
        decision = "P12B-OB-MUTATION-PERSIST-B24-G30-CONFIRMATION-A"
        label = "MUTATION_PERSISTENCE_CONFIRMED_BOTH_PROFILES"
    elif positives:
        decision = "P12B-OB-MUTATION-PERSIST-B24-G30-CONFIRMATION-B"
        label = "MUTATION_PERSISTENCE_MIXED_OR_PROFILE_CONDITIONAL"
    else:
        decision = "P12B-OB-MUTATION-PERSIST-B24-G30-CONFIRMATION-C"
        label = "MUTATION_PERSISTENCE_NOT_SUPPORTED"
    return {
        "decision": decision,
        "decision_label": label,
        "integrity_failures": failures,
        "positive_mutation_persistence_event_count": len(positives),
        "positive_profiles": positive_profiles,
        "positive_seeds": positive_seeds,
        "positive_identity_types": positive_identity_types,
        "classified_low_mutation_event_identity_rows": len(classified_rows),
    }


def reduce_campaign(
    *,
    run_root: Path,
    authority_path: Path,
    source_commit: str,
    output_root: Path,
) -> dict[str, Any]:
    plan = _read_json(run_root / "execution_plan.json", name="execution plan")
    authority = _read_json(authority_path, name="authority binding")
    output_root.mkdir(parents=True, exist_ok=True)
    cell_results = [
        _reduce_cell(
            run_root=run_root,
            batch_manifest_path=path,
            plan=plan,
            authority=authority,
            source_commit=source_commit,
            output_root=output_root,
        )
        for path in p12b_terminal._batch_manifest_paths(run_root)
    ]
    comparisons = p12b_terminal._comparison_integrity(run_root=run_root, plan=plan)
    classified_rows = _classify_candidates(cell_results)
    decision = _decision(
        cell_results=cell_results,
        comparisons=comparisons,
        classified_rows=classified_rows,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "run_root": str(run_root),
        "campaign_id": plan.get("campaign_id"),
        "plan_sha256": plan.get("plan_sha256"),
        "source_commit": source_commit,
        "authority_path": str(authority_path),
        "authority_sha256": _sha256(authority_path),
        "scientific_milestone": SCIENTIFIC_MILESTONE,
        "technical_segments": list(TECHNICAL_SEGMENTS),
        "positive_identity_types": list(POSITIVE_IDENTITY_TYPES),
        "cell_results": cell_results,
        "classified_low_mutation_rows": classified_rows,
        "comparisons": comparisons,
        "decision": decision,
    }
    result["result_sha256"] = canonical_sha256(result)
    (output_root / "p12b_mutation_persistence_reduction_v1.json").write_text(
        pretty_json(result), encoding="utf-8"
    )
    (output_root / "p12b_mutation_persistence_classified_rows_v1.json").write_text(
        pretty_json({"rows": classified_rows}), encoding="utf-8"
    )
    (output_root / "p12b_mutation_persistence_decision_v1.json").write_text(
        pretty_json({**decision, "result_sha256": result["result_sha256"]}),
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = reduce_campaign(
        run_root=args.run_root.expanduser().resolve(),
        authority_path=args.authority.expanduser().resolve(),
        source_commit=args.source_commit,
        output_root=args.output_root.expanduser().resolve(),
    )
    print(
        pretty_json(
            {
                "ok": True,
                "decision": result["decision"],
                "result_sha256": result["result_sha256"],
                "output_root": str(args.output_root),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
