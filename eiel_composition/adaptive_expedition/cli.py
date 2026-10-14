"""Operator-facing Adaptive Expedition V1 commands.

The default ``run`` command consumes one immutable manifest and owns the
repeated coordination loop through the existing Science Request front door. It
does not dispatch providers directly, create execution authority, grant
Qualification, or approximate historical branching.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any, Final

from emergence_contracts import canonical_json_bytes
from emergence_contracts.adaptive_science_expedition_v1 import (
    DecisionV1,
    EvolutionaryAtlasV1,
    ExpeditionManifestV1,
    ObservationFrameV1,
    RetainedParentBankV1,
)
from emergence_contracts.canonical import canonical_sha256
from emergence_contracts.errors import ContractError
from emergence_contracts.science_request_v1 import validate_eiel_science_request_v1

from .coordinator import ExpeditionCoordinatorError, prepare_decision_v1
from .founder_loop import FounderLoopError, build_founder_child_continuation_v1
from .founder_closed_loop import (
    FounderClosedLoopError,
    run_founder_closed_loop_v1,
)
from .ledger import (
    AdaptiveExpeditionLedgerError,
    project_ledger_to_public_journal_v1,
    read_ledger_store_v1,
    reconstruct_expedition_frontier_v1,
    validate_expedition_ledger_v1,
)
from .lifecycle import default_expedition_lifecycle_policy_value_v1
from .policies.adaptive_illumination_v1 import (
    ADAPTIVE_ILLUMINATION_POLICY_ID_V1,
    AdaptiveIlluminationPolicyV1,
)
from .policies.toy_round_robin_v1 import (
    TOY_ROUND_ROBIN_POLICY_ID_V1,
    ToyRoundRobinPolicyV1,
)
from .policy_protocol import (
    ExpeditionPolicyError,
    ExpeditionPolicyV1,
    canonical_plain_value_v1,
)
from .operator_run import run_expedition_operator_v1


class AdaptiveExpeditionCLIError(ValueError):
    """An operator input cannot be interpreted without weakening a boundary."""


PolicyFactoryV1 = Callable[[Mapping[str, object]], ExpeditionPolicyV1]

# Explicit installation is intentional.  A manifest cannot import arbitrary
# code or name a dynamically discovered policy.  Adding a policy changes this
# operator registry, never the generic coordinator.
_INSTALLED_POLICY_FACTORIES_V1: Final[Mapping[str, PolicyFactoryV1]] = {
    ADAPTIVE_ILLUMINATION_POLICY_ID_V1: AdaptiveIlluminationPolicyV1,
    TOY_ROUND_ROBIN_POLICY_ID_V1: ToyRoundRobinPolicyV1,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eiel expedition",
        description=(
            "Validate and deterministically compose no-claims Adaptive Expedition "
            "V1 artifacts. No command grants execution or Qualification authority."
        ),
        epilog=(
            "Normal use is: eiel expedition run MANIFEST --human. Every child still "
            "enters through the existing Science Request front door."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser(
        "validate-manifest",
        help="validate a manifest and its exact installed-policy binding",
    )
    validate.add_argument("manifest", type=Path)
    validate.add_argument("--human", action="store_true")

    run = commands.add_parser(
        "run",
        help="launch or crash-safely resume one manifest-owned Expedition",
    )
    run.add_argument("manifest", type=Path)
    run.add_argument("--human", action="store_true")

    example = commands.add_parser(
        "example-manifest",
        help="generate the bounded G0-G5 engineering example manifest",
    )
    example.add_argument("--founder-request", required=True, type=Path)
    example.add_argument("--output", type=Path)
    example.add_argument("--human", action="store_true")

    def add_shadow_inputs(command: argparse.ArgumentParser) -> None:
        command.add_argument("manifest", type=Path)
        command.add_argument("observations", type=Path, help="JSON array of ObservationFrameV1")
        command.add_argument("atlas", type=Path)
        command.add_argument("parent_bank", type=Path)
        command.add_argument("history", type=Path, help="bounded canonical JSON history array")
        command.add_argument("--frontier-id", required=True)
        command.add_argument("--frontier-ordinal", required=True, type=int)
        command.add_argument("--logical-capacity", required=True, type=int)
        command.add_argument("--outer-rng-cursor", required=True, type=int)
        command.add_argument("--output", type=Path)
        command.add_argument("--human", action="store_true")

    prepare = commands.add_parser(
        "prepare-shadow",
        help="prepare a deterministic shadow DecisionV1 from canonical inputs",
    )
    add_shadow_inputs(prepare)

    replay = commands.add_parser(
        "replay-shadow",
        help="recompute a shadow decision and require byte-identical DecisionV1 replay",
    )
    replay.add_argument("expected_decision", type=Path)
    add_shadow_inputs(replay)

    founder = commands.add_parser(
        "build-founder-child",
        help="generate a tick-zero founder child; does not execute it",
    )
    founder.add_argument("manifest", type=Path)
    founder.add_argument("decision", type=Path)
    founder.add_argument("founder_request", type=Path)
    founder.add_argument("seeds", type=Path, help="JSON array of unique replicate seeds")
    founder.add_argument("--action-id", required=True)
    founder.add_argument(
        "--physical-batch-worlds",
        type=int,
        help=(
            "deprecated nominal packing override; when omitted, derive the "
            "founder request's declared batch projection"
        ),
    )
    founder.add_argument("--request-output", type=Path)
    founder.add_argument("--continuation-output", type=Path)
    founder.add_argument("--human", action="store_true")

    initialize_state = commands.add_parser(
        "initialize-state",
        help="create immutable empty Atlas and Retained Parent Bank revisions",
    )
    initialize_state.add_argument("manifest", type=Path)
    initialize_state.add_argument(
        "--atlas-max-entries",
        "--atlas-maximum-entries",
        dest="atlas_maximum_entries",
        type=int,
        help=(
            "deprecated compatibility assertion; when supplied it must equal "
            "manifest.maximum_atlas_entries"
        ),
    )
    initialize_state.add_argument("--atlas-output", required=True, type=Path)
    initialize_state.add_argument("--parent-bank-output", required=True, type=Path)
    initialize_state.add_argument("--human", action="store_true")

    closed_loop = commands.add_parser(
        "run-founder-closed-loop",
        help=(
            "run or crash-safely resume bounded tick-zero founder children through "
            "the existing Science Request front door"
        ),
    )
    closed_loop.add_argument("manifest", type=Path)
    closed_loop.add_argument("founder_request", type=Path)
    closed_loop.add_argument("--initial-observation-source", action="append", required=True)
    closed_loop.add_argument("--ledger-root", required=True, type=Path)
    closed_loop.add_argument("--runtime-parent", required=True, type=Path)
    closed_loop.add_argument("--atlas", required=True, type=Path)
    closed_loop.add_argument("--parent-bank", required=True, type=Path)
    closed_loop.add_argument("--maximum-frontiers", required=True, type=int)
    closed_loop.add_argument("--logical-capacity", required=True, type=int)
    closed_loop.add_argument("--physical-batch-worlds", required=True, type=int)
    closed_loop.add_argument("--repo-root", type=Path)
    closed_loop.add_argument("--materialization-parent", type=Path)
    closed_loop.add_argument("--human", action="store_true")

    inspect = commands.add_parser(
        "inspect-ledger",
        help="validate and summarize an append-only ledger file or ledger store",
    )
    inspect.add_argument("ledger", type=Path)
    inspect.add_argument("--human", action="store_true")

    replay_ledger = commands.add_parser(
        "replay-ledger",
        help="validate and reconstruct a ledger projection and public journal",
    )
    replay_ledger.add_argument("ledger", type=Path)
    replay_ledger.add_argument("--output", type=Path)
    replay_ledger.add_argument("--human", action="store_true")
    return parser


def _read_json(path: Path, *, label: str) -> Any:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise AdaptiveExpeditionCLIError(f"{label} must not be a symbolic link: {expanded}")
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise AdaptiveExpeditionCLIError(f"{label} is not a regular file: {resolved}")
    try:
        return json.loads(resolved.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AdaptiveExpeditionCLIError(f"{label} is unreadable JSON: {resolved}") from error


def _read_mapping(path: Path, *, label: str) -> dict[str, Any]:
    value = _read_json(path, label=label)
    if not isinstance(value, dict):
        raise AdaptiveExpeditionCLIError(f"{label} must contain a JSON object")
    return value


def _read_array(path: Path, *, label: str) -> list[Any]:
    value = _read_json(path, label=label)
    if not isinstance(value, list):
        raise AdaptiveExpeditionCLIError(f"{label} must contain a JSON array")
    return value


def _write_canonical_exclusive(path: Path, value: object) -> Path:
    target = path.expanduser().absolute()
    if target.is_symlink():
        raise AdaptiveExpeditionCLIError(f"output must not be a symbolic link: {target}")
    if not target.parent.is_dir():
        raise AdaptiveExpeditionCLIError(
            f"output parent must already be a directory: {target.parent}"
        )
    try:
        with target.open("xb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
    except FileExistsError as error:
        raise AdaptiveExpeditionCLIError(f"output already exists: {target}") from error
    except OSError as error:
        raise AdaptiveExpeditionCLIError(f"cannot write output: {target}") from error
    return target


def _write_canonical_idempotent(path: Path, value: object) -> Path:
    """Create one immutable canonical file, or accept the identical prior write.

    This narrow behavior makes two-file state initialization restartable after
    interruption without granting overwrite authority.
    """

    target = path.expanduser().absolute()
    if target.is_symlink():
        raise AdaptiveExpeditionCLIError(f"output must not be a symbolic link: {target}")
    if not target.parent.is_dir():
        raise AdaptiveExpeditionCLIError(
            f"output parent must already be a directory: {target.parent}"
        )
    expected = canonical_json_bytes(value)
    try:
        with target.open("xb") as handle:
            handle.write(expected)
            handle.flush()
    except FileExistsError:
        try:
            existing = target.read_bytes()
        except OSError as error:
            raise AdaptiveExpeditionCLIError(
                f"existing output is unreadable: {target}"
            ) from error
        if existing != expected:
            raise AdaptiveExpeditionCLIError(
                f"existing output differs from the deterministic state: {target}"
            )
    except OSError as error:
        raise AdaptiveExpeditionCLIError(f"cannot write output: {target}") from error
    return target


def _resolve_bound_policy_v1(manifest: ExpeditionManifestV1) -> ExpeditionPolicyV1:
    policy_id = manifest.policy_binding.policy_id
    factory = _INSTALLED_POLICY_FACTORIES_V1.get(policy_id)
    if factory is None:
        installed = ", ".join(sorted(_INSTALLED_POLICY_FACTORIES_V1))
        raise AdaptiveExpeditionCLIError(
            f"manifest policy is not installed: {policy_id}; installed: {installed}"
        )
    policy = factory(manifest.policy_binding.profile.to_value())
    if canonical_plain_value_v1(policy.binding) != canonical_plain_value_v1(
        manifest.policy_binding.to_dict()
    ):
        raise AdaptiveExpeditionCLIError(
            "installed policy implementation/profile does not match the frozen manifest binding"
        )
    return policy


def _build_example_manifest_v1(
    founder_request_value: Mapping[str, object],
    *,
    founder_request_path: Path,
) -> ExpeditionManifestV1:
    """Build the deterministic, versioned engineering example manifest.

    This is a CLI fixture builder, not coordinator truth.  Its search domain,
    policy, budgets, and lifecycle class remain explicit frozen manifest data.
    """

    founder_request = validate_eiel_science_request_v1(founder_request_value)
    policy = AdaptiveIlluminationPolicyV1()
    founder_sha256 = founder_request.request_sha256()
    descriptor_definitions = policy.profile.get("descriptor_definitions")
    if not isinstance(descriptor_definitions, tuple):
        raise AdaptiveExpeditionCLIError(
            "installed example policy has no frozen descriptor definition tuple"
        )
    identity = canonical_sha256(
        {
            "builder_id": "eiel_adaptive_expedition_example_manifest_builder_v1",
            "founder_request_sha256": founder_sha256.to_dict(),
            "policy_binding": canonical_plain_value_v1(policy.binding),
        }
    )
    manifest = ExpeditionManifestV1.from_dict(
        {
            "schema_version": "eiel_adaptive_science_expedition_manifest_v1",
            "expedition_id": f"EIEL-EXPEDITION-ENGINEERING-{identity.hex[:20]}",
            "question_description": (
                "Bounded no-claims engineering expedition over resource availability "
                "and mutation rate from one declared founder basis"
            ),
            "no_claims": True,
            "hypothesis_ids": [],
            "required_science_instrument": "EIEL_SCIENCE_INSTRUMENT_V1",
            "founder_basis_refs": [
                {
                    "schema_version": "run_artifact_ref_v1",
                    "artifact_id": f"founder-science-request-{founder_sha256.hex[:20]}",
                    "artifact_kind": "eiel_science_request_v1",
                    "artifact_schema_version": "eiel_science_request_v1",
                    "locator": founder_request_path.expanduser().resolve().as_uri(),
                    "content_digest": founder_sha256.to_dict(),
                    "size_bytes": len(canonical_json_bytes(founder_request.to_dict())),
                }
            ],
            "search_space": {
                "schema_version": "eiel_adaptive_expedition_search_space_v1",
                "payload": {
                    "controls": [
                        {
                            "control_id": "environment.resource.multiplier",
                            "allowed_values": [0.25, 0.5, 1.0, 2.0],
                        },
                        {
                            "control_id": "mutation.rate",
                            "allowed_values": [0.0, 0.001, 0.01, 0.05],
                        },
                    ]
                },
            },
            "epoch_ticks": 64,
            "maximum_absolute_tick": 320,
            "maximum_frontiers": 5,
            "maximum_actions_per_frontier": 16,
            "wall_time_budget_seconds": 3600,
            "observation_profile_id": "EIEL_ADAPTIVE_ENGINEERING_OBSERVATION_V1",
            "observation_descriptor_set_sha256": canonical_sha256(
                canonical_plain_value_v1(descriptor_definitions)
            ).to_dict(),
            "policy_binding": canonical_plain_value_v1(policy.binding),
            "outer_search_seed": int(identity.hex[:16], 16),
            "maximum_atlas_entries": 256,
            "maximum_retained_parent_entries": 32,
            "maximum_retained_parent_bytes": 536870912,
            "artifact_lifecycle_policy": {
                "schema_version": "eiel_adaptive_expedition_lifecycle_policy_v1",
                "payload": default_expedition_lifecycle_policy_value_v1(),
            },
            "stop_conditions": [
                "integrity_failure",
                "operator_stop",
                "resource_budget_exhausted",
            ],
        }
    )
    _resolve_bound_policy_v1(manifest)
    return manifest


def _load_manifest_v1(path: Path) -> tuple[ExpeditionManifestV1, ExpeditionPolicyV1]:
    manifest = ExpeditionManifestV1.from_dict(_read_mapping(path, label="manifest"))
    return manifest, _resolve_bound_policy_v1(manifest)


def _empty_scientific_state_v1(
    manifest: ExpeditionManifestV1,
    *,
    atlas_maximum_entries: int | None = None,
) -> tuple[EvolutionaryAtlasV1, RetainedParentBankV1]:
    effective_atlas_maximum_entries = (
        manifest.maximum_atlas_entries
        if atlas_maximum_entries is None
        else atlas_maximum_entries
    )
    if effective_atlas_maximum_entries != manifest.maximum_atlas_entries:
        raise AdaptiveExpeditionCLIError(
            "Atlas maximum must equal the immutable manifest budget"
        )
    atlas = EvolutionaryAtlasV1.from_dict(
        {
            "schema_version": "eiel_adaptive_science_evolutionary_atlas_v1",
            "expedition_id": manifest.expedition_id,
            "manifest_sha256": manifest.manifest_sha256().to_dict(),
            "revision": 0,
            "through_frontier_ordinal": 0,
            "descriptor_set_sha256": (
                manifest.observation_descriptor_set_sha256.to_dict()
            ),
            "maximum_entries": effective_atlas_maximum_entries,
            "entries": [],
            "previous_atlas_sha256": None,
        }
    )
    bank = RetainedParentBankV1.from_dict(
        {
            "schema_version": "eiel_adaptive_science_retained_parent_bank_v1",
            "expedition_id": manifest.expedition_id,
            "manifest_sha256": manifest.manifest_sha256().to_dict(),
            "revision": 0,
            "maximum_entries": manifest.maximum_retained_parent_entries,
            "maximum_total_bytes": manifest.maximum_retained_parent_bytes,
            "current_entry_count": 0,
            "current_total_bytes": 0,
            "entries": [],
            "previous_bank_sha256": None,
        }
    )
    bank.validate_against_manifest(manifest)
    return atlas, bank


def _absolute_path(path: Path) -> str:
    return str(path.expanduser().absolute())


def _prepare_from_args(args: argparse.Namespace):
    manifest, policy = _load_manifest_v1(args.manifest)
    observations = tuple(
        ObservationFrameV1.from_dict(value)
        for value in _read_array(args.observations, label="observations")
    )
    atlas = EvolutionaryAtlasV1.from_dict(_read_mapping(args.atlas, label="Atlas"))
    parent_bank = RetainedParentBankV1.from_dict(
        _read_mapping(args.parent_bank, label="Retained Parent Bank")
    )
    history = _read_array(args.history, label="bounded history")
    if any(not isinstance(row, Mapping) for row in history):
        raise AdaptiveExpeditionCLIError("bounded history entries must be JSON objects")
    return prepare_decision_v1(
        manifest=manifest,
        policy=policy,
        frontier_id=args.frontier_id,
        frontier_ordinal=args.frontier_ordinal,
        observation_frames=observations,
        bounded_history=history,
        atlas=atlas,
        retained_parent_bank=parent_bank,
        logical_capacity=args.logical_capacity,
        outer_rng_cursor=args.outer_rng_cursor,
    )


def _load_ledger_v1(path: Path) -> dict[str, Any]:
    expanded = path.expanduser()
    if expanded.is_dir() and not expanded.is_symlink():
        return read_ledger_store_v1(expanded)
    return validate_expedition_ledger_v1(_read_mapping(expanded, label="ledger"))


def _emit(value: object, *, human: bool) -> None:
    if not human:
        print(canonical_json_bytes(value).decode("utf-8"))
        return
    if not isinstance(value, Mapping):
        print(canonical_json_bytes(value).decode("utf-8"))
        return
    for key in (
        "status",
        "expedition_id",
        "manifest_sha256",
        "policy_id",
        "decision_sha256",
        "continuation_sha256",
        "child_request_sha256",
        "head_event_sha256",
        "current_frontier_ordinal",
        "pending_action_count",
        "written_path",
        "request_written_path",
        "continuation_written_path",
        "atlas_sha256",
        "retained_parent_bank_sha256",
        "atlas_written_path",
        "parent_bank_written_path",
        "ledger_root",
        "runtime_parent",
        "runtime_root",
        "frontier_count",
        "child_count",
        "terminal_status",
        "atlas_path",
        "retained_parent_bank_path",
        "journal_path",
        "closeout_path",
    ):
        if key in value and value[key] is not None:
            print(f"{key}: {value[key]}")
    print("authority: none; no execution or Qualification authority granted")


def _decision_result_v1(prepared: Any, *, written_path: Path | None) -> dict[str, object]:
    decision = prepared.decision
    return {
        "status": "SHADOW_DECISION_PREPARED",
        "authority_effect": "none",
        "execution_started": False,
        "qualification_granted": False,
        "decision_sha256": str(decision.decision_sha256()),
        "policy_input_sha256": str(prepared.policy_input_sha256),
        "policy_proposal_sha256": str(prepared.proposal_sha256),
        "next_outer_rng_cursor": prepared.next_outer_rng_cursor,
        "written_path": None if written_path is None else str(written_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        result: object
        if args.command == "validate-manifest":
            manifest, policy = _load_manifest_v1(args.manifest)
            result = {
                "status": "VALID",
                "authority_effect": "none",
                "expedition_id": manifest.expedition_id,
                "manifest_sha256": str(manifest.manifest_sha256()),
                "policy_id": policy.policy_id,
                "policy_binding_verified": True,
                "no_claims": manifest.no_claims,
            }
        elif args.command == "run":
            manifest, policy = _load_manifest_v1(args.manifest)
            result = run_expedition_operator_v1(
                manifest=manifest,
                policy=policy,
                repo_root=Path(__file__).resolve().parents[2],
            ).receipt()
        elif args.command == "example-manifest":
            founder_request = _read_mapping(
                args.founder_request, label="founder Science Request"
            )
            manifest = _build_example_manifest_v1(
                founder_request,
                founder_request_path=args.founder_request,
            )
            manifest_value = manifest.to_dict()
            written = (
                None
                if args.output is None
                else _write_canonical_exclusive(args.output, manifest_value)
            )
            result = (
                manifest_value
                if written is None and not args.human
                else {
                    "status": "EXAMPLE_MANIFEST_GENERATED",
                    "authority_effect": "none",
                    "execution_started": False,
                    "qualification_granted": False,
                    "expedition_id": manifest.expedition_id,
                    "manifest_sha256": str(manifest.manifest_sha256()),
                    "policy_id": manifest.policy_binding.policy_id,
                    "written_path": None if written is None else str(written),
                }
            )
        elif args.command in {"prepare-shadow", "replay-shadow"}:
            prepared = _prepare_from_args(args)
            decision_value = prepared.decision.to_dict()
            if args.command == "replay-shadow":
                expected = DecisionV1.from_dict(
                    _read_mapping(args.expected_decision, label="expected DecisionV1")
                )
                if canonical_json_bytes(expected.to_dict()) != canonical_json_bytes(
                    decision_value
                ):
                    raise AdaptiveExpeditionCLIError(
                        "replayed DecisionV1 differs from the expected sealed decision"
                    )
            written = (
                None
                if args.output is None
                else _write_canonical_exclusive(args.output, decision_value)
            )
            result = (
                decision_value
                if written is None and not args.human
                else {
                    **_decision_result_v1(prepared, written_path=written),
                    "status": (
                        "SHADOW_DECISION_REPLAY_IDENTICAL"
                        if args.command == "replay-shadow"
                        else "SHADOW_DECISION_PREPARED"
                    ),
                }
            )
        elif args.command == "build-founder-child":
            manifest, _policy = _load_manifest_v1(args.manifest)
            decision = DecisionV1.from_dict(
                _read_mapping(args.decision, label="DecisionV1")
            )
            seeds = _read_array(args.seeds, label="replicate seeds")
            continuation = build_founder_child_continuation_v1(
                manifest=manifest,
                decision=decision,
                action_id=args.action_id,
                founder_request=_read_mapping(
                    args.founder_request, label="founder Science Request"
                ),
                replicate_seeds=seeds,
                physical_batch_worlds=args.physical_batch_worlds,
            )
            continuation_value = continuation.to_dict()
            request_value = continuation.child_request.to_value()
            request_written = (
                None
                if args.request_output is None
                else _write_canonical_exclusive(args.request_output, request_value)
            )
            continuation_written = (
                None
                if args.continuation_output is None
                else _write_canonical_exclusive(
                    args.continuation_output, continuation_value
                )
            )
            result = (
                continuation_value
                if request_written is None
                and continuation_written is None
                and not args.human
                else {
                    "status": "FOUNDER_CHILD_GENERATED_NOT_EXECUTED",
                    "authority_effect": "none",
                    "execution_started": False,
                    "qualification_granted": False,
                    "continuation_sha256": str(continuation.continuation_sha256()),
                    "child_request_sha256": str(continuation.child_request_sha256),
                    "request_written_path": (
                        None if request_written is None else str(request_written)
                    ),
                    "continuation_written_path": (
                        None
                        if continuation_written is None
                        else str(continuation_written)
                    ),
                    "next_command": (
                        None
                        if request_written is None
                        else f"./tools/bin/eiel science run {request_written} --human"
                    ),
                }
            )
        elif args.command == "initialize-state":
            manifest, _policy = _load_manifest_v1(args.manifest)
            atlas, parent_bank = _empty_scientific_state_v1(
                manifest,
                atlas_maximum_entries=args.atlas_maximum_entries,
            )
            atlas_written = _write_canonical_idempotent(
                args.atlas_output, atlas.to_dict()
            )
            bank_written = _write_canonical_idempotent(
                args.parent_bank_output, parent_bank.to_dict()
            )
            result = {
                "status": "EMPTY_EXPEDITION_STATE_INITIALIZED",
                "authority_effect": "none",
                "execution_started": False,
                "qualification_status": "OUTSIDE_PROGRAMME",
                "no_claims": True,
                "expedition_id": manifest.expedition_id,
                "manifest_sha256": str(manifest.manifest_sha256()),
                "atlas_sha256": str(atlas.atlas_sha256()),
                "retained_parent_bank_sha256": str(parent_bank.bank_sha256()),
                "atlas_written_path": str(atlas_written),
                "parent_bank_written_path": str(bank_written),
            }
        elif args.command == "run-founder-closed-loop":
            manifest, policy = _load_manifest_v1(args.manifest)
            founder_request = validate_eiel_science_request_v1(
                _read_mapping(args.founder_request, label="founder Science Request")
            )
            atlas = EvolutionaryAtlasV1.from_dict(
                _read_mapping(args.atlas, label="initial Evolutionary Atlas")
            )
            parent_bank = RetainedParentBankV1.from_dict(
                _read_mapping(args.parent_bank, label="initial Retained Parent Bank")
            )
            initial_sources = tuple(
                _absolute_path(Path(source))
                for source in args.initial_observation_source
            )
            loop_result = run_founder_closed_loop_v1(
                manifest=manifest,
                policy=policy,
                founder_request=founder_request,
                initial_observation_sources=initial_sources,
                ledger_root=_absolute_path(args.ledger_root),
                runtime_parent=_absolute_path(args.runtime_parent),
                atlas=atlas,
                retained_parent_bank=parent_bank,
                maximum_frontiers=args.maximum_frontiers,
                logical_capacity=args.logical_capacity,
                physical_batch_worlds=args.physical_batch_worlds,
                repo_root=(
                    None if args.repo_root is None else _absolute_path(args.repo_root)
                ),
                materialization_parent=(
                    None
                    if args.materialization_parent is None
                    else _absolute_path(args.materialization_parent)
                ),
                shadow_mode=False,
            )
            if loop_result.shadow_mode or not loop_result.terminal:
                raise AdaptiveExpeditionCLIError(
                    "live founder closed loop returned without terminal settlement"
                )
            result = {
                "status": "FOUNDER_CLOSED_LOOP_COMPLETED_OR_RECOVERED",
                "authority_effect": "none-created",
                "execution_route": "existing_eiel_science_request_front_door",
                "qualification_status": "OUTSIDE_PROGRAMME",
                "no_claims": True,
                "historical_continuation_authority": False,
                "expedition_id": manifest.expedition_id,
                "manifest_sha256": str(manifest.manifest_sha256()),
                "policy_id": policy.policy_id,
                "ledger_root": _absolute_path(args.ledger_root),
                "runtime_parent": _absolute_path(args.runtime_parent),
                "head_event_sha256": loop_result.ledger["head_event_sha256"],
                "frontier_count": len(loop_result.decisions),
                "child_count": len(loop_result.children),
                "atlas_sha256": str(loop_result.atlas.atlas_sha256()),
                "retained_parent_bank_sha256": str(
                    loop_result.retained_parent_bank.bank_sha256()
                ),
            }
        elif args.command == "inspect-ledger":
            ledger = _load_ledger_v1(args.ledger)
            projection = reconstruct_expedition_frontier_v1(ledger)
            result = {
                "status": "VALID_RECOVERABLE",
                "authority_effect": "none",
                "expedition_id": ledger["expedition_id"],
                "manifest_sha256": ledger["manifest_sha256"],
                "policy_sha256": ledger["policy_sha256"],
                "event_count": len(ledger["events"]),
                "head_event_sha256": ledger["head_event_sha256"],
                "current_frontier_ordinal": projection["current_frontier_ordinal"],
                "pending_action_count": len(projection["pending_action_ids"]),
                "terminal": projection["terminal"],
            }
        else:
            ledger = _load_ledger_v1(args.ledger)
            replay_value = {
                "authority_effect": "none",
                "frontier_projection": reconstruct_expedition_frontier_v1(ledger),
                "public_journal": project_ledger_to_public_journal_v1(ledger),
            }
            written = (
                None
                if args.output is None
                else _write_canonical_exclusive(args.output, replay_value)
            )
            result = (
                replay_value
                if written is None and not args.human
                else {
                    "status": "LEDGER_REPLAYED",
                    "authority_effect": "none",
                    "expedition_id": ledger["expedition_id"],
                    "head_event_sha256": ledger["head_event_sha256"],
                    "current_frontier_ordinal": replay_value[
                        "frontier_projection"
                    ]["current_frontier_ordinal"],
                    "pending_action_count": len(
                        replay_value["frontier_projection"]["pending_action_ids"]
                    ),
                    "written_path": None if written is None else str(written),
                }
            )
    except (
        AdaptiveExpeditionCLIError,
        AdaptiveExpeditionLedgerError,
        ContractError,
        ExpeditionCoordinatorError,
        ExpeditionPolicyError,
        FounderClosedLoopError,
        FounderLoopError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"eiel expedition: {error}", file=sys.stderr)
        return 2
    _emit(result, human=args.human)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
