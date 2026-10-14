from __future__ import annotations

import json
from types import SimpleNamespace

from emergence_contracts import canonical_json_bytes
from emergence_contracts.adaptive_science_expedition_v1 import (
    DecisionV1,
    EvolutionaryAtlasV1,
    ExpeditionManifestV1,
    GeneratedChildContinuationV1,
    RetainedParentBankV1,
)
from emergence_contracts.science_request_v1 import validate_eiel_science_request_v1
from eiel_composition.adaptive_expedition import cli as expedition_cli_module
from eiel_composition.adaptive_expedition.cli import main as expedition_main
from eiel_composition.adaptive_expedition.founder_closed_loop import (
    FounderClosedLoopError,
)
from eiel_composition.adaptive_expedition.ledger import new_expedition_ledger_v1
from eiel_composition.adaptive_expedition.policies.adaptive_illumination_v1 import (
    AdaptiveIlluminationPolicyV1,
)
from tests.test_eiel_adaptive_expedition_coordinator_v1 import (
    _atlas,
    _bank,
    _frame,
    _manifest,
)
from tests.test_eiel_adaptive_expedition_founder_loop_v1 import (
    _action_id,
    _founder_request,
    _prepared,
)
from tools import eiel


def _write(path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value))


def _shadow_paths(tmp_path):
    policy = AdaptiveIlluminationPolicyV1()
    manifest = _manifest(policy)
    paths = {
        "manifest": tmp_path / "manifest.json",
        "observations": tmp_path / "observations.json",
        "atlas": tmp_path / "atlas.json",
        "parent_bank": tmp_path / "parent-bank.json",
        "history": tmp_path / "history.json",
    }
    _write(paths["manifest"], manifest.to_dict())
    _write(paths["observations"], [_frame().to_dict()])
    _write(paths["atlas"], _atlas(manifest).to_dict())
    _write(paths["parent_bank"], _bank(manifest).to_dict())
    _write(paths["history"], [])
    return paths


def _shadow_arguments(paths) -> list[str]:
    return [
        str(paths["manifest"]),
        str(paths["observations"]),
        str(paths["atlas"]),
        str(paths["parent_bank"]),
        str(paths["history"]),
        "--frontier-id",
        "frontier-cli-0000",
        "--frontier-ordinal",
        "0",
        "--logical-capacity",
        "4",
        "--outer-rng-cursor",
        "0",
    ]


def test_top_level_help_and_dispatch_expose_a_core_tool(monkeypatch) -> None:
    assert "Adaptive Expedition:" in eiel.render_help()
    assert "python -m tools.eiel expedition --help" in eiel.render_help()
    forwarded: list[str] = []

    def fake_core_tool_main(name: str):
        assert name == "expedition"

        def invoke(arguments: list[str]) -> int:
            forwarded.extend(arguments)
            return 0

        return invoke

    monkeypatch.setattr(eiel, "_core_tool_main", fake_core_tool_main)
    assert eiel.main(["expedition", "validate-manifest", "manifest.json"]) == 0
    assert forwarded == ["validate-manifest", "manifest.json"]


def test_manifest_validation_requires_the_exact_installed_policy_binding(
    tmp_path, capsys
) -> None:
    policy = AdaptiveIlluminationPolicyV1()
    manifest = _manifest(policy).to_dict()
    manifest_path = tmp_path / "manifest.json"
    _write(manifest_path, manifest)

    assert expedition_main(["validate-manifest", str(manifest_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "VALID"
    assert result["policy_binding_verified"] is True
    assert result["authority_effect"] == "none"

    manifest["policy_binding"]["implementation_sha256"]["hex"] = "0" * 64
    changed = tmp_path / "changed.json"
    _write(changed, manifest)
    assert expedition_main(["validate-manifest", str(changed)]) == 2
    assert "does not match the frozen manifest binding" in capsys.readouterr().err


def test_example_manifest_is_deterministic_valid_and_fully_bound(tmp_path, capsys) -> None:
    founder_path = tmp_path / "founder.json"
    manifest_path = tmp_path / "example-manifest.json"
    _write(founder_path, _founder_request())

    assert expedition_main(
        [
            "example-manifest",
            "--founder-request",
            str(founder_path),
            "--output",
            str(manifest_path),
        ]
    ) == 0
    generated = json.loads(capsys.readouterr().out)
    assert generated["status"] == "EXAMPLE_MANIFEST_GENERATED"

    manifest_value = json.loads(manifest_path.read_bytes())
    manifest = ExpeditionManifestV1.from_dict(manifest_value)
    assert manifest.no_claims is True
    assert manifest.epoch_ticks == 64
    assert manifest.maximum_frontiers == 5
    assert manifest.maximum_absolute_tick == 320
    controls = manifest.search_space.to_value()["controls"]
    assert [row["control_id"] for row in controls] == [
        "environment.resource.multiplier",
        "mutation.rate",
    ]
    assert manifest.artifact_lifecycle_policy.to_value()["child_contract_class"] == (
        "SCIENCE_COMPACT"
    )

    assert expedition_main(
        ["example-manifest", "--founder-request", str(founder_path)]
    ) == 0
    replayed = json.loads(capsys.readouterr().out)
    assert canonical_json_bytes(replayed) == manifest_path.read_bytes()
    assert expedition_main(["validate-manifest", str(manifest_path)]) == 0
    assert json.loads(capsys.readouterr().out)["policy_binding_verified"] is True


def test_shadow_decision_prepare_and_replay_are_byte_identical(tmp_path, capsys) -> None:
    paths = _shadow_paths(tmp_path)
    decision_path = tmp_path / "decision.json"
    replay_path = tmp_path / "replayed-decision.json"

    assert expedition_main(
        [
            "prepare-shadow",
            *_shadow_arguments(paths),
            "--output",
            str(decision_path),
        ]
    ) == 0
    prepared = json.loads(capsys.readouterr().out)
    assert prepared["status"] == "SHADOW_DECISION_PREPARED"
    DecisionV1.from_dict(json.loads(decision_path.read_bytes()))

    assert expedition_main(
        [
            "replay-shadow",
            str(decision_path),
            *_shadow_arguments(paths),
            "--output",
            str(replay_path),
        ]
    ) == 0
    replayed = json.loads(capsys.readouterr().out)
    assert replayed["status"] == "SHADOW_DECISION_REPLAY_IDENTICAL"
    assert decision_path.read_bytes() == replay_path.read_bytes()


def test_founder_child_command_only_generates_a_tick_zero_science_request(
    tmp_path, capsys
) -> None:
    manifest, _request, prepared = _prepared()
    manifest_path = tmp_path / "manifest.json"
    decision_path = tmp_path / "decision.json"
    founder_path = tmp_path / "founder.json"
    seeds_path = tmp_path / "seeds.json"
    child_path = tmp_path / "child-request.json"
    continuation_path = tmp_path / "continuation.json"
    _write(manifest_path, manifest.to_dict())
    _write(decision_path, prepared.decision.to_dict())
    _write(founder_path, _founder_request())
    _write(seeds_path, [101, 202, 303])

    action_id = _action_id(prepared, "EXPLORE_GLOBAL")
    assert expedition_main(
        [
            "build-founder-child",
            str(manifest_path),
            str(decision_path),
            str(founder_path),
            str(seeds_path),
            "--action-id",
            action_id,
            "--physical-batch-worlds",
            "3",
            "--request-output",
            str(child_path),
            "--continuation-output",
            str(continuation_path),
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "FOUNDER_CHILD_GENERATED_NOT_EXECUTED"
    assert result["execution_started"] is False
    child = validate_eiel_science_request_v1(json.loads(child_path.read_bytes()))
    continuation = GeneratedChildContinuationV1.from_dict(
        json.loads(continuation_path.read_bytes())
    )
    assert child.expect_worlds == 3
    assert continuation.effective_next_absolute_tick == 0
    assert continuation.parent_state_ref is None


def test_initialize_state_is_manifest_bound_deterministic_and_restartable(
    tmp_path, capsys
) -> None:
    manifest, _request, _prepared_decision = _prepared()
    manifest_path = tmp_path / "manifest.json"
    atlas_path = tmp_path / "atlas.json"
    parent_bank_path = tmp_path / "parent-bank.json"
    _write(manifest_path, manifest.to_dict())
    arguments = [
        "initialize-state",
        str(manifest_path),
        "--atlas-max-entries",
        str(manifest.maximum_atlas_entries),
        "--atlas-output",
        str(atlas_path),
        "--parent-bank-output",
        str(parent_bank_path),
    ]

    assert expedition_main(arguments) == 0
    first_receipt = json.loads(capsys.readouterr().out)
    first_atlas_bytes = atlas_path.read_bytes()
    first_bank_bytes = parent_bank_path.read_bytes()

    # An interruption after either immutable file was written is recoverable:
    # rerunning accepts only byte-identical prior state and fills any missing peer.
    assert expedition_main(arguments) == 0
    second_receipt = json.loads(capsys.readouterr().out)
    assert first_receipt == second_receipt
    assert atlas_path.read_bytes() == first_atlas_bytes
    assert parent_bank_path.read_bytes() == first_bank_bytes

    atlas = EvolutionaryAtlasV1.from_dict(json.loads(first_atlas_bytes))
    parent_bank = RetainedParentBankV1.from_dict(json.loads(first_bank_bytes))
    assert atlas.expedition_id == manifest.expedition_id
    assert atlas.manifest_sha256 == manifest.manifest_sha256()
    assert atlas.descriptor_set_sha256 == manifest.observation_descriptor_set_sha256
    assert atlas.maximum_entries == manifest.maximum_atlas_entries
    assert atlas.entries == ()
    assert parent_bank.expedition_id == manifest.expedition_id
    assert parent_bank.manifest_sha256 == manifest.manifest_sha256()
    assert parent_bank.entries == ()
    parent_bank.validate_against_manifest(manifest)
    assert first_atlas_bytes == canonical_json_bytes(atlas.to_dict())
    assert first_bank_bytes == canonical_json_bytes(parent_bank.to_dict())
    assert first_receipt["no_claims"] is True
    assert first_receipt["qualification_status"] == "OUTSIDE_PROGRAMME"


def test_founder_closed_loop_cli_reuses_exact_inputs_after_stubbed_crash(
    tmp_path, capsys, monkeypatch
) -> None:
    manifest, founder_request, _prepared_decision = _prepared()
    manifest_path = tmp_path / "manifest.json"
    founder_path = tmp_path / "founder.json"
    atlas_path = tmp_path / "atlas.json"
    parent_bank_path = tmp_path / "parent-bank.json"
    source_a = tmp_path / "retained-science-a"
    source_b = tmp_path / "retained-science-b"
    ledger_root = tmp_path / "ledger"
    runtime_parent = tmp_path / "runtime"
    materialization_parent = tmp_path / "materializations"
    source_a.mkdir()
    source_b.mkdir()
    materialization_parent.mkdir()
    _write(manifest_path, manifest.to_dict())
    _write(founder_path, founder_request.to_dict())
    _write(atlas_path, _atlas(manifest).to_dict())
    _write(parent_bank_path, _bank(manifest).to_dict())

    snapshots: list[dict[str, object]] = []

    def crash_safe_stub(**options):
        snapshots.append(
            {
                "manifest_sha256": options["manifest"].manifest_sha256().hex,
                "policy_binding": options["policy"].binding,
                "founder_request_sha256": options[
                    "founder_request"
                ].request_sha256().hex,
                "initial_observation_sources": options[
                    "initial_observation_sources"
                ],
                "ledger_root": options["ledger_root"],
                "runtime_parent": options["runtime_parent"],
                "maximum_frontiers": options["maximum_frontiers"],
                "logical_capacity": options["logical_capacity"],
                "physical_batch_worlds": options["physical_batch_worlds"],
                "repo_root": options["repo_root"],
                "materialization_parent": options["materialization_parent"],
                "shadow_mode": options["shadow_mode"],
            }
        )
        if len(snapshots) == 1:
            ledger_root.mkdir()
            (ledger_root / "durable-crash-boundary").write_text("sealed\n")
            raise FounderClosedLoopError("injected crash after durable ledger boundary")
        return SimpleNamespace(
            ledger={"head_event_sha256": "a" * 64},
            decisions=(object(), object()),
            children=(object(),),
            atlas=options["atlas"],
            retained_parent_bank=options["retained_parent_bank"],
            shadow_mode=False,
            terminal=True,
        )

    monkeypatch.setattr(
        expedition_cli_module,
        "run_founder_closed_loop_v1",
        crash_safe_stub,
    )
    arguments = [
        "run-founder-closed-loop",
        str(manifest_path),
        str(founder_path),
        "--initial-observation-source",
        str(source_a),
        "--initial-observation-source",
        str(source_b),
        "--ledger-root",
        str(ledger_root),
        "--runtime-parent",
        str(runtime_parent),
        "--atlas",
        str(atlas_path),
        "--parent-bank",
        str(parent_bank_path),
        "--maximum-frontiers",
        "2",
        "--logical-capacity",
        "7",
        "--physical-batch-worlds",
        "3",
        "--repo-root",
        str(tmp_path),
        "--materialization-parent",
        str(materialization_parent),
    ]

    assert expedition_main(arguments) == 2
    assert "injected crash" in capsys.readouterr().err
    assert expedition_main(arguments) == 0
    receipt = json.loads(capsys.readouterr().out)

    assert snapshots[0] == snapshots[1]
    assert snapshots[1]["shadow_mode"] is False
    assert snapshots[1]["initial_observation_sources"] == (
        str(source_a.absolute()),
        str(source_b.absolute()),
    )
    assert receipt["status"] == "FOUNDER_CLOSED_LOOP_COMPLETED_OR_RECOVERED"
    assert receipt["execution_route"] == "existing_eiel_science_request_front_door"
    assert receipt["historical_continuation_authority"] is False
    assert receipt["no_claims"] is True
    assert receipt["qualification_status"] == "OUTSIDE_PROGRAMME"
    assert receipt["frontier_count"] == 2
    assert receipt["child_count"] == 1


def test_ledger_inspection_and_replay_validate_recoverable_state(tmp_path, capsys) -> None:
    ledger = new_expedition_ledger_v1(
        expedition_id="expedition-cli-ledger",
        manifest_sha256="1" * 64,
        policy_sha256="2" * 64,
    )
    ledger_path = tmp_path / "ledger.json"
    replay_path = tmp_path / "ledger-replay.json"
    _write(ledger_path, ledger)

    assert expedition_main(["inspect-ledger", str(ledger_path)]) == 0
    inspection = json.loads(capsys.readouterr().out)
    assert inspection["status"] == "VALID_RECOVERABLE"
    assert inspection["event_count"] == 0
    assert inspection["authority_effect"] == "none"

    assert expedition_main(
        ["replay-ledger", str(ledger_path), "--output", str(replay_path)]
    ) == 0
    receipt = json.loads(capsys.readouterr().out)
    replay = json.loads(replay_path.read_bytes())
    assert receipt["status"] == "LEDGER_REPLAYED"
    assert replay["frontier_projection"]["pending_action_ids"] == []
    assert replay["public_journal"] == []
