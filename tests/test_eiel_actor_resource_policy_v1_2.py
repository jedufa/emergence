from __future__ import annotations

from pathlib import Path

from tests.eiel_actor_v11_helpers import run_eiel, write_safe_erm_report


def test_wait_guidance_obeys_agents_policy_without_background_polling(tmp_path: Path) -> None:
    report = write_safe_erm_report(tmp_path / "blocked.json", blocked=True)
    result = run_eiel(
        [
            "resourceManager",
            "assess",
            "--device",
            "cuda",
            "--report",
            str(report),
            "--generated-at",
            "2026-08-03T12:00:00Z",
        ],
        state_root=tmp_path / "state",
        expected=1,
    )
    assessment = result["assessment"]
    assert result["status"] == assessment["decision"] == "WAIT"
    policy = assessment["operating_policy"]
    assert policy["retry_after_seconds"] == 30
    assert policy["background_polling"] is False
    assert policy["continue_non_contending_work"] is True
    assert policy["erm_wait_is_binding_for_codex_launch"] is True

    actions = assessment["next_actions"]
    assert actions[0]["action"] == "do_not_launch_contested_work"
    assert actions[1]["action"] == "continue_non_contending_work"
    assert "source inspection" in actions[1]["description"]
    assert actions[2]["action"] == "rerun_fresh_inspection"
    assert actions[2]["retry_after_seconds"] == 30
    assert actions[2]["command"] == "python -m tools.eiel resourceManager inspect"


def test_safe_cpu_work_can_continue_while_gpu_is_contested(tmp_path: Path) -> None:
    report = write_safe_erm_report(tmp_path / "blocked.json", blocked=True)
    result = run_eiel(
        [
            "resourceManager",
            "recommend",
            "--device",
            "either",
            "--report",
            str(report),
            "--generated-at",
            "2026-08-03T12:00:00Z",
        ],
        state_root=tmp_path / "state",
    )
    assessment = result["assessment"]
    assert assessment["decision"] == "ADMIT_WITH_LIMITS"
    assert assessment["recommended_route"] == "cpu"
    assert assessment["operating_policy"]["background_polling"] is False


def test_unknown_cuda_telemetry_requires_manual_review_not_guessing(tmp_path: Path) -> None:
    report = write_safe_erm_report(tmp_path / "unknown.json", gpu_available=False)
    result = run_eiel(
        [
            "resourceManager",
            "assess",
            "--device",
            "cuda",
            "--report",
            str(report),
            "--generated-at",
            "2026-08-03T12:00:00Z",
        ],
        state_root=tmp_path / "state",
        expected=1,
    )
    assert result["status"] == "MANUAL_REVIEW"
    assert "gpu_state_unknown" in result["reason_codes"]
    assert result["assessment"]["recommended_route"] == "none"
