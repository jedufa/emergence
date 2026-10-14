from engine.evolution.genome_upgrade_v2_build_admission_v1 import (
    MONITOR_MIN_AVAILABLE_BYTES,
    PREFLIGHT_MIN_AVAILABLE_BYTES,
    build_admission_policy_v1,
    evaluate_build_snapshot_v1,
)


def _process(pid: int, ppid: int, comm: str, command: str = "") -> dict:
    return {"pid": pid, "ppid": ppid, "comm": comm, "command": command}


def test_preflight_requires_memory_and_no_compiler_tree() -> None:
    ready = evaluate_build_snapshot_v1(
        memory={"available_bytes": PREFLIGHT_MIN_AVAILABLE_BYTES},
        processes=[],
        phase="preflight",
    )
    assert ready["admitted"] is True

    blocked = evaluate_build_snapshot_v1(
        memory={"available_bytes": PREFLIGHT_MIN_AVAILABLE_BYTES - 1},
        processes=[_process(10, 1, "ninja", "ninja -v")],
        phase="preflight",
    )
    assert blocked["admitted"] is False
    assert blocked["reasons"] == [
        "available_memory_below_admission_floor",
        "compiler_tree_already_active",
    ]


def test_monitor_rejects_second_tree_and_accepts_one_guarded_tree() -> None:
    one = [
        _process(100, 1, "python", "torch.utils.cpp_extension"),
        _process(101, 100, "ninja", "ninja -v"),
        _process(102, 101, "nvcc", "nvcc"),
        _process(103, 102, "cicc", "cicc"),
    ]
    admitted = evaluate_build_snapshot_v1(
        memory={"available_bytes": MONITOR_MIN_AVAILABLE_BYTES},
        processes=one,
        phase="monitor",
        guarded_root_pid=100,
    )
    assert admitted["admitted"] is True
    assert admitted["compiler_tree_roots"] == [100]

    two = one + [
        _process(200, 1, "ninja", "ninja -v"),
        _process(201, 200, "cc1plus", "cc1plus"),
    ]
    blocked = evaluate_build_snapshot_v1(
        memory={"available_bytes": MONITOR_MIN_AVAILABLE_BYTES},
        processes=two,
        phase="monitor",
        guarded_root_pid=100,
    )
    assert blocked["admitted"] is False
    assert "multiple_compiler_trees_active" in blocked["reasons"]
    assert "compiler_tree_outside_guarded_process_group" in blocked["reasons"]


def test_monitor_groups_direct_nvcc_and_its_compiler_children() -> None:
    direct = [
        _process(300, 1, "python", "genome_upgrade_v2_build_guard_v1.py"),
        _process(301, 300, "nvcc", "nvcc -DTORCH_EXTENSION_NAME=x"),
        _process(302, 301, "cc1plus", "cc1plus -DTORCH_EXTENSION_NAME=x"),
        _process(303, 301, "ptxas", "ptxas"),
    ]

    admitted = evaluate_build_snapshot_v1(
        memory={"available_bytes": MONITOR_MIN_AVAILABLE_BYTES},
        processes=direct,
        phase="monitor",
        guarded_root_pid=300,
    )

    assert admitted["admitted"] is True
    assert admitted["compiler_process_count"] == 3
    assert admitted["compiler_tree_roots"] == [300]


def test_build_policy_forbids_overlap_and_cpu_fixture_builds() -> None:
    policy = build_admission_policy_v1()
    assert policy["overlapping_builds_allowed"] == 0
    assert policy["maximum_monitored_build_trees"] == 1
    assert policy["cpu_only_perf_build_fixture_allowed"] is False
