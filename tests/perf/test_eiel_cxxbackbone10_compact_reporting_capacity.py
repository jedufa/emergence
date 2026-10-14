from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OWNER = ROOT / "engine" / "perf" / "native" / "cxxbackbone10_resident_state_owner.cpp"


def _braced_body(source: str, opening_brace: int) -> str:
    depth = 1
    for index in range(opening_brace + 1, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening_brace + 1 : index]
    raise AssertionError("unbalanced C++ braces")


def _definition_body(source: str, name: str) -> str:
    match = re.search(rf"\b{re.escape(name)}\s*\([^;{{}}]*\)\s*\{{", source)
    assert match is not None, f"missing C++ definition {name}"
    return _braced_body(source, match.end() - 1)


def test_b1_starts_with_the_224_row_reporting_baseline() -> None:
    source = OWNER.read_text(encoding="ascii")
    constructor = _definition_body(source, "ResidentRuntimeStateV10")

    assert "kCompactReportingBaselineRowCapacityV1 = 224" in source
    assert "kCompactReportingBaselineRowCapacityV1," in constructor
    assert "compact_reporting_ledger::kMaximumRowCapacityV1," not in constructor


def test_reporting_capacity_helper_checks_bounds_before_multiplication() -> None:
    source = OWNER.read_text(encoding="ascii")
    helper = _definition_body(
        source, "ensure_compact_reporting_ledger_row_capacity_cpu"
    )

    positive_steps = "steps_per_generation > 0"
    non_negative_births = "max_births >= 0"
    guarded_division = (
        "(std::numeric_limits<int64_t>::max() - active_agent_count)"
    )
    capacity_product = "active_agent_count + steps_per_generation * max_births"
    maximum_check = "compact_reporting_ledger::kMaximumRowCapacityV1"

    for contract in (
        positive_steps,
        non_negative_births,
        guarded_division,
        capacity_product,
        maximum_check,
    ):
        assert contract in helper
    assert helper.index(positive_steps) < helper.index(guarded_division)
    assert helper.index(non_negative_births) < helper.index(guarded_division)
    assert helper.index(guarded_division) < helper.index(capacity_product)
    assert helper.index(capacity_product) < helper.index(maximum_check)


def test_reporting_capacity_growth_preserves_lineage_streak_state() -> None:
    source = OWNER.read_text(encoding="ascii")
    helper = _definition_body(
        source, "ensure_compact_reporting_ledger_row_capacity_cpu"
    )

    state_read = "compact_reporting_ledger_->lineage_metrics_state()"
    replacement = "auto grown_ledger = std::make_unique<"
    state_argument = "lineage_metrics_state\n        );"
    replacement_commit = "compact_reporting_ledger_ = std::move(grown_ledger);"

    for contract in (state_read, replacement, state_argument, replacement_commit):
        assert contract in helper
    assert helper.index(state_read) < helper.index(replacement)
    assert helper.index(replacement) < helper.index(state_argument)
    assert helper.index(state_argument) < helper.index(replacement_commit)


def test_flat_and_patch_size_reporting_before_generation_start() -> None:
    source = OWNER.read_text(encoding="ascii")
    helper_call = (
        "ensure_compact_reporting_ledger_row_capacity_cpu(\n"
        "            steps_per_generation,\n"
        "            max_births\n"
        "        );"
    )
    behavior_start = "behavior_start_generation_cpu(generation_index);"

    for function_name in (
        "execute_generation_flat_with_population_birth_commit_controlled_cpu",
        "execute_generation_patch_with_population_birth_commit_controlled_cpu",
    ):
        body = _definition_body(source, function_name)
        assert body.count(helper_call) == 1
        assert body.count(behavior_start) == 1
        assert body.index(helper_call) < body.index(behavior_start)
