from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BINDING = (
    ROOT
    / "engine"
    / "perf"
    / "cuda_architecture_b_resident_run_control_v2_binding.hpp"
)


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
    match = re.search(
        rf"\b{re.escape(name)}\s*\([^;{{}}]*\)\s*\{{",
        source,
    )
    assert match is not None, f"missing C++ definition {name}"
    return _braced_body(source, match.end() - 1)


def _initializer_literals(source: str, name: str) -> tuple[str, ...]:
    match = re.search(rf"\b{re.escape(name)}\b[^;{{}}]*(?:=\s*)?\{{", source)
    assert match is not None, f"missing initializer {name}"
    body = _braced_body(source, match.end() - 1)
    return tuple(re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', body))


def _dict_assignment_keys(body: str) -> tuple[str, ...]:
    return tuple(re.findall(r'out\["([a-z0-9_]+)"\]\s*=', body))


def test_binding_header_is_self_contained_and_cuda_free() -> None:
    source = BINDING.read_text(encoding="utf-8")
    lowered = source.lower()

    assert source.startswith("#pragma once\n")
    assert '#include "cuda_architecture_b_resident_run_control_v2.hpp"' in source
    assert "#include <pybind11/pybind11.h>" in source
    for prohibited in ("<torch/", "torch::", "cuda_runtime", "cuda.h"):
        assert prohibited not in lowered
    assert (
        "namespace eiel::cuda_architecture_b::resident_run_control_v2_binding"
        in source
    )


def test_cancel_request_parser_requires_the_exact_closed_five_field_dict() -> None:
    source = BINDING.read_text(encoding="utf-8")
    parser = _definition_body(source, "cancel_request_from_dict_v2")

    assert "run_control::kCancelRequestFieldsV2.size()" in parser
    assert "py::len(request)" in parser
    assert "request.contains" in parser
    assert "for (const std::string_view field" in parser
    assert parser.index("py::len(request)") < parser.index("out.api_version")
    assert parser.index("request.contains") < parser.index("out.api_version")
    for field in (
        "api_version",
        "request_id",
        "expected_controller_id",
        "expected_owner_epoch",
        "reason",
    ):
        assert f'request["{field}"]' in parser


def test_cancel_receipt_conversion_preserves_the_exact_public_contract() -> None:
    source = BINDING.read_text(encoding="utf-8")
    assert _initializer_literals(source, "kCancelReceiptFieldsV2") == (
        "request_id",
        "controller_id",
        "owner_epoch",
        "disposition",
        "first_cancel_request_id",
        "cancel_sequence",
        "state_observed",
        "terminal_status_changed",
    )

    match = re.search(
        r"receipt_to_dict_v2\s*\(\s*"
        r"const run_control::CancelRequestReceiptV2& receipt\s*\)\s*\{",
        source,
    )
    assert match is not None
    body = _braced_body(source, match.end() - 1)
    assert _dict_assignment_keys(body) == _initializer_literals(
        source, "kCancelReceiptFieldsV2"
    )
    assert 'out["schema_version"]' not in body
    assert 'out["accepted_control_coordinate"]' not in body
    assert "py::none()" in body


def test_all_run_control_receipts_have_inline_dict_serializers() -> None:
    source = BINDING.read_text(encoding="utf-8")

    for receipt_type in (
        "StartReceiptV2",
        "GenerationStartReceiptV2",
        "ControlWindowReceiptV2",
        "CancelRequestReceiptV2",
        "CancellationObservationReceiptV2",
        "FailureReceiptV2",
        "TerminalReceiptV2",
        "BoundaryResolutionReceiptV2",
    ):
        assert re.search(
            r"inline py::dict receipt_to_dict_v2\s*\(\s*"
            rf"const run_control::{receipt_type}& receipt\s*\)",
            source,
        )

    for helper in (
        "identity_to_dict_v2",
        "control_coordinate_to_dict_v2",
        "boundary_metadata_to_dict_v2",
        "failure_metadata_to_dict_v2",
        "manifest_to_dict_v2",
    ):
        assert f"inline py::dict {helper}" in source


def test_boundary_ack_parser_is_closed_and_all_settlement_calls_release_gil() -> None:
    source = BINDING.read_text(encoding="utf-8")
    parser = _definition_body(source, "boundary_metadata_from_dict_v2")
    cancel_ack = _definition_body(source, "acknowledge_cancellation_checkpoint_v2")
    complete_ack = _definition_body(source, "acknowledge_run_complete_v2")
    failure = _definition_body(source, "publish_failure_v2")

    assert "exactly 11 fields" in parser
    assert "py::len(boundary)" in parser
    assert "boundary.contains" in parser
    for body, call in (
        (cancel_ack, "controller.acknowledge_cancellation_checkpoint"),
        (complete_ack, "controller.acknowledge_run_complete"),
        (failure, "controller.publish_failure"),
    ):
        assert "py::gil_scoped_release release" in body
        assert call in body


def test_public_binding_is_narrow_and_releases_the_gil_around_control_calls() -> None:
    source = BINDING.read_text(encoding="utf-8")
    request = _definition_body(source, "request_cancel_v2")
    manifest = _definition_body(source, "manifest_v2")
    final_result = _definition_body(source, "final_result_v2")
    binding = _definition_body(source, "bind_run_control_v2")

    assert "cancel_request_from_dict_v2(request)" in request
    assert "py::gil_scoped_release release" in request
    assert "controller.request_cancel(parsed)" in request
    assert "receipt_to_dict_v2(receipt)" in request
    assert "py::gil_scoped_release release" in manifest
    assert "controller.manifest()" in manifest
    assert "manifest_to_dict_v2(manifest)" in manifest
    assert "py::gil_scoped_release release" in final_result
    assert "controller.final_result()" in final_result
    assert "return py::none()" in final_result

    assert '"RunControlV2"' in binding
    assert "std::shared_ptr<run_control::RunControlV2>" in binding
    assert ".def(py::init<>())" in binding
    assert "py::init<std::string, std::int64_t>()" in binding
    assert 'py::arg("owner_epoch") = 1' in binding
    assert '.def("request_cancel", &request_cancel_v2' in binding
    assert '"acknowledge_cancellation_checkpoint"' in binding
    assert '"acknowledge_run_complete"' in binding
    assert '"publish_failure"' in binding
    assert '.def("manifest", &manifest_v2)' in binding
    assert '.def("final_result", &final_result_v2)' in binding
    for prohibited_method in (
        "start_run",
        "begin_generation",
        "inter_tick_control_point",
        "consume_cancellation_at_boundary",
        "publish_run_complete",
    ):
        assert f'.def("{prohibited_method}"' not in binding
