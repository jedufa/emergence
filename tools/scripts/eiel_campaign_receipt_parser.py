"""Deterministic parsing of EIEL campaign execution receipts from mixed stdout.

The campaign runner normally prints one JSON execution receipt. Some providers
also emit diagnostic log lines to stdout before or after that receipt. This
module fail-closed extracts exactly one authoritative campaign execution receipt
from such mixed output and binds that interpretation to the raw stdout/stderr
and completed manifest state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


RECEIPT_SCHEMA_VERSION = "eiel_campaign_execution_receipt_v1"
INTERPRETATION_SCHEMA_VERSION = (
    "eiel_campaign_mixed_stdout_interpretation_receipt_v1"
)


class CampaignReceiptParseError(ValueError):
    """Mixed stdout cannot support a unique authoritative receipt."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _json_object_candidates(stdout: bytes) -> list[tuple[int, int, Mapping[str, Any]]]:
    text = stdout.decode("utf-8", "replace")
    decoder = json.JSONDecoder()
    candidates: list[tuple[int, int, Mapping[str, Any]]] = []
    for char_start, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, char_end = decoder.raw_decode(text[char_start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, Mapping):
            continue
        byte_start = len(text[:char_start].encode("utf-8"))
        byte_end = byte_start + len(text[char_start : char_start + char_end].encode("utf-8"))
        candidates.append((byte_start, byte_end, parsed))
    return candidates


def _validate_execution_receipt(
    receipt: Mapping[str, Any], *, expected_campaign_id: str
) -> None:
    required = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "campaign_id": expected_campaign_id,
        "ok": True,
        "status": "completed",
        "reporting_parity_status": "reporting_parity_proven",
    }
    for key, expected in required.items():
        actual = receipt.get(key)
        if actual != expected:
            raise CampaignReceiptParseError(
                f"execution receipt field {key!r} mismatch: expected {expected!r}, got {actual!r}"
            )
    for key in (
        "execution_route_sha256",
        "plan_sha256",
        "provider_id",
        "profile_id",
        "reporting_parity_receipt_path",
    ):
        value = receipt.get(key)
        if not isinstance(value, str) or not value:
            raise CampaignReceiptParseError(
                f"execution receipt field {key!r} must be a non-empty string"
            )
    counts = receipt.get("counts")
    if not isinstance(counts, Mapping):
        raise CampaignReceiptParseError("execution receipt counts must be an object")
    if counts.get("failed") != 0 or counts.get("interrupted") != 0:
        raise CampaignReceiptParseError(
            f"execution receipt has nonzero failed/interrupted counts: {counts!r}"
        )
    if counts.get("pending") != 0 or counts.get("running") != 0:
        raise CampaignReceiptParseError(
            f"execution receipt has nonzero pending/running counts: {counts!r}"
        )


@dataclass(frozen=True, slots=True)
class ParsedCampaignReceipt:
    byte_start: int
    byte_end: int
    receipt: Mapping[str, Any]

    @property
    def extracted_sha256(self) -> str:
        return _sha256_bytes(_canonical_bytes(self.receipt))


def parse_campaign_execution_receipt(
    stdout: bytes, *, expected_campaign_id: str
) -> ParsedCampaignReceipt:
    """Extract exactly one valid campaign execution receipt from stdout bytes."""

    valid: list[ParsedCampaignReceipt] = []
    for byte_start, byte_end, candidate in _json_object_candidates(stdout):
        if candidate.get("schema_version") != RECEIPT_SCHEMA_VERSION:
            continue
        try:
            _validate_execution_receipt(
                candidate, expected_campaign_id=expected_campaign_id
            )
        except CampaignReceiptParseError:
            continue
        valid.append(
            ParsedCampaignReceipt(
                byte_start=byte_start,
                byte_end=byte_end,
                receipt=dict(candidate),
            )
        )
    if not valid:
        raise CampaignReceiptParseError(
            "stdout does not contain one valid authoritative campaign execution receipt"
        )
    unique_payloads = {_canonical_bytes(item.receipt) for item in valid}
    if len(unique_payloads) != 1 or len(valid) != 1:
        raise CampaignReceiptParseError(
            f"stdout contains ambiguous campaign execution receipts: {len(valid)} valid candidates"
        )
    return valid[0]


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - exact JSON exception not important here
        raise CampaignReceiptParseError(f"cannot read manifest JSON: {path}") from exc
    if not isinstance(loaded, Mapping):
        raise CampaignReceiptParseError(f"manifest is not an object: {path}")
    return loaded


def summarize_manifest_completion(run_root: Path) -> dict[str, Any]:
    """Return fail-closed manifest completion evidence for a campaign run root."""

    manifests = sorted(run_root.rglob("manifest.json"))
    if not manifests:
        raise CampaignReceiptParseError(f"no manifest.json files found under {run_root}")
    statuses: dict[str, int] = {}
    errors: list[dict[str, Any]] = []
    for path in manifests:
        manifest = _load_json(path)
        status = str(manifest.get("status"))
        statuses[status] = statuses.get(status, 0) + 1
        error = manifest.get("error")
        if status != "completed" or error is not None:
            errors.append(
                {
                    "path": str(path.relative_to(run_root)),
                    "status": status,
                    "error": error,
                }
            )
    if errors:
        raise CampaignReceiptParseError(
            f"manifest completion check failed with {len(errors)} non-completed/error manifests"
        )
    return {
        "run_root": str(run_root),
        "manifest_count": len(manifests),
        "statuses": statuses,
        "errors": [],
        "all_completed": statuses == {"completed": len(manifests)},
    }


def build_interpretation_receipt(
    *,
    stdout_path: Path,
    stderr_path: Path | None,
    run_root: Path,
    expected_campaign_id: str,
    command_returncode: int,
) -> dict[str, Any]:
    """Build a supplemental receipt without rewriting original campaign output."""

    stdout = stdout_path.read_bytes()
    stderr = stderr_path.read_bytes() if stderr_path is not None else b""
    if command_returncode != 0:
        raise CampaignReceiptParseError(
            f"campaign command return code must be zero, got {command_returncode}"
        )
    parsed = parse_campaign_execution_receipt(
        stdout, expected_campaign_id=expected_campaign_id
    )
    manifest_summary = summarize_manifest_completion(run_root)
    receipt = dict(parsed.receipt)
    if receipt.get("campaign_id") != expected_campaign_id:
        raise CampaignReceiptParseError("campaign id drift after receipt parsing")
    if receipt.get("status") != "completed" or receipt.get("ok") is not True:
        raise CampaignReceiptParseError("parsed receipt is not a completed ok receipt")
    unsigned: dict[str, Any] = {
        "schema_version": INTERPRETATION_SCHEMA_VERSION,
        "campaign_id": expected_campaign_id,
        "command_returncode": command_returncode,
        "stdout_path": str(stdout_path),
        "stdout_sha256": _sha256_bytes(stdout),
        "stderr_path": str(stderr_path) if stderr_path is not None else None,
        "stderr_sha256": _sha256_bytes(stderr),
        "extracted_receipt_byte_start": parsed.byte_start,
        "extracted_receipt_byte_end": parsed.byte_end,
        "extracted_receipt_sha256": parsed.extracted_sha256,
        "extracted_receipt_schema_version": receipt.get("schema_version"),
        "extracted_receipt_status": receipt.get("status"),
        "extracted_receipt_ok": receipt.get("ok"),
        "reporting_parity_status": receipt.get("reporting_parity_status"),
        "execution_route_sha256": receipt.get("execution_route_sha256"),
        "plan_sha256": receipt.get("plan_sha256"),
        "provider_id": receipt.get("provider_id"),
        "profile_id": receipt.get("profile_id"),
        "manifest_completion": manifest_summary,
        "interpretation": "campaign_completed_with_mixed_stdout_receipt",
    }
    return {**unsigned, "receipt_sha256": _sha256_bytes(_canonical_bytes(unsigned))}


def write_interpretation_receipt(
    output_path: Path,
    *,
    stdout_path: Path,
    stderr_path: Path | None,
    run_root: Path,
    expected_campaign_id: str,
    command_returncode: int,
) -> dict[str, Any]:
    receipt = build_interpretation_receipt(
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        run_root=run_root,
        expected_campaign_id=expected_campaign_id,
        command_returncode=command_returncode,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_bytes(receipt) + b"\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", required=True, type=Path)
    parser.add_argument("--stderr", type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--return-code", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        receipt = write_interpretation_receipt(
            args.output,
            stdout_path=args.stdout,
            stderr_path=args.stderr,
            run_root=args.run_root,
            expected_campaign_id=args.campaign_id,
            command_returncode=args.return_code,
        )
    except CampaignReceiptParseError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                "receipt_path": str(args.output),
                "receipt_sha256": receipt["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
