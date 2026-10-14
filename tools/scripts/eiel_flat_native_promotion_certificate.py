#!/usr/bin/env python3
"""Generate and validate the fail-closed Flat native promotion certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "engine"
for import_root in (ENGINE, ROOT):
    import_root_s = str(import_root)
    if import_root_s not in sys.path:
        sys.path.insert(0, import_root_s)

from native_runtime.promotion_certificate import (  # noqa: E402
    CERTIFICATE_SCHEMA_VERSION,
    PromotionCertificateValidationError,
    canonical_promotion_certificate_bytes,
    generate_promotion_certificate,
    load_promotion_certificate,
)


CERTIFICATE_FILENAME = f"{CERTIFICATE_SCHEMA_VERSION}.json"
EXIT_REJECTED = 1
EXIT_INVALID = 2


class PromotionCertificateCliError(ValueError):
    """Raised for malformed CLI input or an output write failure."""


def _reject_json_constant(value: str) -> None:
    raise PromotionCertificateCliError(
        f"non-finite JSON number {value!r} is not permitted"
    )


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PromotionCertificateCliError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _load_strict_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PromotionCertificateCliError(f"cannot read {label} {path}: {exc}") from exc
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_fields,
            parse_constant=_reject_json_constant,
        )
    except PromotionCertificateCliError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PromotionCertificateCliError(
            f"malformed JSON in {label} {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise PromotionCertificateCliError(f"{label} must contain one JSON object")
    return payload


def _write_temp_file(target: Path, contents: bytes) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o644)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_certificate_atomic(
    output_dir: Path,
    certificate: dict[str, Any],
) -> tuple[Path, Path, str]:
    directory = output_dir.expanduser().resolve()
    target = directory / CERTIFICATE_FILENAME
    sidecar = Path(f"{target}.sha256")
    canonical = canonical_promotion_certificate_bytes(certificate)
    digest = hashlib.sha256(canonical).hexdigest()
    sidecar_contents = f"{digest}  {target.name}\n".encode("ascii")
    certificate_temp: Path | None = None
    sidecar_temp: Path | None = None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        certificate_temp = _write_temp_file(target, canonical)
        sidecar_temp = _write_temp_file(sidecar, sidecar_contents)
        os.replace(certificate_temp, target)
        certificate_temp = None
        os.replace(sidecar_temp, sidecar)
        sidecar_temp = None
        _fsync_directory(directory)
    except OSError as exc:
        raise PromotionCertificateCliError(
            f"cannot atomically write promotion certificate under {directory}: {exc}"
        ) from exc
    finally:
        if certificate_temp is not None:
            certificate_temp.unlink(missing_ok=True)
        if sidecar_temp is not None:
            sidecar_temp.unlink(missing_ok=True)
    return target, sidecar, digest


def _summary(
    certificate: dict[str, Any],
    *,
    mode: str,
    certificate_path: Path | None = None,
    sidecar_path: Path | None = None,
    digest: str | None = None,
) -> dict[str, Any]:
    verdict = certificate["verdict"]
    eligible = bool(verdict["eligible"])
    result: dict[str, Any] = {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "mode": mode,
        "status": "passed" if eligible else "rejected",
        "ok": eligible,
        "decision": verdict["decision"],
        "reasons": list(verdict["reasons"]),
    }
    if certificate_path is not None:
        result["certificate_path"] = str(certificate_path)
    if sidecar_path is not None:
        result["sha256_path"] = str(sidecar_path)
    if digest is not None:
        result["sha256"] = digest
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or validate a Flat native promotion certificate",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    generate = subparsers.add_parser(
        "generate",
        help="Generate a promotable certificate from strict JSON fields",
    )
    generate.add_argument("--fields-json", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)

    validate = subparsers.add_parser(
        "validate",
        help="Validate a canonical certificate and its SHA-256 sidecar",
    )
    validate.add_argument("--certificate-json", type=Path, required=True)
    return parser


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.mode == "generate":
            fields = _load_strict_json_object(args.fields_json, "fields input")
            certificate = generate_promotion_certificate(fields)
            if not bool(certificate["verdict"]["eligible"]):
                _emit(_summary(certificate, mode="generate"))
                return EXIT_REJECTED
            target, sidecar, digest = _write_certificate_atomic(
                args.output_dir,
                certificate,
            )
            _emit(
                _summary(
                    certificate,
                    mode="generate",
                    certificate_path=target,
                    sidecar_path=sidecar,
                    digest=digest,
                )
            )
            return 0

        certificate_path = args.certificate_json.expanduser().resolve()
        certificate = load_promotion_certificate(certificate_path)
        summary = _summary(
            certificate,
            mode="validate",
            certificate_path=certificate_path,
            sidecar_path=Path(f"{certificate_path}.sha256"),
            digest=hashlib.sha256(certificate_path.read_bytes()).hexdigest(),
        )
        _emit(summary)
        return 0 if bool(certificate["verdict"]["eligible"]) else EXIT_REJECTED
    except (PromotionCertificateCliError, PromotionCertificateValidationError) as exc:
        _emit(
            {
                "schema_version": CERTIFICATE_SCHEMA_VERSION,
                "mode": args.mode,
                "status": "invalid",
                "ok": False,
                "error": str(exc),
            }
        )
        return EXIT_INVALID
    except OSError as exc:
        _emit(
            {
                "schema_version": CERTIFICATE_SCHEMA_VERSION,
                "mode": args.mode,
                "status": "invalid",
                "ok": False,
                "error": f"cannot complete {args.mode}: {exc}",
            }
        )
        return EXIT_INVALID


if __name__ == "__main__":
    raise SystemExit(main())
