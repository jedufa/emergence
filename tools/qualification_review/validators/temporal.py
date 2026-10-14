"""Declared freshness-contract validation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ..io import InputStore, ReviewInputError, json_pointer
from ..model import CheckResult


def parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ReviewInputError(f"invalid ISO-8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ReviewInputError(f"freshness timestamp lacks timezone: {value!r}")
    return parsed.astimezone(timezone.utc)


def format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def validate_freshness(
    spec: dict,
    store: InputStore,
    now: datetime,
) -> tuple[CheckResult, datetime | None]:
    result = CheckResult(spec["id"], "freshness")
    valid_until: datetime | None = None
    try:
        path = store.resolve(spec["path"])
        source = spec.get("timestamp_source", "json")
        if source == "mtime":
            store.hash_file(path)
            observed = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        else:
            document = store.read_json(spec["path"])
            observed = parse_utc(json_pointer(document, spec["timestamp_pointer"]))
        if observed > now + timedelta(seconds=spec.get("future_tolerance_seconds", 0)):
            result.fail(f"freshness timestamp is in the future: {format_utc(observed)}")
        valid_until = observed + timedelta(seconds=spec["max_age_seconds"])
        if now > valid_until:
            result.fail(
                f"input is stale: valid until {format_utc(valid_until)}, "
                f"checked at {format_utc(now)}"
            )
        result.details = {
            "path": spec["path"],
            "observed_at_utc": format_utc(observed),
            "valid_until_utc": format_utc(valid_until),
            "max_age_seconds": spec["max_age_seconds"],
        }
    except (ReviewInputError, OSError) as exc:
        result.fail(str(exc))
    return result, valid_until
