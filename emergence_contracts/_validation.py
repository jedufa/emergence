"""Internal validation helpers shared by immutable contract DTOs."""

from __future__ import annotations

from collections.abc import Iterable

from .errors import contract_invalid


INT64_MAX = (1 << 63) - 1


def require_int(
    value: object,
    *,
    field: str,
    minimum: int = 0,
    maximum: int = INT64_MAX,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise contract_invalid(
            "value must be an integer", field=field, value_type=type(value).__name__
        )
    if value < minimum or value > maximum:
        raise contract_invalid(
            f"integer must be in [{minimum}, {maximum}]", field=field
        )
    return value


def require_optional_positive_int(value: object, *, field: str) -> None:
    if value is not None:
        require_int(value, field=field, minimum=1)


def require_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise contract_invalid(
            "value must be a non-empty, unpadded string",
            field=field,
            value_type=type(value).__name__,
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise contract_invalid("value contains a control character", field=field)
    return value


def require_instance(value: object, expected: type[object], *, field: str) -> None:
    if not isinstance(value, expected):
        raise contract_invalid(f"value must be {expected.__name__}", field=field)


def require_tuple_instances(
    values: object,
    expected: type[object],
    *,
    field: str,
    unique: bool = False,
    sorted_by_string: bool = False,
) -> None:
    if not isinstance(values, tuple):
        raise contract_invalid(
            "canonical collection must be a tuple",
            field=field,
            value_type=type(values).__name__,
        )
    if any(not isinstance(value, expected) for value in values):
        raise contract_invalid(
            f"all values must be {expected.__name__}", field=field
        )
    rendered = [str(value) for value in values]
    if unique and len(rendered) != len(set(rendered)):
        raise contract_invalid("canonical collection must be unique", field=field)
    if sorted_by_string and rendered != sorted(rendered):
        raise contract_invalid("canonical collection must be sorted", field=field)


def require_sorted_unique_text(values: object, *, field: str) -> None:
    require_tuple_instances(
        values, str, field=field, unique=True, sorted_by_string=True
    )
    assert isinstance(values, tuple)
    for value in values:
        require_text(value, field=field)


def sorted_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))
