import enum
from typing import Any, List, Literal, Optional

import pytest

from newera import function_schema, type_schema
from newera.schema import coerce_arguments, parse_docstring


class Day(str, enum.Enum):
    SAT = "sat"
    SUN = "sun"


def book(patient: str, day: Literal["sat", "sun"], minutes: int = 30, vip: bool = False,
         note: Optional[str] = None, tags: Optional[List[str]] = None, price: float = 0.0) -> dict:
    """Book an appointment.

    Second line of the summary.

    Args:
        patient: Full name of the patient.
        day (str): Which day. Saturday
            or Sunday.
        minutes: Length of the visit.
        vip: Priority booking.

    Returns:
        A booking.
    """


def test_summary_and_args():
    summary, docs = parse_docstring(book.__doc__)
    assert summary == "Book an appointment."
    assert docs["patient"] == "Full name of the patient."
    assert docs["day"] == "Which day. Saturday or Sunday."
    assert "Returns" not in docs and "A booking." not in docs.values()


def test_function_schema():
    description, params = function_schema(book)
    assert description == "Book an appointment."
    p = params["properties"]
    assert p["patient"] == {"type": "string", "description": "Full name of the patient."}
    assert p["day"]["enum"] == ["sat", "sun"] and p["day"]["type"] == "string"
    assert p["minutes"] == {"type": "integer", "default": 30, "description": "Length of the visit."}
    assert p["vip"]["type"] == "boolean" and p["vip"]["default"] is False
    assert p["note"] == {"type": "string"}
    assert p["tags"] == {"type": "array", "items": {"type": "string"}}
    assert p["price"]["type"] == "number"
    assert params["required"] == ["patient", "day"]
    assert params["additionalProperties"] is False


def test_enum_and_union_and_any():
    assert type_schema(Day) == {"enum": ["sat", "sun"], "type": "string"}
    assert type_schema(Any) == {}
    assert type_schema(list) == {"type": "array"}
    assert type_schema(dict) == {"type": "object"}
    assert type_schema(Optional[int]) == {"type": "integer"}
    assert type_schema(Literal[1, 2]) == {"enum": [1, 2], "type": "integer"}


def test_pep604_unions():
    import sys
    if sys.version_info < (3, 10):
        pytest.skip("X | None needs Python 3.10")
    ns = {}
    exec("def f(a: int | None = None, b: str | int = 'x', c: list[int] = None): pass", ns)
    _, params = function_schema(ns["f"])
    assert params["properties"]["a"] == {"type": "integer"}
    assert params["properties"]["b"]["anyOf"] == [{"type": "string"}, {"type": "integer"}]
    assert params["properties"]["c"] == {"type": "array", "items": {"type": "integer"}}
    assert "required" not in params


def test_unsupported_type_is_a_clear_error():
    class Thing: ...

    def f(x: Thing): ...

    with pytest.raises(TypeError, match="Unsupported type"):
        function_schema(f)


def test_enum_arguments_are_coerced():
    def f(day: Day, n: int): ...

    out = coerce_arguments(f, {"day": "sun", "n": 2})
    assert out["day"] is Day.SUN and out["n"] == 2


def test_no_docstring_and_no_params():
    def f(): ...

    assert function_schema(f) == ("", {"type": "object", "properties": {}, "additionalProperties": False})
