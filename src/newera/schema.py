"""A JSON Schema for a Python function, from its type hints and docstring.

    def book(patient: str, day: Literal["sat", "sun"], minutes: int = 30) -> dict:
        '''Book an appointment.

        Args:
            patient: Full name of the patient.
            day: Which day.
            minutes: Length of the visit.
        '''

becomes

    {"type": "object",
     "properties": {"patient": {"type": "string", "description": "Full name of the patient."},
                    "day": {"type": "string", "enum": ["sat", "sun"], "description": "Which day."},
                    "minutes": {"type": "integer", "default": 30, "description": "Length of the visit."}},
     "required": ["patient", "day"], "additionalProperties": false}

Supported: str, int, float, bool, Literal, Enum, Optional / X | None, list[X],
dict, Union, Any. A parameter with a default is not required.
"""

from __future__ import annotations

import enum
import inspect
import json
import re
import types
import typing
from typing import Any, Callable, Literal, Union, get_args, get_origin, get_type_hints

_NONE = type(None)


def type_schema(tp: Any) -> dict:
    """The JSON Schema of one annotation."""
    if tp is Any or tp is inspect.Parameter.empty:
        return {}
    origin = get_origin(tp)
    if origin is Literal:
        values = list(get_args(tp))
        kinds = {type(v) for v in values}
        schema: dict = {"enum": values}
        if kinds == {str}:
            schema["type"] = "string"
        elif kinds <= {int} and bool not in kinds:
            schema["type"] = "integer"
        return schema
    if origin is Union or (hasattr(types, "UnionType") and origin is getattr(types, "UnionType")):
        args = [a for a in get_args(tp) if a is not _NONE]
        if len(args) == 1:
            return type_schema(args[0])
        return {"anyOf": [type_schema(a) for a in args]}
    if origin in (list, set, frozenset, tuple, typing.List, typing.Sequence) or tp in (list, set, tuple):
        args = get_args(tp)
        schema = {"type": "array"}
        if args and args[0] is not Ellipsis:
            schema["items"] = type_schema(args[0])
        return schema
    if origin in (dict, typing.Dict) or tp is dict:
        return {"type": "object"}
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        values = [m.value for m in tp]
        schema = {"enum": values}
        if all(isinstance(v, str) for v in values):
            schema["type"] = "string"
        elif all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            schema["type"] = "integer"
        return schema
    if tp is bool:
        return {"type": "boolean"}
    if tp is int:
        return {"type": "integer"}
    if tp is float:
        return {"type": "number"}
    if tp is str:
        return {"type": "string"}
    raise TypeError("Unsupported type for a tool parameter: %r. Use str, int, float, bool, Literal, "
                    "Enum, Optional, list[...], dict or Any." % (tp,))


_SECTION = re.compile(r"^\s*(Args|Arguments|Parameters|Params)\s*:\s*$", re.I)
_OTHER_SECTION = re.compile(r"^\s*(Returns?|Raises?|Yields?|Examples?|Notes?)\s*:\s*$", re.I)
_PARAM = re.compile(r"^(\*{0,2}\w+)\s*(?:\(([^)]*)\))?\s*:\s*(.*)$")


def parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """(summary, {param: description}) from a Google-style docstring. The
    summary is the text before a blank line or the Args section."""
    if not doc:
        return "", {}
    lines = inspect.cleandoc(doc).splitlines()
    summary: list[str] = []
    i = 0
    while i < len(lines) and lines[i].strip() and not _SECTION.match(lines[i]) and not _OTHER_SECTION.match(lines[i]):
        summary.append(lines[i].strip())
        i += 1
    params: dict[str, str] = {}
    in_args, current, indent = False, None, None
    for line in lines[i:]:
        if _SECTION.match(line):
            in_args, current, indent = True, None, None
            continue
        if _OTHER_SECTION.match(line):
            in_args = False
            continue
        if not in_args or not line.strip():
            continue
        depth = len(line) - len(line.lstrip())
        if indent is None:
            indent = depth
        m = _PARAM.match(line.strip()) if depth <= indent else None
        if m:
            current = m.group(1).lstrip("*")
            params[current] = m.group(3).strip()
        elif current:
            params[current] = (params[current] + " " + line.strip()).strip()
    return " ".join(summary), params


def function_schema(fn: Callable) -> tuple[str, dict]:
    """(description, parameters schema) for `fn`."""
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:  # noqa: BLE001 - unresolved forward references: fall back to raw annotations
        hints = {k: p.annotation for k, p in sig.parameters.items()}
    summary, docs = parse_docstring(fn.__doc__)
    props: dict[str, dict] = {}
    required: list[str] = []
    for name, p in sig.parameters.items():
        if name in ("self", "cls") or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        schema = type_schema(hints.get(name, p.annotation))
        if name in docs:
            schema["description"] = docs[name]
        if p.default is inspect.Parameter.empty:
            required.append(name)
        else:
            default = p.default.value if isinstance(p.default, enum.Enum) else p.default
            if default is not None:
                try:
                    json.dumps(default)
                    schema["default"] = default
                except (TypeError, ValueError):
                    pass
        props[name] = schema
    params: dict = {"type": "object", "properties": props, "additionalProperties": False}
    if required:
        params["required"] = required
    return summary, params


def coerce_arguments(fn: Callable, args: dict) -> dict:
    """The model's JSON arguments as Python values: Enum parameters get their
    member, everything else is already right (the server validated it)."""
    try:
        hints = get_type_hints(fn)
    except Exception:  # noqa: BLE001
        return args
    out = dict(args)
    for name, value in args.items():
        tp = hints.get(name)
        if tp is None:
            continue
        origin = get_origin(tp)
        if origin is Union or (hasattr(types, "UnionType") and origin is getattr(types, "UnionType")):
            members = [a for a in get_args(tp) if a is not _NONE]
            tp = members[0] if len(members) == 1 else tp
        if isinstance(tp, type) and issubclass(tp, enum.Enum) and not isinstance(value, enum.Enum):
            try:
                out[name] = tp(value)
            except ValueError:
                pass
    return out
