"""Canonical packaged JSON Schema validators for public file boundaries."""

from __future__ import annotations

import json
from functools import cache
from importlib.resources import files
from typing import Any, Literal

import jsonschema

SchemaName = Literal["scene", "cameras", "render-request", "render-result", "run"]

_SCHEMA_FILES: dict[SchemaName, str] = {
    "scene": "scene.schema.json",
    "cameras": "cameras.schema.json",
    "render-request": "render-request.schema.json",
    "render-result": "render-result.schema.json",
    "run": "run.schema.json",
}


@cache
def load_schema(name: SchemaName) -> dict[str, Any]:
    resource = files("nht_pipeline").joinpath("schemas", _SCHEMA_FILES[name])
    payload = json.loads(resource.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(payload)
    return payload


@cache
def schema_validator(name: SchemaName) -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(
        load_schema(name), format_checker=jsonschema.FormatChecker()
    )


def validate_schema_payload(
    name: SchemaName, payload: Any, *, context: str
) -> None:
    errors = sorted(
        schema_validator(name).iter_errors(payload),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if not errors:
        return
    error = errors[0]
    location = ".".join(str(part) for part in error.absolute_path) or "<root>"
    raise ValueError(
        f"{context} violates canonical {name} schema at {location}: {error.message}"
    ) from error
