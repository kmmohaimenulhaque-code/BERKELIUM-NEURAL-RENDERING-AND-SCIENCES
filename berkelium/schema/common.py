"""Shared schema primitives: quantities, references, provenance, diagnostics."""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import Field

from .._base import Id, Strict  # noqa: F401  (re-exported)
from .._qmodel import QuantityModel  # noqa: F401  (re-exported)

SCHEMA_VERSION = "0.1.0"

# "component.port" or "component" reference
Ref = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_\-]{0,63}(\.[A-Za-z_][A-Za-z0-9_\-]{0,63})*$")]

_AUTHOR_RE = re.compile(r"^(user|system|llm:[^\s@]+(@[^\s]+)?|cem:[a-z0-9_.]+@\d+\.\d+(\.\d+)?)$")


Author = Annotated[str, Field(pattern=_AUTHOR_RE.pattern,
                              description="user | system | llm:<model>[@adapter] | cem:<name>@<version>")]


class Provenance(Strict):
    """Deterministic provenance. Wall-clock timestamps are deliberately excluded from hashed content."""

    author: Author
    generator: str | None = Field(None, description="Code that produced this node, name@version")
    inputs_hash: str | None = Field(None, description="sha256 of the canonical inputs")
    parent_revision: int | None = None
    references: list[str] = Field(default_factory=list, description="Source/standard citations used")


Severity = Literal["info", "warning", "error"]


class Diagnostic(Strict):
    code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{1,63}$")
    severity: Severity
    message: str
    path: str | None = Field(None, description="JSON Pointer into the document, if applicable")
    hint: str | None = None
