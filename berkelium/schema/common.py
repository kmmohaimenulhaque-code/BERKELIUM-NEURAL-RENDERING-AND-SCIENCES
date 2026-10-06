"""Shared schema primitives: quantities, references, provenance, diagnostics."""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..units import Quantity, UnitError, parse_unit

SCHEMA_VERSION = "0.1.0"

Id = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_\-]{0,63}$", description="Stable node id")]
# "component.port" or "component" reference
Ref = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_\-]{0,63}(\.[A-Za-z_][A-Za-z0-9_\-]{0,63})*$")]

_AUTHOR_RE = re.compile(r"^(user|system|llm:[^\s@]+(@[^\s]+)?|cem:[a-z0-9_.]+@\d+\.\d+(\.\d+)?)$")


class Strict(BaseModel):
    """Base for all canonical models: unknown fields are rejected (authority separation, ADR-001)."""

    model_config = ConfigDict(extra="forbid", frozen=False, populate_by_name=True)


class QuantityModel(Strict):
    """A value with an explicit unit, e.g. {"value": 12, "unit": "mm"}."""

    value: float
    unit: str = "1"

    @field_validator("unit")
    @classmethod
    def _unit_parses(cls, v: str) -> str:
        try:
            parse_unit(v)
        except UnitError as e:
            raise ValueError(str(e)) from None
        return v

    def q(self) -> Quantity:
        return Quantity.of(self.value, self.unit)

    @classmethod
    def from_q(cls, q: Quantity, unit: str) -> QuantityModel:
        return cls(value=q.to(unit), unit=unit)


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
