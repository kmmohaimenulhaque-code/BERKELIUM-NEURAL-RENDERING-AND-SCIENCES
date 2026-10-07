"""Base model and id types shared by schema and geometry (kept separate to avoid import cycles)."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

Id = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_\-]{0,63}$", description="Stable node id")]


class Strict(BaseModel):
    """Base for all canonical models: unknown fields are rejected (authority separation, ADR-001)."""

    model_config = ConfigDict(extra="forbid", frozen=False, populate_by_name=True)
