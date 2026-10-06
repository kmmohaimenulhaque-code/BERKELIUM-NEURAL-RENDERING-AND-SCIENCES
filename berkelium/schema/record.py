"""DesignRecord — the full layered record written by the core."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ..geometry.ir import GeometryGraph
from .common import SCHEMA_VERSION, Provenance, Strict
from .design import DesignProposal, Intent, Specification, Structure
from .evaluation import Evaluation


class RealizedComponent(Strict):
    component: str
    cem: str | None = None
    parameters: dict[str, float | int | bool | str] = Field(default_factory=dict,
                                                           description="Resolved parameters (SI-free plain values; "
                                                                       "units in parameter_units)")
    parameter_units: dict[str, str] = Field(default_factory=dict)
    geometry: GeometryGraph | None = None
    provenance: Provenance


class DesignRecord(Strict):
    schema_version: Literal["0.1.0"] = SCHEMA_VERSION
    kind: Literal["design_record"] = "design_record"
    id: str
    revision: int = 0
    intent: Intent
    specification: Specification
    structure: Structure
    realized: list[RealizedComponent] = Field(default_factory=list, description="Layer 4 as executed")
    evaluation: Evaluation = Field(default_factory=Evaluation)
    provenance: Provenance
    proposal_hash: str
    content_hash: str = Field("", description="sha256 of the canonical record with content_hash=''")

    def proposal(self) -> DesignProposal:
        return DesignProposal(intent=self.intent, specification=self.specification, structure=self.structure)
