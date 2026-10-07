"""Layer 5 — evaluation. Written ONLY by deterministic core code, never by a model (ADR-001)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from ..physics.schema import SimulationResult
from .common import Diagnostic, Id, QuantityModel, Strict

Status = Literal["pass", "fail", "warn", "indeterminate", "not_evaluated", "error"]
# indeterminate: evaluated, but the evidence (an error interval) neither proves nor disproves the claim
FidelityKind = Literal["rule", "geometric", "analytic", "numerical"]


class Fidelity(Strict):
    kind: FidelityKind
    method: str | None = Field(None, description="e.g. lewis_barth_preliminary, brep_check")
    assumptions: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list, description="Standards / sources for formulas used")


class ValidationResult(Strict):
    validator: str = Field(description="validator id@version")
    level: Annotated[int, Field(ge=0, le=7)]
    status: Status
    target: str = Field(description="What was checked: component id, relation id, constraint id, '$'")
    message: str
    measured: QuantityModel | None = None
    limit: QuantityModel | None = None
    comparator: str | None = None
    evidence: list[str] = Field(default_factory=list, description="artifact hashes, face tags, measurement ids")
    fidelity: Fidelity


class ValidationReport(Strict):
    results: list[ValidationResult] = Field(default_factory=list)
    summary: Status = "not_evaluated"
    counts: dict[str, int] = Field(default_factory=dict)
    highest_level_evaluated: int = -1
    physically_validated: bool = Field(False, description="True only if at least one L7 NUMERICAL result passed "
                                       "and no L7 result failed/errored/was indeterminate")

    @model_validator(mode="after")
    def _physically_validated_needs_evidence(self):
        if self.physically_validated:
            l7 = [r for r in self.results if r.level == 7]
            if not any(r.status == "pass" and r.fidelity.kind == "numerical" for r in l7) or \
                    any(r.status in ("fail", "error", "indeterminate") for r in l7):
                raise ValueError("physically_validated=True requires a passing L7 numerical result and no "
                                 "failed/errored/indeterminate L7 results")
        return self


class Measurement(Strict):
    id: Id
    target: str
    name: str = Field(description="volume, surface_area, bbox, center_of_mass, contact_ratio, ...")
    value: QuantityModel | None = None
    vector: list[float] | None = None
    method: Literal["backend", "analytic", "derived"]
    backend: str | None = Field(None, description="backend name@version for backend measurements")
    tolerance: QuantityModel | None = None
    references: list[str] = Field(default_factory=list)


class ArtifactRef(Strict):
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    format: Literal["step", "stl", "glb", "brep", "json"]
    target: str
    backend: str
    representation: Literal["brep", "mesh", "field", "data"]
    lossy: bool = Field(description="True if this artifact approximates the authoritative geometry")
    tolerance: QuantityModel | None = None
    bytes: int
    path: str | None = Field(None, description="Relative path in the artifact store")


class Derivation(Strict):
    """A CEM-computed engineering value (analytic), with its formula reference."""

    id: Id
    target: str
    value: QuantityModel
    formula: str
    references: list[str] = Field(default_factory=list)


class Evaluation(Strict):
    derivations: list[Derivation] = Field(default_factory=list)
    measurements: list[Measurement] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    validation: ValidationReport = Field(default_factory=ValidationReport)
    simulation: list[SimulationResult] = Field(default_factory=list,
                                               description="L7 solver results (deterministic core only)")
    diagnostics: list[Diagnostic] = Field(default_factory=list)
