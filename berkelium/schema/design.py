"""Design layers 1-4 (intent, specification, structure, geometry) and the LLM-writable DesignProposal."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from ..geometry.ir import GeometryGraph
from ..manufacturing import ManufacturingProfile
from ..physics.schema import AnalysisCase, Material
from .common import SCHEMA_VERSION, Id, QuantityModel, Ref, Strict

ParamValue = QuantityModel | float | int | bool | str


def _check_expr(src: str) -> str:
    from ..expr import parse
    parse(src)  # raises ExprSyntaxError (a ValueError) -> pydantic validation error
    return src


# ------------------------------------------------------------------ layer 1: intent
class ExtractedRequirement(Strict):
    text: str = Field(description="Verbatim fragment of the prompt this requirement came from")
    span: Annotated[list[int], Field(min_length=2, max_length=2)] | None = None
    requirement_id: Id | None = None


class Intent(Strict):
    prompt: str | None = None
    summary: str | None = None
    extracted: list[ExtractedRequirement] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list, description="Assumptions the author made explicitly")
    open_questions: list[str] = Field(default_factory=list, description="Ambiguities to ask instead of guess")


# ------------------------------------------------------------------ layer 2: specification
Comparator = Literal["==", "<=", ">=", "<", ">", "approx"]


class Requirement(Strict):
    id: Id
    quantity: str = Field(description="What is required, e.g. 'ratio', 'center_distance', 'torque', "
                                      "or a dotted measurement name")
    comparator: Comparator
    target: QuantityModel
    tolerance: QuantityModel | None = Field(None, description="Absolute tolerance for '==' / 'approx'")
    strength: Literal["hard", "soft"] = "hard"
    priority: Annotated[int, Field(ge=0, le=10)] = 5
    source: Literal["user", "derived", "standard"] = "user"
    applies_to: Ref | None = None


ParamKind = Literal["length", "angle", "count", "real", "enum", "bool", "material_ref", "expr"]


class Parameter(Strict):
    id: Id
    kind: ParamKind
    value: ParamValue
    lower: float | None = None
    upper: float | None = None
    unit: str | None = Field(None, description="Unit of lower/upper bounds")
    choices: list[str] | None = None
    status: Literal["free", "fixed", "derived"] = "free"
    description: str | None = None

    @model_validator(mode="after")
    def _expr_kind(self) -> Parameter:
        if self.kind == "expr":
            if not isinstance(self.value, str):
                raise ValueError("expr parameters need a string expression value")
            _check_expr(self.value)
        if self.kind == "enum" and self.choices is not None and self.value not in self.choices:
            raise ValueError(f"enum value {self.value!r} not in choices")
        return self


class Constraint(Strict):
    id: Id
    expr: str = Field(description="Boolean expression over parameters and measurements")
    cls: Literal["parametric", "geometric", "assembly", "manufacturing", "performance"] = "parametric"
    strength: Literal["hard", "soft"] = "hard"
    description: str | None = None

    _v = field_validator("expr")(classmethod(lambda cls, v: _check_expr(v)))


class Specification(Strict):
    requirements: list[Requirement] = Field(default_factory=list)
    manufacturing: ManufacturingProfile | None = None
    parameters: list[Parameter] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    materials: list[Material] = Field(default_factory=list, description="Materials referenced by analyses")
    analyses: list[AnalysisCase] = Field(default_factory=list,
                                         description="Requested analyses (inputs only; results are core-written)")

    @model_validator(mode="after")
    def _unique(self) -> Specification:
        for name, items in (("requirement", self.requirements), ("parameter", self.parameters),
                            ("constraint", self.constraints)):
            ids = [i.id for i in items]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {name} ids")
        return self


# ------------------------------------------------------------------ layer 3: structure
class Frame(Strict):
    origin: Annotated[list[float], Field(min_length=3, max_length=3)] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    z_axis: Annotated[list[float], Field(min_length=3, max_length=3)] = Field(default_factory=lambda: [0.0, 0.0, 1.0])
    x_axis: Annotated[list[float], Field(min_length=3, max_length=3)] = Field(default_factory=lambda: [1.0, 0.0, 0.0])


class Port(Strict):
    id: Id
    kind: str = Field(description="Semantic type: shaft_bore, mount_face, fluid_port, gear_mesh, ...")
    frame: Frame = Field(default_factory=Frame)


class Placement(Strict):
    translation: Annotated[list[float], Field(min_length=3, max_length=3)] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation_axis: Annotated[list[float], Field(min_length=3, max_length=3)] = Field(default_factory=lambda: [0.0, 0.0, 1.0])
    rotation_deg: float = 0.0


class Component(Strict):
    id: Id
    kind: Literal["cem", "procedural"]
    cem: str | None = Field(None, pattern=r"^[a-z0-9_.]+@\d+\.\d+$", description="CEM reference name@major.minor")
    requirements: dict[str, ParamValue] = Field(default_factory=dict, description="CEM requirement inputs")
    parameters: dict[str, ParamValue] = Field(default_factory=dict, description="CEM parameter overrides")
    geometry: GeometryGraph | None = Field(None, description="Authored op graph (procedural components)")
    ports: list[Port] = Field(default_factory=list)
    placement: Placement = Field(default_factory=Placement)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _kind(self) -> Component:
        if self.kind == "cem" and (self.cem is None or self.geometry is not None):
            raise ValueError("cem components need 'cem' and must not author geometry")
        if self.kind == "procedural" and (self.geometry is None or self.cem is not None):
            raise ValueError("procedural components need 'geometry' and no 'cem'")
        return self


class Relation(Strict):
    id: Id
    kind: Literal["mate", "mesh", "fluid_connect", "attach", "pattern_of"]
    a: Ref
    b: Ref
    params: dict[str, ParamValue] = Field(default_factory=dict)


class Structure(Strict):
    components: Annotated[list[Component], Field(min_length=1)]
    relations: list[Relation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _refs(self) -> Structure:
        ids = [c.id for c in self.components]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate component ids")
        for r in self.relations:
            for ref in (r.a, r.b):
                if ref.split(".")[0] not in ids:
                    raise ValueError(f"relation {r.id!r} references unknown component {ref!r}")
        return self


# ------------------------------------------------------------------ the LLM-writable document
class DesignProposal(Strict):
    """Layers 1-4 only. The ONLY document a model may emit (ADR-001). Because ``extra='forbid'``
    is set on every model, a proposal cannot carry measurements, validation or simulation results."""

    schema_version: Literal["0.1.0"] = SCHEMA_VERSION
    kind: Literal["design_proposal"] = "design_proposal"
    intent: Intent = Field(default_factory=Intent)
    specification: Specification = Field(default_factory=Specification)
    structure: Structure
