"""Physics layer schemas (PHYSICS-1).

Two halves, mirroring ADR-001:

* INPUTS — ``AnalysisCase`` and everything it contains (materials, regions, loads, BCs, mesh/solver/
  convergence specs, quantities of interest). A model MAY propose these inside a DesignProposal.
* RESULTS — ``SimulationResult`` and everything it contains (estimates, convergence studies, mesh
  statistics, provenance). Written ONLY by deterministic solver code; they live under
  ``DesignRecord.evaluation.simulation`` which a proposal cannot contain.

Unit convention at the physics boundary: every load / material value is a QuantityModel with an explicit
unit and is converted to SI exactly once (``si()``). Geometry-frame selectors use millimetres, matching the
Geometry IR and implicit-field convention.

Central abstraction — the Estimate: a computed quantity is never a bare float. It carries a value, a
fidelity, a status, and separately-accounted error bounds (discretisation, model-form). Requirement checks
over Estimates are three-valued (pass / fail / indeterminate) on the whole error interval.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .._base import Id, Strict
from .._qmodel import QuantityModel
from ..units import Quantity, check_dim

PHYSICS_SCHEMA_VERSION = "0.1.0"

PhysicsKind = Literal["structural_linear_static", "thermal_steady", "pipe_flow_reduced_order", "cfd"]
FidelityLevel = Literal["analytic", "reduced_order", "numerical", "unsupported", "not_evaluated"]
SolveStatus = Literal["converged", "non_converged", "failed", "unsupported", "not_evaluated"]


def si(q: QuantityModel, dim_unit: str) -> float:
    """Convert to the SI unit ``dim_unit`` after a dimension check (no silent conversions)."""
    check_dim(q.unit, dim_unit)
    return Quantity.of(q.value, q.unit).to(dim_unit)


# ----------------------------------------------------------------------------------------------- materials
class LinearElasticIsotropic(Strict):
    model: Literal["linear_elastic_isotropic"] = "linear_elastic_isotropic"
    youngs_modulus: QuantityModel
    poisson_ratio: Annotated[float, Field(gt=-1.0, lt=0.5)]
    yield_strength: QuantityModel | None = None

    @model_validator(mode="after")
    def _units(self):
        check_dim(self.youngs_modulus.unit, "Pa")
        if self.youngs_modulus.value <= 0:
            raise ValueError("Young's modulus must be > 0")
        if self.yield_strength is not None:
            check_dim(self.yield_strength.unit, "Pa")
        return self


class ThermalIsotropic(Strict):
    model: Literal["thermal_isotropic"] = "thermal_isotropic"
    conductivity: QuantityModel

    @model_validator(mode="after")
    def _units(self):
        check_dim(self.conductivity.unit, "W/(m*K)")
        if self.conductivity.value <= 0:
            raise ValueError("thermal conductivity must be > 0")
        return self


class Material(Strict):
    id: Id
    name: str | None = None
    density: QuantityModel | None = None
    elastic: LinearElasticIsotropic | None = None
    thermal: ThermalIsotropic | None = None
    source: str = Field(description="Where the property values come from (handbook/table, datasheet, 'user')")
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _units(self):
        if self.density is not None:
            check_dim(self.density.unit, "kg/m^3")
        return self


# ----------------------------------------------------------------------------------------------- regions
class Region(Strict):
    """A semantic boundary/volume region, resolved deterministically on ANY mesh.

    ``field`` is an implicit-field expression in geometry millimetres over facet centroid (x, y, z) and unit
    outward normal (nx, ny, nz): a facet belongs to the region iff field <= 0 — the same level-set convention
    as Berkelium implicit geometry, so regions are backend- and mesher-independent. ``tag`` names a set
    provided by the mesh generator (e.g. 'xmin' for structured boxes)."""

    id: Id
    tag: str | None = None
    field: str | None = None
    description: str | None = None

    @model_validator(mode="after")
    def _one(self):
        if (self.tag is None) == (self.field is None):
            raise ValueError("region needs exactly one of 'tag' or 'field'")
        return self


# ------------------------------------------------------------------------------------- loads and conditions
Vec3 = Annotated[list[float], Field(min_length=3, max_length=3)]


class Fixed(Strict):
    kind: Literal["fixed"] = "fixed"
    region: Id
    components: list[Literal["x", "y", "z"]] = Field(default_factory=lambda: ["x", "y", "z"])


class PrescribedDisplacement(Strict):
    kind: Literal["displacement"] = "displacement"
    region: Id
    component: Literal["x", "y", "z"]
    value: QuantityModel


class Traction(Strict):
    """Uniform surface traction (force/area) on a region."""
    kind: Literal["traction"] = "traction"
    region: Id
    vector: Vec3
    magnitude: QuantityModel


class SurfaceForce(Strict):
    """A resultant force spread UNIFORMLY over a region's area (the honest replacement for a point load:
    a true point load has unbounded displacement in 3-D elasticity)."""
    kind: Literal["surface_force"] = "surface_force"
    region: Id
    vector: Vec3
    magnitude: QuantityModel


class Pressure(Strict):
    """Normal pressure, positive pushes INTO the body (along -outward normal)."""
    kind: Literal["pressure"] = "pressure"
    region: Id
    magnitude: QuantityModel


class BodyForce(Strict):
    """Body acceleration (e.g. gravity); requires material density."""
    kind: Literal["body_acceleration"] = "body_acceleration"
    vector: Vec3
    magnitude: QuantityModel


class Temperature(Strict):
    kind: Literal["temperature"] = "temperature"
    region: Id
    value: QuantityModel


class HeatFlux(Strict):
    """Prescribed heat flux INTO the body through a region (W/m^2)."""
    kind: Literal["heat_flux"] = "heat_flux"
    region: Id
    value: QuantityModel


class Convection(Strict):
    kind: Literal["convection"] = "convection"
    region: Id
    h: QuantityModel
    ambient: QuantityModel


class VolumetricHeat(Strict):
    kind: Literal["volumetric_heat"] = "volumetric_heat"
    value: QuantityModel


Condition = Annotated[Fixed | PrescribedDisplacement | Traction | SurfaceForce | Pressure | BodyForce |
                      Temperature | HeatFlux | Convection | VolumetricHeat, Field(discriminator="kind")]

STRUCTURAL_KINDS = {"fixed", "displacement", "traction", "surface_force", "pressure", "body_acceleration"}
THERMAL_KINDS = {"temperature", "heat_flux", "convection", "volumetric_heat"}


# --------------------------------------------------------------------------------- numerics and outputs
class MeshSpec(Strict):
    generator: Literal["structured_box", "gmsh"] = "gmsh"
    element_order: Literal[1, 2] = 2
    size: QuantityModel = Field(description="Characteristic element size on the COARSEST level")
    divisions: Annotated[list[int], Field(min_length=3, max_length=3)] | None = Field(
        None, description="structured_box only: cells along x, y, z on the coarsest level")
    box_mm: Annotated[list[float], Field(min_length=3, max_length=3)] | None = Field(
        None, description="structured_box only: box extents in mm (default divisions * size)")


class ConvergenceSpec(Strict):
    levels: Annotated[int, Field(ge=1, le=4)] = 3
    refinement_ratio: Annotated[float, Field(ge=1.3, le=4.0)] = 2.0
    safety_factor: Annotated[float, Field(ge=1.0, le=3.0)] = 1.25
    max_relative_error: Annotated[float, Field(gt=0, le=1)] = 0.05


class SolverSpec(Strict):
    backend: Literal["skfem"] = "skfem"
    linear_solver: Literal["direct", "cg"] = "direct"
    residual_tolerance: Annotated[float, Field(gt=0, le=1e-3)] = 1e-8


class QoI(Strict):
    """A quantity of interest the analysis must report. Requirements reference it as
    ``sim.<case_id>.<qoi_id>``."""

    id: Id
    kind: Literal["average", "max", "min", "integral", "reaction", "flux"]
    field: Literal["displacement", "von_mises", "temperature", "heat_flux_normal",
                   "pressure_drop", "friction_factor", "reynolds"]
    component: Literal["x", "y", "z", "magnitude"] | None = None
    region: Id | None = Field(None, description="Boundary region; None = whole domain")
    unit: str


class PipeFlowSpec(Strict):
    """Inputs for reduced-order internal pipe flow (Darcy-Weisbach)."""
    diameter: QuantityModel
    length: QuantityModel
    roughness: QuantityModel
    density: QuantityModel
    viscosity: QuantityModel
    flow_rate: QuantityModel
    minor_loss_k: Annotated[float, Field(ge=0)] = 0.0


class AnalysisCase(Strict):
    id: Id
    physics: PhysicsKind
    target: str = Field(description="Component id whose realised geometry is analysed")
    material: Id | None = None
    regions: list[Region] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    outputs: list[QoI] = Field(default_factory=list)
    mesh: MeshSpec | None = None
    solver: SolverSpec = Field(default_factory=SolverSpec)
    convergence: ConvergenceSpec = Field(default_factory=ConvergenceSpec)
    pipe: PipeFlowSpec | None = None
    assumptions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self):
        rid = {r.id for r in self.regions}
        if len(rid) != len(self.regions):
            raise ValueError("duplicate region id")
        allowed = {"structural_linear_static": STRUCTURAL_KINDS, "thermal_steady": THERMAL_KINDS}.get(self.physics)
        for c in self.conditions:
            if allowed is not None and c.kind not in allowed:
                raise ValueError(f"condition {c.kind!r} not valid for {self.physics}")
            if getattr(c, "region", None) is not None and c.region not in rid:
                raise ValueError(f"condition references unknown region {c.region!r}")
        dims = {"displacement": ("value", "m"), "traction": ("magnitude", "Pa"), "surface_force": ("magnitude", "N"),
                "pressure": ("magnitude", "Pa"), "body_acceleration": ("magnitude", "m/s^2"),
                "temperature": ("value", "K"), "heat_flux": ("value", "W/m^2"), "volumetric_heat": ("value", "W/m^3")}
        for c in self.conditions:
            if c.kind in dims:
                check_dim(getattr(c, dims[c.kind][0]).unit, dims[c.kind][1])
            if c.kind == "convection":
                check_dim(c.h.unit, "W/(m^2*K)")
                check_dim(c.ambient.unit, "K")
        for o in self.outputs:
            if o.region is not None and o.region not in rid:
                raise ValueError(f"output {o.id} references unknown region {o.region!r}")
        if self.physics == "pipe_flow_reduced_order" and self.pipe is None:
            raise ValueError("pipe_flow_reduced_order needs 'pipe'")
        return self


# --------------------------------------------------------------------------------------------- results
class Estimate(Strict):
    """A computed quantity with honest error accounting."""

    id: Id
    value: float
    unit: str
    fidelity: FidelityLevel
    status: SolveStatus
    discretization_error: float | None = Field(None, description="Absolute bound (same unit); GCI-based")
    model_form_error: float | None = Field(None, description="Absolute bound from the model's stated accuracy")
    method: str
    notes: list[str] = Field(default_factory=list)

    def interval(self) -> tuple[float, float] | None:
        """[lo, hi] if every error component is known; None if an error component is unknown."""
        if self.status != "converged":
            return None
        if self.fidelity == "numerical" and self.discretization_error is None:
            return None
        b = (self.discretization_error or 0.0) + (self.model_form_error or 0.0)
        return (self.value - b, self.value + b)


class LevelSolution(Strict):
    level: int
    h_mm: float
    n_cells: int
    n_dofs: int
    values: dict[str, float]
    residual: float
    balance: float | None = None
    mesh_sha256: str


class ConvergenceStudy(Strict):
    qoi: Id
    method: Literal["richardson_gci_celik2008", "single_level", "exact_reproduction"]
    refinement_ratios: list[float] = Field(default_factory=list)
    observed_order: float | None = None
    extrapolated: float | None = None
    gci_fine: float | None = Field(None, description="Relative GCI on the finest grid")
    asymptotic_ratio: float | None = Field(None, description="GCI23 / (r^p GCI12); ~1 in the asymptotic range")
    status: SolveStatus
    message: str


class MeshStats(Strict):
    generator: str
    element_type: str
    n_nodes: int
    n_cells: int
    n_boundary_facets: int
    h_mm: float
    min_quality: float
    mean_quality: float
    n_poor: int = Field(description="cells with shape quality < 0.1")
    regions: dict[str, int] = Field(description="facet count per resolved region")
    sha256: str


class SimProvenance(Strict):
    solver: str
    solver_version: str
    mesher: str
    mesher_version: str
    geometry_sha256: str | None = None
    case_sha256: str
    material_sha256: str | None = None
    settings: dict = Field(default_factory=dict)
    references: list[str] = Field(default_factory=list)


class FieldArtifact(Strict):
    name: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    format: Literal["vtu", "json"]
    bytes: int
    level: int


class SimulationResult(Strict):
    schema_version: str = PHYSICS_SCHEMA_VERSION
    case_id: Id
    physics: PhysicsKind
    target: str
    fidelity: FidelityLevel
    status: SolveStatus
    message: str
    estimates: list[Estimate] = Field(default_factory=list)
    convergence: list[ConvergenceStudy] = Field(default_factory=list)
    levels: list[LevelSolution] = Field(default_factory=list)
    mesh: list[MeshStats] = Field(default_factory=list)
    fields: list[FieldArtifact] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    provenance: SimProvenance | None = None
    wall_seconds: float | None = Field(None, exclude=True,
                                       description="Informational only; never serialised, so records stay "
                                                   "byte-reproducible (stage timings live in the job log)")
