"""Cantilever beam CEM: rectangular section sized from load, stiffness and strength requirements.

Analytic model (Euler-Bernoulli, end load P on a cantilever of length L — Gere & Goodno, Mechanics of
Materials, cantilever deflection table; flexure formula sigma = M c / I):
    delta = P L^3 / (3 E I),   sigma_max = 6 P L / (b h^2),   I = b h^3 / 12,   b = h / a
Sizing:  h >= (4 P L^3 a / (E delta_max))^(1/4)   and   h >= (6 P L a / sigma_allow)^(1/3)
Validity: Euler-Bernoulli neglects shear deformation; flagged when L/h < 10. The numerical L7 analysis
(berkelium.physics) is the independent check — ``default_analysis`` builds that case, the CEM never
asserts its result.
"""

from __future__ import annotations

import math
from typing import Annotated

import numpy as np
from pydantic import Field

from ...._base import Strict
from ....geometry.ir import GeometryGraph
from ....physics.schema import AnalysisCase
from ....schema.common import QuantityModel
from ....schema.evaluation import Derivation, Fidelity, ValidationResult
from ....units import Quantity
from ...protocol import CEM, CEMMeta, ComponentGeometry, DeriveResult

REF = "Gere & Goodno, Mechanics of Materials: cantilever end-load deflection; flexure formula"
Pos = Annotated[float, Field(gt=0)]


class Requirements(Strict):
    tip_load_N: Pos
    length_mm: Pos
    max_tip_deflection_mm: Pos
    allowable_stress_MPa: Pos
    youngs_modulus_GPa: Pos
    aspect_ratio: Annotated[float, Field(ge=0.5, le=10)] = Field(2.0, description="h / b")


class Parameters(Strict):
    length_mm: Pos
    width_mm: Pos
    height_mm: Pos
    tip_load_N: Pos
    youngs_modulus_GPa: Pos


def _eb(p: Parameters) -> tuple[float, float]:
    E, P, L = p.youngs_modulus_GPa * 1e9, p.tip_load_N, p.length_mm * 1e-3
    b, h = p.width_mm * 1e-3, p.height_mm * 1e-3
    inertia = b * h ** 3 / 12
    return P * L ** 3 / (3 * E * inertia) * 1e3, 6 * P * L / (b * h * h) / 1e6   # mm, MPa


class CantileverBeamCEM(CEM):
    meta = CEMMeta(name="structure.cantilever_beam", version="0.1.0", domain="structures",
                   summary="Rectangular cantilever sized for an end load: stiffness + strength (Euler-Bernoulli), "
                           "with a numerical 3-D elasticity check available as an L7 analysis",
                   kind="cem", references=(REF,), tags=("structure", "beam"))
    Requirements = Requirements
    Parameters = Parameters

    def derive(self, req: Requirements) -> DeriveResult:  # type: ignore[override]
        P, L, E, a = req.tip_load_N, req.length_mm * 1e-3, req.youngs_modulus_GPa * 1e9, req.aspect_ratio
        h_def = (4 * P * L ** 3 * a / (E * req.max_tip_deflection_mm * 1e-3)) ** 0.25 * 1e3
        h_str = (6 * P * L * a / (req.allowable_stress_MPa * 1e6)) ** (1 / 3) * 1e3
        h = math.ceil(max(h_def, h_str) * 10) / 10
        p = Parameters(length_mm=req.length_mm, width_mm=round(h / a, 3), height_mm=h, tip_load_N=P,
                       youngs_modulus_GPa=req.youngs_modulus_GPa)
        d = [Derivation(id="h_for_deflection", target="beam", formula="(4 P L^3 a / (E delta_max))^(1/4)",
                        value=QuantityModel(value=h_def, unit="mm"), references=[REF]),
             Derivation(id="h_for_stress", target="beam", formula="(6 P L a / sigma_allow)^(1/3)",
                        value=QuantityModel(value=h_str, unit="mm"), references=[REF])]
        gov = "deflection" if h_def >= h_str else "stress"
        return DeriveResult(p, d, [f"section height governed by {gov}; rounded up to 0.1 mm",
                                   "Euler-Bernoulli beam: plane sections, shear deformation neglected"])

    def check(self, p: Parameters) -> list[ValidationResult]:  # type: ignore[override]
        delta, sigma = _eb(p)
        fid = Fidelity(kind="analytic", method="euler_bernoulli_cantilever", references=[REF],
                       assumptions=["linear elastic, small deflection, end load, rigid clamp"])
        slender = p.length_mm / p.height_mm
        return [ValidationResult(validator="beam.slenderness@0.1", level=6, status="pass" if slender >= 10 else "warn",
                                 target="beam", message=f"L/h = {slender:.2f} (Euler-Bernoulli validity wants >= 10)",
                                 measured=QuantityModel(value=slender, unit="1"), fidelity=fid)] + \
            [ValidationResult(validator="beam.eb@0.1", level=6, status="pass", target="beam",
                              message=f"analytic tip deflection {delta:.4g} mm, root stress {sigma:.4g} MPa",
                              measured=QuantityModel(value=delta, unit="mm"), fidelity=fid)]

    def quantities(self, p: Parameters) -> dict[str, Quantity]:  # type: ignore[override]
        delta, sigma = _eb(p)
        return {"tip_deflection_eb": Quantity.of(delta, "mm"), "root_stress_eb": Quantity.of(sigma, "MPa"),
                "height": Quantity.of(p.height_mm, "mm"), "width": Quantity.of(p.width_mm, "mm")}

    def expand(self, p: Parameters) -> ComponentGeometry:  # type: ignore[override]
        from ...protocol import BodySpec
        g = GeometryGraph(ops=[{"op": "box", "id": "beam", "size": [p.length_mm, p.width_mm, p.height_mm],
                                "tags": ["root_face:x=0", "tip_face:x=L"]}], outputs={"beam": "beam"})
        return ComponentGeometry(graph=g, bodies={"beam": BodySpec("beam", ["structure"])})

    def analyses(self) -> dict[str, str]:
        return {"tip_deflection_euler_bernoulli": "analytic", "tip_deflection_3d_elasticity": "solver_required"}

    def sample(self, rng: np.random.Generator, mode: str) -> Requirements:
        L = float(rng.choice([80, 100, 150, 200, 300]))
        P = float(rng.choice([50, 100, 200, 500, 1000]))
        r = Requirements(tip_load_N=P, length_mm=L, max_tip_deflection_mm=round(L / float(rng.choice([200, 300, 500])), 3),
                         allowable_stress_MPa=float(rng.choice([100, 150, 200])), youngs_modulus_GPa=207.0,
                         aspect_ratio=float(rng.choice([1.0, 1.5, 2.0, 3.0])))
        return r


def default_analysis(component: str, p: Parameters, material: str, size_mm: float | None = None,
                     levels: int = 3, ratio: float = 1.5) -> AnalysisCase:
    """The 3-D linear-elastic check of the beam: clamped root, end load spread over the tip face."""
    L = p.length_mm
    s = size_mm or min(p.width_mm, p.height_mm) * 0.7
    return AnalysisCase(
        id=f"{component}_elastic", physics="structural_linear_static", target=component, material=material,
        regions=[{"id": "root", "field": "x - 0.001", "description": "clamped face x = 0"},
                 {"id": "tip", "field": f"{L - 0.001!r} - x", "description": "loaded face x = L"}],
        conditions=[{"kind": "fixed", "region": "root"},
                    {"kind": "surface_force", "region": "tip", "vector": [0, 0, -1],
                     "magnitude": {"value": p.tip_load_N, "unit": "N"}}],
        outputs=[{"id": "tip_deflection", "kind": "average", "field": "displacement", "component": "z",
                  "region": "tip", "unit": "mm"},
                 {"id": "root_reaction", "kind": "reaction", "field": "displacement", "component": "z",
                  "region": "root", "unit": "N"}],
        mesh={"generator": "gmsh", "element_order": 2, "size": {"value": s, "unit": "mm"}},
        convergence={"levels": levels, "refinement_ratio": ratio},
        assumptions=["end load applied as uniform traction over the tip face (Saint-Venant)",
                     "root fully clamped (all displacement components zero)"])
