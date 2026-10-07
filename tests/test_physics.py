"""PHYSICS-1..6: schemas, analytic oracles, numerical verification with measured errors, failure paths.

Reference solutions (exact unless stated):
  axial bar        u = sigma L / E                         (P1 reproduces exactly)
  heated slab      T = T0 + (T1-T0)x/L + Q x (L-x)/(2k)     (P2 reproduces exactly)
  convective slab  q = (T0 - Tinf) / (L/k + 1/h)            (P1 reproduces exactly)
  radial conduction Q = 2 pi k H (Ti - To) / ln(ro/ri)      (curved: discretisation error, gmsh path)
  Lame cylinder    plane strain u_r(a), sigma_vm(a)         (curved: discretisation error, gmsh path)
  cantilever       Timoshenko beam (Cowper kappa)            (MODEL reference: 3-D clamp differs slightly)
  pipe             laminar Hagen-Poiseuille; Colebrook vs Haaland (1983)
"""
import math

import pytest
from pydantic import ValidationError

from berkelium.physics.fluids import colebrook
from berkelium.physics.mesh import gmsh_version
from berkelium.physics.run import run_case
from berkelium.physics.schema import AnalysisCase, Material
from berkelium.physics.verification import study
from berkelium.schema.hashing import sha256_of

E, NU, K = 200e9, 0.3, 50.0
MAT = Material(id="m", source="test values", elastic={"youngs_modulus": {"value": 200, "unit": "GPa"},
               "poisson_ratio": NU}, thermal={"conductivity": {"value": K, "unit": "W/(m*K)"}})
BOX = {"generator": "structured_box", "size": {"value": 10, "unit": "mm"}, "divisions": [10, 1, 1],
       "box_mm": [100, 10, 10]}
needs_gmsh = pytest.mark.skipif(gmsh_version() is None, reason="gmsh executable not installed")


def case(**kw):
    return AnalysisCase(**{"id": "c", "target": "t", "material": "m", **kw})


def bar_case(levels=2):
    return case(physics="structural_linear_static",
                regions=[{"id": "x0", "tag": "xmin"}, {"id": "xL", "tag": "xmax"}, {"id": "y0", "tag": "ymin"},
                         {"id": "z0", "tag": "zmin"}],
                conditions=[{"kind": "fixed", "region": "x0", "components": ["x"]},
                            {"kind": "fixed", "region": "y0", "components": ["y"]},
                            {"kind": "fixed", "region": "z0", "components": ["z"]},
                            {"kind": "traction", "region": "xL", "vector": [1, 0, 0],
                             "magnitude": {"value": 100, "unit": "MPa"}}],
                outputs=[{"id": "tip", "kind": "average", "field": "displacement", "component": "x", "region": "xL",
                          "unit": "mm"}, {"id": "vm", "kind": "max", "field": "von_mises", "unit": "MPa"},
                         {"id": "rx", "kind": "reaction", "field": "displacement", "component": "x",
                          "region": "x0", "unit": "N"}],
                mesh={**BOX, "element_order": 1}, convergence={"levels": levels})


# ------------------------------------------------------------------------------------------ schemas
def test_schema_rejects_inconsistent_cases():
    with pytest.raises(ValidationError, match="not valid for"):
        case(physics="structural_linear_static", regions=[{"id": "a", "tag": "xmin"}],
             conditions=[{"kind": "temperature", "region": "a", "value": {"value": 1, "unit": "K"}}])
    with pytest.raises(ValidationError, match="unknown region"):
        case(physics="thermal_steady", conditions=[{"kind": "temperature", "region": "a",
                                                    "value": {"value": 1, "unit": "K"}}])
    with pytest.raises(ValidationError):   # pressure given in metres
        case(physics="structural_linear_static", regions=[{"id": "a", "tag": "xmin"}],
             conditions=[{"kind": "pressure", "region": "a", "magnitude": {"value": 1, "unit": "m"}}])
    with pytest.raises(ValidationError):   # Young's modulus not a pressure
        Material(id="x", source="t", elastic={"youngs_modulus": {"value": 1, "unit": "mm"}, "poisson_ratio": 0.3})
    with pytest.raises(ValidationError):   # nu >= 0.5
        Material(id="x", source="t", elastic={"youngs_modulus": {"value": 1, "unit": "GPa"}, "poisson_ratio": 0.5})
    with pytest.raises(ValidationError, match="exactly one"):
        case(physics="thermal_steady", regions=[{"id": "a"}])


# ---------------------------------------------------------------------------------- verification core
def test_gci_recovers_order_and_limit():
    s = study("q", [1 + 0.001 * h ** 2 for h in (4, 2, 1)], [4, 2, 1], 1.25, 0.05, 2.0)
    assert s.status == "converged" and abs(s.observed_order - 2) < 1e-9 and abs(s.extrapolated - 1) < 1e-12
    osc = study("q", [1.0, 1.1, 1.05], [4, 2, 1], 1.25, 0.05, 2.0)
    assert osc.status == "non_converged"
    sing = study("q", [1 + h ** 0.3 for h in (4, 2, 1)], [4, 2, 1], 1.25, 0.5, 2.0)
    assert sing.status == "non_converged" and "<<" in sing.message


# ------------------------------------------------------------------------------- exact reproductions
def test_axial_bar_exact():
    r = run_case(bar_case(), {"m": MAT})
    v = {e.id: e for e in r.estimates}
    assert r.status == "converged" and r.fidelity == "numerical"
    assert abs(v["tip"].value - 100e6 * 0.1 / E * 1e3) < 1e-12
    assert abs(v["vm"].value - 100) < 1e-9 and abs(v["rx"].value + 100e6 * 1e-4) < 1e-6
    assert all(lv.residual < 1e-10 and lv.balance < 1e-10 for lv in r.levels)
    assert r.provenance.case_sha256 and r.provenance.material_sha256 and r.mesh[0].sha256


def test_heated_slab_p2_exact():
    T0, T1, Q, L = 300.0, 400.0, 1e6, 0.1
    c = case(physics="thermal_steady", regions=[{"id": "a", "field": "x - 0.0001"},
                                               {"id": "b", "field": "99.9999 - x"}],
             conditions=[{"kind": "temperature", "region": "a", "value": {"value": T0, "unit": "K"}},
                         {"kind": "temperature", "region": "b", "value": {"value": T1, "unit": "K"}},
                         {"kind": "volumetric_heat", "value": {"value": Q, "unit": "W/m^3"}}],
             outputs=[{"id": "qa", "kind": "flux", "field": "heat_flux_normal", "region": "a", "unit": "W"},
                      {"id": "qb", "kind": "flux", "field": "heat_flux_normal", "region": "b", "unit": "W"}],
             mesh={**BOX, "element_order": 2}, convergence={"levels": 2})
    r = run_case(c, {"m": MAT})
    v = {e.id: e.value for e in r.estimates}
    A = 1e-4
    assert r.status == "converged"
    assert abs(v["qa"] - K * ((T1 - T0) / L + Q * L / (2 * K)) * A) < 1e-9      # 10 W out at x=0
    assert abs(v["qb"] - (-K * ((T1 - T0) / L - Q * L / (2 * K)) * A)) < 1e-9   # 0 W at x=L
    assert all(lv.balance < 1e-10 for lv in r.levels)


def test_convective_slab_exact():
    h, Tinf, T0, L = 25.0, 293.15, 373.15, 0.1
    c = case(physics="thermal_steady", regions=[{"id": "a", "tag": "xmin"}, {"id": "b", "tag": "xmax"}],
             conditions=[{"kind": "temperature", "region": "a", "value": {"value": T0, "unit": "K"}},
                         {"kind": "convection", "region": "b", "h": {"value": h, "unit": "W/(m^2*K)"},
                          "ambient": {"value": Tinf, "unit": "K"}}],
             outputs=[{"id": "q", "kind": "flux", "field": "heat_flux_normal", "region": "b", "unit": "W"},
                      {"id": "Ts", "kind": "average", "field": "temperature", "region": "b", "unit": "K"}],
             mesh={**BOX, "element_order": 1}, convergence={"levels": 2})
    r = run_case(c, {"m": MAT})
    v = {e.id: e.value for e in r.estimates}
    q = (T0 - Tinf) / (L / K + 1 / h)
    assert abs(v["q"] - q * 1e-4) < 1e-10 and abs(v["Ts"] - (Tinf + q / h)) < 1e-9


# --------------------------------------------------------------------- discretisation error, measured
def test_cantilever_converges_near_timoshenko():
    F, Lm, b = 100.0, 0.1, 0.01
    c = case(physics="structural_linear_static",
             regions=[{"id": "root", "tag": "xmin"}, {"id": "tip", "tag": "xmax"}],
             conditions=[{"kind": "fixed", "region": "root"},
                         {"kind": "surface_force", "region": "tip", "vector": [0, 0, -1],
                          "magnitude": {"value": F, "unit": "N"}}],
             outputs=[{"id": "d", "kind": "average", "field": "displacement", "component": "z", "region": "tip",
                       "unit": "mm"}, {"id": "R", "kind": "reaction", "field": "displacement", "component": "z",
                                       "region": "root", "unit": "N"}],
             mesh={**BOX, "element_order": 2}, convergence={"levels": 3})
    r = run_case(c, {"m": MAT})
    est = {e.id: e for e in r.estimates}
    inertia, A, G = b ** 4 / 12, b * b, E / (2 * (1 + NU))
    kappa = 10 * (1 + NU) / (12 + 11 * NU)                      # Cowper (1966), rectangle
    timo = (F * Lm ** 3 / (3 * E * inertia) + F * Lm / (kappa * G * A)) * 1e3
    d = est["d"]
    assert r.status == "converged" and d.status == "converged" and d.discretization_error is not None
    assert abs(-d.value - timo) / timo < 0.02                   # model reference, not exact
    assert abs(est["R"].value - F) < 1e-6                       # global equilibrium
    assert r.convergence[0].observed_order > 1.0


@needs_gmsh
def test_radial_conduction_gmsh_error_within_band():
    from tests._geom import tube_step
    ri, ro, H, Ti, To = 10.0, 20.0, 10.0, 400.0, 300.0
    c = case(physics="thermal_steady",
             regions=[{"id": "in", "field": "sqrt(x^2+y^2) - 10.01"}, {"id": "out", "field": "19.99 - sqrt(x^2+y^2)"}],
             conditions=[{"kind": "temperature", "region": "in", "value": {"value": Ti, "unit": "K"}},
                         {"kind": "temperature", "region": "out", "value": {"value": To, "unit": "K"}}],
             outputs=[{"id": "Q", "kind": "flux", "field": "heat_flux_normal", "region": "out", "unit": "W"}],
             mesh={"generator": "gmsh", "element_order": 2, "size": {"value": 4, "unit": "mm"}},
             convergence={"levels": 3})
    r = run_case(c, {"m": MAT}, step=tube_step(ri, ro, H))
    exact = 2 * math.pi * K * H * 1e-3 * (Ti - To) / math.log(ro / ri)
    e = r.estimates[0]
    assert r.status == "converged" and 1.5 < r.convergence[0].observed_order < 2.5
    assert abs(e.value - exact) <= e.discretization_error        # true error inside the GCI band
    assert abs(e.value - exact) / exact < 2e-3


@needs_gmsh
def test_lame_cylinder_gmsh_error_within_band():
    from tests._geom import quarter_tube_step
    a, b, p = 0.01, 0.02, 10e6
    ur = (1 + NU) * p * a ** 2 / (E * (b ** 2 - a ** 2)) * ((1 - 2 * NU) * a + b ** 2 / a) * 1e3
    sr, st = -p, p * (b ** 2 + a ** 2) / (b ** 2 - a ** 2)
    sz = NU * (sr + st)
    vm = math.sqrt(0.5 * ((sr - st) ** 2 + (st - sz) ** 2 + (sz - sr) ** 2)) / 1e6
    c = case(physics="structural_linear_static",
             regions=[{"id": "in", "field": "sqrt(x^2+y^2) - 10.01"}, {"id": "sx", "field": "abs(x) - 0.001"},
                      {"id": "sy", "field": "abs(y) - 0.001"}, {"id": "z0", "field": "abs(z) - 0.001"},
                      {"id": "zH", "field": "abs(z - 10) - 0.001"}],
             conditions=[{"kind": "fixed", "region": "sx", "components": ["x"]},
                         {"kind": "fixed", "region": "sy", "components": ["y"]},
                         {"kind": "fixed", "region": "z0", "components": ["z"]},
                         {"kind": "fixed", "region": "zH", "components": ["z"]},
                         {"kind": "pressure", "region": "in", "magnitude": {"value": 10, "unit": "MPa"}}],
             outputs=[{"id": "ur", "kind": "average", "field": "displacement", "component": "magnitude",
                       "region": "in", "unit": "mm"},
                      {"id": "vmi", "kind": "average", "field": "von_mises", "region": "in", "unit": "MPa"}],
             mesh={"generator": "gmsh", "element_order": 2, "size": {"value": 2.5, "unit": "mm"}},
             convergence={"levels": 3, "refinement_ratio": 1.5})
    r = run_case(c, {"m": MAT}, step=quarter_tube_step(10, 20, 10))
    est = {e.id: e for e in r.estimates}
    assert r.status == "converged"
    assert abs(est["ur"].value - ur) <= est["ur"].discretization_error
    assert abs(est["vmi"].value - vm) <= est["vmi"].discretization_error


# ------------------------------------------------------------------------------------ reduced order
def test_pipe_laminar_matches_hagen_poiseuille_and_is_not_cfd():
    rho, mu, D, L = 998.0, 1.002e-3, 0.05, 10.0
    Q = 1500 * mu / (rho * D) * math.pi * D ** 2 / 4
    c = case(physics="pipe_flow_reduced_order", material=None,
             pipe={"diameter": {"value": D, "unit": "m"}, "length": {"value": L, "unit": "m"},
                   "roughness": {"value": 0, "unit": "m"}, "density": {"value": rho, "unit": "kg/m^3"},
                   "viscosity": {"value": mu, "unit": "Pa*s"}, "flow_rate": {"value": Q, "unit": "m^3/s"}})
    r = run_case(c, {})
    dp = next(e for e in r.estimates if e.id == "pressure_drop")
    assert r.fidelity == "reduced_order" and "NOT CFD" in r.message
    assert abs(dp.value - 128 * mu * L * Q / (math.pi * D ** 4)) / dp.value < 1e-12


def test_colebrook_solution_and_transitional_honesty():
    for re, rr in ((1e5, 0.0), (1e6, 1e-4), (5e4, 1e-3)):
        f, resid, _ = colebrook(re, rr)
        haaland = (-1.8 * math.log10((rr / 3.7) ** 1.11 + 6.9 / re)) ** -2
        assert resid < 1e-12 and abs(f - haaland) / f < 0.03
    c = case(physics="pipe_flow_reduced_order", material=None,
             pipe={"diameter": {"value": 0.05, "unit": "m"}, "length": {"value": 1, "unit": "m"},
                   "roughness": {"value": 0, "unit": "m"}, "density": {"value": 1000, "unit": "kg/m^3"},
                   "viscosity": {"value": 1e-3, "unit": "Pa*s"},
                   "flow_rate": {"value": 3000 * 1e-3 / (1000 * 0.05) * math.pi * 0.05 ** 2 / 4, "unit": "m^3/s"}})
    r = run_case(c, {})
    assert r.status == "non_converged" and r.warnings


def test_cfd_is_unsupported_not_faked():
    c = case(physics="cfd", material=None)
    r = run_case(c, {})
    assert r.status == "unsupported" and r.fidelity == "unsupported" and not r.estimates


# ------------------------------------------------------------------------------------- failure paths
def test_failure_paths_are_explicit():
    unconstrained = bar_case().model_copy(update={"conditions": bar_case().conditions[3:]})
    r = run_case(unconstrained, {"m": MAT})
    assert r.status == "failed" and "rigid-body" in r.message
    r = run_case(bar_case(), {})
    assert r.status == "failed" and "material" in r.message
    bad = bar_case().model_copy(update={"regions": [*bar_case().regions[:3], {"id": "z0", "field": "z + 5"}]})
    r = run_case(AnalysisCase.model_validate(bad.model_dump()), {"m": MAT})
    assert r.status == "failed" and "selects no boundary facets" in r.message
    nothermal = MAT.model_copy(update={"thermal": None})
    c = case(physics="thermal_steady", regions=[{"id": "a", "tag": "xmin"}],
             conditions=[{"kind": "temperature", "region": "a", "value": {"value": 300, "unit": "K"}}],
             mesh={**BOX, "element_order": 1}, convergence={"levels": 1})
    assert run_case(c, {"m": nothermal}).status == "failed"


def test_results_are_deterministic():
    a = run_case(bar_case(), {"m": MAT}).model_dump(mode="json", exclude={"wall_seconds"})
    b = run_case(bar_case(), {"m": MAT}).model_dump(mode="json", exclude={"wall_seconds"})
    assert sha256_of(a) == sha256_of(b)
