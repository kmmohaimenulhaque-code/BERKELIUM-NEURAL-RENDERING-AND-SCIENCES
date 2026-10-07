"""Laws for two unrelated domains, expressed with the SAME primitives — the generality test of ADR-019.

Quantity definitions matter: ``beam.tip_deflection`` is the small-strain linear-elastic response of an
ideally clamped prismatic cantilever under an end load spread over the tip face. Every law below targets that
idealised quantity; real-world deviations (clamp compliance, material scatter) belong to measurements and
parameter uncertainty, not to these laws.
"""

from __future__ import annotations

import math

from ..physics.fluids import COLEBROOK_MODEL_FORM, colebrook
from ..units import Quantity
from .core import Est, Law

GERE = "Gere & Goodno, Mechanics of Materials (cantilever end load)"
COWPER = "Cowper (1966) J. Appl. Mech. 33:335-340 (shear coefficient)"


def _si(p, k, u):
    return p[k].to(u)


def _eb(p) -> Est:
    E, P, L, b, h = _si(p, "E", "Pa"), _si(p, "P", "N"), _si(p, "L", "m"), _si(p, "b", "m"), _si(p, "h", "m")
    return Est(P * L ** 3 / (3 * E * b * h ** 3 / 12) * 1e3, "mm", 0.0, None,
               note="Euler-Bernoulli: no published error bound for this quantity -> cannot decide alone")


def _timo(p) -> Est:
    E, nu, P, L = _si(p, "E", "Pa"), p["nu"].to("1"), _si(p, "P", "N"), _si(p, "L", "m")
    b, h = _si(p, "b", "m"), _si(p, "h", "m")
    G, k = E / (2 * (1 + nu)), 10 * (1 + nu) / (12 + 11 * nu)
    d = P * L ** 3 / (3 * E * b * h ** 3 / 12) + P * L / (k * G * b * h)
    return Est(d * 1e3, "mm", 0.0, None, note="Timoshenko: ignores clamp end effects; bound unknown")


def _fem(p, levels: int = 3) -> Est:
    from ..physics.run import run_case
    from ..physics.schema import AnalysisCase, Material
    L, b, h = p["L"].to("mm"), p["b"].to("mm"), p["h"].to("mm")
    n = max(2, math.ceil(L / h))
    mat = Material(id="m", source="evidence point", elastic={
        "youngs_modulus": {"value": p["E"].to("GPa"), "unit": "GPa"}, "poisson_ratio": p["nu"].to("1")})
    case = AnalysisCase(
        id="beam", physics="structural_linear_static", target="beam", material="m",
        regions=[{"id": "root", "tag": "xmin"}, {"id": "tip", "tag": "xmax"}],
        conditions=[{"kind": "fixed", "region": "root"},
                    {"kind": "surface_force", "region": "tip", "vector": [0, 0, -1],
                     "magnitude": {"value": p["P"].to("N"), "unit": "N"}}],
        outputs=[{"id": "d", "kind": "average", "field": "displacement", "component": "z", "region": "tip",
                  "unit": "mm"}],
        mesh={"generator": "structured_box", "element_order": 2, "size": {"value": h, "unit": "mm"},
              "divisions": [n, 1, 1], "box_mm": [L, b, h]},
        convergence={"levels": levels})
    r = run_case(case, {"m": mat})
    e = r.estimates[0] if r.estimates else None
    if e is None or e.status != "converged" or e.discretization_error is None:
        return Est(e.value if e else float("nan"), "mm", 0.0, 0.0, False, note=f"FEM {r.status}: {r.message}")
    return Est(-e.value, "mm", e.discretization_error, 0.0, True,
               note=f"3-D linear elasticity, P2, GCI over {levels} levels; case {r.provenance.case_sha256[:12]}")


def beam_laws(fem_levels: int = 3) -> list[Law]:
    groups = {"slenderness": "L / h", "aspect": "h / b"}
    lin = ["P * L^2 / (E * b * h^3) <= 0.01"]          # small-deflection regime (delta/L ~ 4x this)
    return [
        Law("euler_bernoulli", "beam.tip_deflection", "mm", "analytic", 1.0, _eb,
            lin + ["slenderness >= 5"], groups, [GERE]),
        Law("timoshenko", "beam.tip_deflection", "mm", "analytic", 1.1, _timo,
            lin + ["slenderness >= 2"], groups, [GERE, COWPER]),
        Law("fem_p2_gci", "beam.tip_deflection", "mm", "numerical", 1000.0, lambda p: _fem(p, fem_levels),
            lin, groups, ["berkelium.physics (scikit-fem P2) + Celik 2008 GCI"]),
    ]


def _pipe_re(p) -> float:
    return p["rho"].to("kg/m^3") * p["V"].to("m/s") * p["D"].to("m") / p["mu"].to("Pa*s")


def _dp(p, f: float) -> float:
    return f * p["L"].to("m") / p["D"].to("m") * 0.5 * p["rho"].to("kg/m^3") * p["V"].to("m/s") ** 2


def pipe_laws() -> list[Law]:
    groups = {"Re": "rho * V * D / mu", "rr": "eps / D"}
    lam = Law("hagen_poiseuille", "pipe.pressure_drop", "Pa", "analytic", 1.0,
              lambda p: Est(_dp(p, 64 / _pipe_re(p)), "Pa", 0.0, 0.0, note="exact for fully developed laminar"),
              ["Re < 2300"], groups, ["Hagen-Poiseuille"])

    def turb(p):
        f, res, _ = colebrook(_pipe_re(p), p["eps"].to("m") / p["D"].to("m"))
        d = _dp(p, f)
        return Est(d, "Pa", 0.0, COLEBROOK_MODEL_FORM * d, note=f"Colebrook residual {res:.1e}")
    cole = Law("colebrook", "pipe.pressure_drop", "Pa", "reduced_order", 2.0, turb, ["Re > 4000", "rr <= 0.05"],
               groups, ["Colebrook 1939; +-15 % per White, Fluid Mechanics"])
    return [lam, cole]


def point(**kw: tuple[float, str]) -> dict[str, Quantity]:
    return {k: Quantity.of(v, u) for k, (v, u) in kw.items()}
