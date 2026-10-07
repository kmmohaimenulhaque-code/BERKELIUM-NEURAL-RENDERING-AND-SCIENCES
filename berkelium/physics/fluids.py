"""Fluids, with fidelity stated honestly (PHYSICS-5).

Level A — reduced_order: fully developed incompressible pipe flow, Darcy-Weisbach.
  laminar  (Re < 2300): f = 64/Re — exact for Hagen-Poiseuille flow (no model-form band).
  turbulent (Re > 4000): Colebrook (1939) implicit equation solved to a stated residual. Colebrook/Moody
    friction factors carry roughly +-15 % uncertainty against experiment (White, Fluid Mechanics) — carried
    as model_form_error, not hidden.
  transitional (2300..4000): no reliable correlation — status 'non_converged' with both bounds as a band.
Level B — numerical CFD: ``CFDAdapter`` protocol. Level C — OpenFOAM out-of-process adapter: detected,
  never emulated. Without a real CFD backend the result is 'unsupported'. Nothing here is labelled CFD.
"""

from __future__ import annotations

import math
import shutil
from typing import Protocol

from .schema import AnalysisCase, Estimate, SimProvenance, SimulationResult, si

REFS = ["Darcy-Weisbach: White, Fluid Mechanics, ch. 6",
        "Colebrook, C.F. (1939) J. Inst. Civil Eng. 11:133-156",
        "Laminar f=64/Re: Hagen-Poiseuille exact solution"]
COLEBROOK_MODEL_FORM = 0.15


def colebrook(re: float, rel_rough: float, tol: float = 1e-14) -> tuple[float, float, int]:
    """Solve 1/sqrt(f) = -2 log10(eps/3.7D + 2.51/(Re sqrt f)) by fixed point on x = 1/sqrt(f).
    Returns (f, |residual|, iterations)."""
    x = -2.0 * math.log10(rel_rough / 3.7 + 2.51 / re * 8.0)  # f=1/64-ish start; converges globally here
    i = 0
    for i in range(1, 200):  # noqa: B007
        xn = -2.0 * math.log10(rel_rough / 3.7 + 2.51 * x / re)
        if abs(xn - x) < tol * abs(xn):
            x = xn
            break
        x = xn
    f = 1.0 / x ** 2
    resid = abs(1 / math.sqrt(f) + 2.0 * math.log10(rel_rough / 3.7 + 2.51 / (re * math.sqrt(f))))
    return f, resid, i


def pipe_flow(case: AnalysisCase) -> SimulationResult:
    p = case.pipe
    D, L, eps = si(p.diameter, "m"), si(p.length, "m"), si(p.roughness, "m")
    rho, mu, Qv = si(p.density, "kg/m^3"), si(p.viscosity, "Pa*s"), si(p.flow_rate, "m^3/s")
    if min(D, L, rho, mu, Qv) <= 0 or eps < 0:
        return SimulationResult(case_id=case.id, physics=case.physics, target=case.target, fidelity="not_evaluated",
                                status="failed", message="non-physical pipe inputs (must be positive)")
    A = math.pi * D ** 2 / 4
    V = Qv / A
    re = rho * V * D / mu
    dyn = 0.5 * rho * V ** 2
    warns, notes = [], []
    if re < 2300:
        f, mf, st, how = 64.0 / re, 0.0, "converged", "laminar exact (64/Re)"
    elif re > 4000:
        f, resid, its = colebrook(re, eps / D)
        mf, st, how = COLEBROOK_MODEL_FORM * f, "converged", f"Colebrook, {its} iterations, residual {resid:.1e}"
    else:
        fl = 64.0 / re
        ft, _, _ = colebrook(4000.0, eps / D)
        f, mf, st = 0.5 * (fl + ft), 0.5 * abs(ft - fl), "non_converged"
        how = "transitional regime: no reliable correlation; band spans laminar..turbulent limits"
        warns.append(how)
    dp = (f * L / D + p.minor_loss_k) * dyn
    dp_mf = mf * L / D * dyn
    notes.append(f"Re={re:.6g}")
    est = [
        Estimate(id="reynolds", value=re, unit="1", fidelity="reduced_order", status="converged",
                 method="Re = rho V D / mu"),
        Estimate(id="friction_factor", value=f, unit="1", fidelity="reduced_order", status=st,
                 model_form_error=mf, method=how),
        Estimate(id="pressure_drop", value=dp, unit="Pa", fidelity="reduced_order", status=st, model_form_error=dp_mf,
                 method="dp = (f L/D + K) rho V^2 / 2", notes=notes +
                 (["minor-loss K is user-supplied; its uncertainty is not included"] if p.minor_loss_k else [])),
    ]
    ids = {o.id: o for o in case.outputs}
    for o in case.outputs:   # expose under requested QoI ids with requested units
        src = next(e for e in est if e.id == o.field)
        from ..units import Quantity
        k = Quantity.of(1.0, src.unit).to(o.unit)
        est.append(src.model_copy(update={"id": o.id, "value": src.value * k, "unit": o.unit,
                                          "model_form_error": (src.model_form_error or 0.0) * k}))
    del ids
    from ..schema.hashing import sha256_of
    return SimulationResult(case_id=case.id, physics=case.physics, target=case.target, fidelity="reduced_order",
                            status=st, message=f"reduced-order pipe flow ({how}); NOT CFD", estimates=est,
                            assumptions=["incompressible, fully developed, isothermal, circular pipe",
                                         *case.assumptions], warnings=warns,
                            provenance=SimProvenance(solver="berkelium.darcy_weisbach", solver_version="0.1",
                                                     mesher="none", mesher_version="-",
                                                     case_sha256=sha256_of(case.model_dump(mode="json")),
                                                     references=REFS))


class CFDAdapter(Protocol):
    name: str

    def available(self) -> bool: ...

    def run(self, case: AnalysisCase, step: bytes) -> SimulationResult: ...


class OpenFOAMAdapter:
    """Out-of-process OpenFOAM (GPL-3.0) adapter. Detection only in this milestone: case writing,
    snappyHexMesh and solver execution are not implemented, so it never returns numerical results."""

    name = "openfoam"

    def available(self) -> bool:
        return shutil.which("simpleFoam") is not None and shutil.which("blockMesh") is not None

    def run(self, case: AnalysisCase, step: bytes) -> SimulationResult:
        why = "OpenFOAM not installed" if not self.available() else "OpenFOAM adapter not implemented yet"
        return SimulationResult(case_id=case.id, physics=case.physics, target=case.target, fidelity="unsupported",
                                status="unsupported", message=f"CFD not run: {why}")
