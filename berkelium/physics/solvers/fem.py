"""Finite-element solvers on scikit-fem (BSD-3-Clause, in-process; ADR-016).

* Linear static elasticity, isotropic, small strain: find u with
      int sigma(u):eps(v) dx = int rho*a.v dx + int t.v ds        (Lagrange P1/P2 tetrahedra)
* Steady heat conduction: find T with
      int k grad T.grad v dx + int_conv h T v ds = int Q v dx + int q v ds + int_conv h T_inf v ds

Every solve reports its algebraic residual and a discrete balance (equilibrium / heat balance) so a broken
assembly or solve is caught, and every quantity of interest is computed by quadrature on the solution —
never by a formula standing in for the field.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..mesh import MeshBundle
from ..schema import AnalysisCase, Material, QoI, si

AX = {"x": 0, "y": 1, "z": 2}


class SolveError(RuntimeError):
    pass


@dataclass
class LevelOutput:
    values: dict[str, float]
    residual: float
    balance: float | None
    n_dofs: int
    point_data: dict[str, np.ndarray] = field(default_factory=dict)
    cell_data: dict[str, np.ndarray] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _unit_vec(v) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = np.linalg.norm(a)
    if n == 0:
        raise SolveError("zero direction vector")
    return a / n


def _solve(K, f, D, x, linear_solver: str, tol: float):
    import scipy.sparse.linalg as sla
    import skfem as fem
    A, b, _xI, free = fem.condense(K, f, x=x, D=D)
    if b.size == 0:
        raise SolveError("no free degrees of freedom")
    if linear_solver == "direct":
        y = sla.spsolve(A.tocsc(), b)
    else:
        y, info = sla.cg(A, b, rtol=tol * 1e-2, maxiter=20 * b.size)
        if info != 0:
            raise SolveError(f"CG did not converge (info={info})")
    u = x.copy()
    u[free] = y
    res = float(np.linalg.norm(A @ y - b) / max(np.linalg.norm(b), 1e-300))
    if not np.all(np.isfinite(u)):
        raise SolveError("non-finite solution (singular system: missing constraints?)")
    return u, res


def _area(m, facets) -> float:
    import skfem as fem
    fb = fem.FacetBasis(m, fem.ElementTetP1(), facets=facets)
    return float(fem.Functional(lambda w: 1.0 + 0.0 * w.x[0]).assemble(fb))


def _von_mises(G, lam, mu):
    eps = 0.5 * (G + G.transpose(1, 0, *range(2, G.ndim)))
    tr = eps[0, 0] + eps[1, 1] + eps[2, 2]
    s = 2 * mu * eps
    for i in range(3):
        s[i, i] = s[i, i] + lam * tr
    return np.sqrt(0.5 * ((s[0, 0] - s[1, 1]) ** 2 + (s[1, 1] - s[2, 2]) ** 2 + (s[2, 2] - s[0, 0]) ** 2)
                   + 3 * (s[0, 1] ** 2 + s[1, 2] ** 2 + s[2, 0] ** 2))


def elastic(b: MeshBundle, case: AnalysisCase, mat: Material, regions: dict[str, np.ndarray],
            order: int) -> LevelOutput:
    import skfem as fem
    from skfem.models.elasticity import lame_parameters, linear_elasticity
    if mat.elastic is None:
        raise SolveError(f"material {mat.id} has no elastic model")
    E = si(mat.elastic.youngs_modulus, "Pa")
    nu = mat.elastic.poisson_ratio
    lam, mu = lame_parameters(E, nu)
    m = b.mesh
    e = fem.ElementVector(fem.ElementTetP2() if order == 2 else fem.ElementTetP1())
    ib = fem.Basis(m, e)
    K = linear_elasticity(lam, mu).assemble(ib)
    f = np.zeros(ib.N)
    x = np.zeros(ib.N)
    D: list[np.ndarray] = []
    warns: list[str] = []
    applied = np.zeros(3)
    for c in case.conditions:
        if c.kind == "fixed":
            for comp in c.components:
                D.append(ib.get_dofs(regions[c.region]).all(f"u^{AX[comp] + 1}"))
        elif c.kind == "displacement":
            d = ib.get_dofs(regions[c.region]).all(f"u^{AX[c.component] + 1}")
            x[d] = si(c.value, "m")
            D.append(d)
        elif c.kind in ("traction", "surface_force"):
            fac = regions[c.region]
            mag = si(c.magnitude, "Pa") if c.kind == "traction" else si(c.magnitude, "N") / _area(m, fac)
            tv = mag * _unit_vec(c.vector)
            fb = fem.FacetBasis(m, e, facets=fac)
            f += fem.LinearForm(lambda v, w, tv=tv: tv[0] * v[0] + tv[1] * v[1] + tv[2] * v[2]).assemble(fb)
            applied += tv * _area(m, fac)
        elif c.kind == "pressure":
            fac = regions[c.region]
            p = si(c.magnitude, "Pa")
            fb = fem.FacetBasis(m, e, facets=fac)
            f += fem.LinearForm(lambda v, w, p=p: -p * (w.n[0] * v[0] + w.n[1] * v[1] + w.n[2] * v[2])).assemble(fb)
            fbs = fem.FacetBasis(m, fem.ElementTetP1(), facets=fac)
            applied += np.array([fem.Functional(lambda w, i=i, p=p: -p * w.n[i]).assemble(fbs) for i in range(3)])
        elif c.kind == "body_acceleration":
            if mat.density is None:
                raise SolveError("body acceleration requires material density")
            g = si(c.magnitude, "m/s^2") * _unit_vec(c.vector) * si(mat.density, "kg/m^3")
            f += fem.LinearForm(lambda v, w, g=g: g[0] * v[0] + g[1] * v[1] + g[2] * v[2]).assemble(ib)
            vol = float(fem.Functional(lambda w: 1.0 + 0.0 * w.x[0]).assemble(fem.Basis(m, fem.ElementTetP1())))
            applied += g * vol
    if not D:
        raise SolveError("no displacement constraints: rigid-body motion is unconstrained")
    Dall = np.unique(np.concatenate(D))
    u, res = _solve(K, f, Dall, x, case.solver.linear_solver, case.solver.residual_tolerance)
    R = K @ u - f
    R_D = np.zeros(ib.N)
    R_D[Dall] = R[Dall]
    comp_dofs = [ib.get_dofs().all(f"u^{i + 1}") for i in range(3)]   # all dofs of component i
    tot_react = np.array([R_D[cd].sum() for cd in comp_dofs])
    bal = float(np.linalg.norm(tot_react + applied) / max(np.linalg.norm(applied), np.linalg.norm(tot_react), 1e-300))
    uh = ib.interpolate(u)
    vm = _von_mises(uh.grad, lam, mu)                      # (nel, nqp)
    vals: dict[str, float] = {}
    for q in case.outputs:
        vals[q.id] = _elastic_qoi(q, m, e, u, uh, vm, R_D, regions, ib, lam, mu)
    if any(c.kind == "surface_force" for c in case.conditions):
        warns.append("surface_force is applied as a uniform traction over its region (no point loads)")
    nodal = u[ib.nodal_dofs].T                             # (nverts, 3)
    return LevelOutput(vals, res, bal, ib.N, point_data={"displacement": nodal},
                       cell_data={"von_mises": vm.max(axis=1)}, warnings=warns)


def _elastic_qoi(q: QoI, m, e, u, uh, vm, R_D, regions, ib, lam, mu) -> float:
    import skfem as fem

    from ...units import Quantity
    to = lambda val, si_unit: Quantity.of(val, si_unit).to(q.unit)  # noqa: E731
    if q.kind == "reaction":
        if q.region is None:
            raise SolveError("reaction needs a region")
        dofs = ib.get_dofs(regions[q.region])
        if q.component in (None, "magnitude"):
            return to(float(np.linalg.norm([R_D[dofs.all(f"u^{i + 1}")].sum() for i in range(3)])), "N")
        return to(float(R_D[dofs.all(f"u^{AX[q.component] + 1}")].sum()), "N")
    if q.field == "displacement":
        comp = q.component or "magnitude"
        if q.region is not None:
            fb = fem.FacetBasis(m, e, facets=regions[q.region])
            uf = fb.interpolate(u)
            if q.kind == "average":
                num = fem.Functional(lambda w: _pick(w["uu"], comp)).assemble(fb, uu=uf)
                return to(num / _area(m, regions[q.region]), "m")
            arr = _pick(uf, comp)
        else:
            if q.kind == "average":
                raise SolveError("domain average of displacement not supported; give a region")
            arr = _pick(uh, comp)
        a = np.asarray(arr)
        return to(float(a.max() if q.kind == "max" else a.min()), "m")
    if q.field == "von_mises":
        if q.kind not in ("max", "average"):
            raise SolveError("von_mises supports max/average")
        if q.region is not None:
            fb = fem.FacetBasis(m, e, facets=regions[q.region])
            vmf = _von_mises(fb.interpolate(u).grad, lam, mu)
            if q.kind == "max":
                return to(float(vmf.max()), "Pa")
            return to(float((vmf * fb.dx).sum() / fb.dx.sum()), "Pa")
        if q.kind == "max":
            return to(float(vm.max()), "Pa")
        num = fem.Functional(lambda w: w["vm"]).assemble(fem.Basis(m, fem.ElementTetP1()), vm=vm)
        vol = fem.Functional(lambda w: 1.0 + 0.0 * w.x[0]).assemble(fem.Basis(m, fem.ElementTetP1()))
        return to(float(num / vol), "Pa")
    raise SolveError(f"unsupported structural output {q.kind}/{q.field}")


def _pick(ufield, comp):
    v = ufield.value if hasattr(ufield, "value") else ufield
    if comp == "magnitude":
        return np.sqrt((v ** 2).sum(axis=0))
    return v[AX[comp]]


def thermal(b: MeshBundle, case: AnalysisCase, mat: Material, regions: dict[str, np.ndarray],
            order: int) -> LevelOutput:
    import skfem as fem
    from skfem.helpers import dot, grad
    if mat.thermal is None:
        raise SolveError(f"material {mat.id} has no thermal model")
    k = si(mat.thermal.conductivity, "W/(m*K)")
    m = b.mesh
    e = fem.ElementTetP2() if order == 2 else fem.ElementTetP1()
    ib = fem.Basis(m, e)
    K = fem.BilinearForm(lambda u, v, w: k * dot(grad(u), grad(v))).assemble(ib)
    f = np.zeros(ib.N)
    x = np.zeros(ib.N)
    D: list[np.ndarray] = []
    src_total = 0.0
    neumann_in: dict[str, float] = {}
    conv: dict[str, tuple[float, float]] = {}
    for c in case.conditions:
        if c.kind == "temperature":
            d = ib.get_dofs(regions[c.region]).all()
            x[d] = si(c.value, "K")
            D.append(d)
        elif c.kind == "heat_flux":
            qv = si(c.value, "W/m^2")
            fb = fem.FacetBasis(m, e, facets=regions[c.region])
            f += fem.LinearForm(lambda v, w, qv=qv: qv * v).assemble(fb)
            neumann_in[c.region] = neumann_in.get(c.region, 0.0) + qv * _area(m, regions[c.region])
        elif c.kind == "convection":
            h, Ta = si(c.h, "W/(m^2*K)"), si(c.ambient, "K")
            fb = fem.FacetBasis(m, e, facets=regions[c.region])
            K = K + fem.BilinearForm(lambda u, v, w, h=h: h * u * v).assemble(fb)
            f += fem.LinearForm(lambda v, w, h=h, Ta=Ta: h * Ta * v).assemble(fb)
            conv[c.region] = (h, Ta)
        elif c.kind == "volumetric_heat":
            Q = si(c.value, "W/m^3")
            f += fem.LinearForm(lambda v, w, Q=Q: Q * v).assemble(ib)
            src_total += Q * float(fem.Functional(lambda w: 1.0 + 0.0 * w.x[0]).assemble(ib))
    if not D and not conv:
        raise SolveError("no temperature or convection condition: temperature level is undetermined")
    Dall = np.unique(np.concatenate(D)) if D else np.array([], dtype=np.int64)
    if Dall.size:
        T, res = _solve(K, f, Dall, x, case.solver.linear_solver, case.solver.residual_tolerance)
    else:
        import scipy.sparse.linalg as sla
        T = sla.spsolve(K.tocsc(), f)
        res = float(np.linalg.norm(K @ T - f) / max(np.linalg.norm(f), 1e-300))
    R = K @ T - f

    def out_flow(region: str) -> float:
        """Heat leaving the body through a region (W)."""
        fac = regions[region]
        total = 0.0
        if region in conv:
            h, Ta = conv[region]
            fb = fem.FacetBasis(m, e, facets=fac)
            total += float(fem.Functional(lambda w, h=h, Ta=Ta: h * (w["t"] - Ta)).assemble(fb, t=fb.interpolate(T)))
        total -= neumann_in.get(region, 0.0)
        dd = np.intersect1d(ib.get_dofs(fac).all(), Dall)
        if dd.size:
            total -= float(R[dd].sum())
        return total

    bc_regions = sorted(set(neumann_in) | set(conv) | {c.region for c in case.conditions if c.kind == "temperature"})
    if Dall.size:
        net_out = -float(R[Dall].sum()) + sum(out_flow(r) for r in set(neumann_in) | set(conv))
    else:
        net_out = sum(out_flow(r) for r in bc_regions)
    scale = max(abs(src_total), sum(abs(v) for v in neumann_in.values()), abs(net_out), 1e-300)
    bal = abs(net_out - src_total) / scale if scale > 1e-300 else 0.0
    from ...units import Quantity
    vals = {}
    for q in case.outputs:
        to = lambda val, s_u, q=q: Quantity.of(val, s_u).to(q.unit)  # noqa: E731
        if q.field == "temperature":
            if q.region is not None and q.kind == "average":
                fb = fem.FacetBasis(m, e, facets=regions[q.region])
                num = float(fem.Functional(lambda w: w["t"]).assemble(fb, t=fb.interpolate(T)))
                vals[q.id] = to(num / _area(m, regions[q.region]), "K")
            elif q.kind in ("max", "min"):   # Lagrange dofs are point values (vertices, edge midpoints)
                vals[q.id] = to(float(T.max() if q.kind == "max" else T.min()), "K")
            else:
                raise SolveError(f"unsupported temperature output {q.kind}")
        elif q.kind == "flux" and q.field == "heat_flux_normal":
            if q.region is None:
                raise SolveError("flux needs a region")
            vals[q.id] = to(out_flow(q.region), "W")
        else:
            raise SolveError(f"unsupported thermal output {q.kind}/{q.field}")
    return LevelOutput(vals, res, bal, ib.N, point_data={"temperature": T[ib.nodal_dofs[0]]})
