"""Solution verification: discretisation error from systematic refinement (PHYSICS-6 core).

Method: Roache's Grid Convergence Index as formalised by Celik et al. (2008), "Procedure for Estimation and
Reporting of Uncertainty Due to Discretization in CFD Applications", J. Fluids Eng. 130(7) 078001 —
including its fixed-point equation for the observed order with non-constant refinement ratio (needed for
unstructured Gmsh refinements). Roache (1994), J. Fluids Eng. 116:405-413, for the GCI and Fs = 1.25.

The result is per quantity of interest: a QoI can converge while another (e.g. a peak stress at a
re-entrant constraint) does not. Non-monotone or order-less sequences are reported as non_converged; they
are never silently averaged.
"""

from __future__ import annotations

import math

from .schema import ConvergenceStudy


def observed_order(f1: float, f2: float, f3: float, r21: float, r32: float, iters: int = 200) -> float | None:
    e21, e32 = f2 - f1, f3 - f2
    if e21 == 0 or e32 == 0:
        return None
    ratio = e32 / e21
    s = 1.0 if ratio > 0 else -1.0
    p = abs(math.log(abs(ratio))) / math.log(r21)
    for _ in range(iters):
        try:
            q = math.log((r21 ** p - s) / (r32 ** p - s))
        except (ValueError, ZeroDivisionError):
            return None
        pn = abs(math.log(abs(ratio)) + q) / math.log(r21)
        if abs(pn - p) < 1e-12:
            return pn
        p = pn
    return p


def study(qoi: str, values: list[float], h: list[float], fs: float, max_rel: float, formal_order: float,
          abs_floor: float = 0.0) -> ConvergenceStudy:
    """values/h ordered coarse -> fine."""
    if len(values) == 1:
        return ConvergenceStudy(qoi=qoi, method="single_level", status="not_evaluated",
                                message="one mesh level: discretisation error not estimated")
    f = values[::-1]          # f[0] = finest
    hh = h[::-1]
    scale = max(abs(f[0]), abs_floor, 1e-300)
    if all(abs(v - f[0]) <= 1e-6 * scale for v in f):
        return ConvergenceStudy(qoi=qoi, method="exact_reproduction", refinement_ratios=[hh[1] / hh[0]],
                                extrapolated=f[0], gci_fine=0.0, status="converged",
                                message="unchanged across levels within 1e-6 of the case scale (the discrete "
                                        "space reproduces this quantity)")
    if len(values) == 2:
        r = hh[1] / hh[0]
        p = formal_order
        gci = fs * abs((f[0] - f[1]) / scale) / (r ** p - 1)
        st = "converged" if gci <= max_rel else "non_converged"
        return ConvergenceStudy(qoi=qoi, method="richardson_gci_celik2008", refinement_ratios=[r],
                                observed_order=None, gci_fine=gci, status=st,
                                message=f"2 levels: formal order {p} ASSUMED (not observed); Fs should be 3.0")
    r21, r32 = hh[1] / hh[0], hh[2] / hh[1]
    e21, e32 = f[1] - f[0], f[2] - f[1]
    p = observed_order(f[0], f[1], f[2], r21, r32)
    if p is None or e21 * e32 <= 0:
        return ConvergenceStudy(qoi=qoi, method="richardson_gci_celik2008", refinement_ratios=[r21, r32],
                                observed_order=p, status="non_converged",
                                message="oscillatory or stagnant sequence; no asymptotic error estimate")
    # Conservative (Roache): never let an observed order ABOVE the formal one shrink the error band.
    pu = min(p, formal_order)
    ext = (r21 ** p * f[0] - f[1]) / (r21 ** p - 1)
    gci12 = fs * abs(e21 / scale) / (r21 ** pu - 1)
    gci23 = fs * abs(e32 / max(abs(f[1]), abs_floor, 1e-300)) / (r32 ** pu - 1)
    asym = gci23 / (r21 ** pu * gci12) if gci12 > 0 else None
    problems = []
    if p < 0.5 * formal_order:
        problems.append(f"observed order {p:.2f} << formal {formal_order} (singularity or pre-asymptotic)")
    if p > 2.0 * formal_order:
        problems.append(f"observed order {p:.2f} >> formal {formal_order} (pre-asymptotic or error cancellation)")
    if gci12 > max_rel:
        problems.append(f"GCI {gci12:.3%} > allowed {max_rel:.1%}")
    st = "non_converged" if problems else "converged"
    msg = "; ".join(problems) or f"p={p:.2f} (GCI uses {pu:.2f}), GCI={gci12:.3%}, extrapolated={ext:.6g}"
    return ConvergenceStudy(qoi=qoi, method="richardson_gci_celik2008", refinement_ratios=[r21, r32],
                            observed_order=p, extrapolated=ext, gci_fine=gci12, asymptotic_ratio=asym,
                            status=st, message=msg)
