"""Spur gear outline by simulated generation with the basic rack cutter [GEOM].

Kinematics (gear centre at the origin, rack rolling without slip on the generating pitch circle r):
a rack point (u, v) — u along the rolling line, v away from the gear centre, origin at the pitch
point at phi = 0 — appears in the gear frame at  p(phi) = R(-phi) [u - r phi, r + v].

* Involute flank: the straight cutter flank. At each phi the contact point is where the flank
  normal passes through the instantaneous centre I (rack coords (r phi, 0)) — the law of gearing.
* Root fillet: the rounded cutter tip (radius rho, centre C). The generated surface is the envelope
  of the tip circle; its contact point is Q = C + rho * unit(C - I)  (same law, circle normal).
  This is the true trochoidal fillet produced by hobbing/shaping, not an arc approximation.
* Profile shift x moves the cutter reference line x*m away from the gear centre.

Cutter: 20 deg-style straight-sided rack with addendum 1.25 m (cuts the dedendum), tooth thickness
pi m / 2 on its reference line, tip radius rho = 0.300 m [SHI basic rack]. Undercut is NOT modelled:
callers must reject x < x_min (the CEM does), so the fillet meets the involute tangentially.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import formulas as F


@dataclass(frozen=True)
class Flank:
    fillet: list[tuple[float, float]]    # from root bottom up to the form point
    involute: list[tuple[float, float]]  # from the form point up to the tip circle
    r_form: float
    r_root: float
    r_tip: float
    phi_bottom: float


def _rot(p: tuple[float, float], a: float) -> tuple[float, float]:
    c, s = math.cos(a), math.sin(a)
    return (c * p[0] - s * p[1], s * p[0] + c * p[1])


def generate_flank(m: float, z: int, x: float, alpha: float, n_fillet: int = 16, n_involute: int = 24,
                   rho_coeff: float = F.RACK_TIP_RADIUS_COEFF) -> Flank:
    r = m * z / 2.0
    v_ref = x * m
    v_tip = v_ref - F.DEDENDUM_COEFF * m
    rho = rho_coeff * m
    v_c = v_tip + rho
    u_c = math.pi * m / 4.0 + (v_c - v_ref) * math.tan(alpha) - rho / math.cos(alpha)
    if u_c < 0:
        raise ValueError("rack tip radii overlap (tip land negative)")
    if v_c >= 0:
        raise ValueError("profile shift too large: cutter tip centre outside the rolling line")
    n = (math.cos(alpha), -math.sin(alpha))  # outward normal of the right cutter flank
    f0 = (math.pi * m / 4.0, v_ref)          # point on the right cutter flank

    def to_gear(u: float, v: float, phi: float) -> tuple[float, float]:
        return _rot((u - r * phi, r + v), -phi)

    def fillet_point(phi: float) -> tuple[float, float]:
        iu, iv = r * phi, 0.0
        du, dv = u_c - iu, v_c - iv
        L = math.hypot(du, dv)
        return to_gear(u_c + rho * du / L, v_c + rho * dv / L, phi)

    def flank_point(phi: float) -> tuple[float, float]:
        iu = r * phi
        t = n[0] * (iu - f0[0]) + n[1] * (0.0 - f0[1])
        return to_gear(iu - t * n[0], 0.0 - t * n[1], phi)

    phi_b = u_c / r                                   # tip centre directly below the pitch point
    tu, tv = u_c + rho * n[0], v_c + rho * n[1]       # tangency of tip round and straight flank
    phi_t = (tu + tv / math.tan(alpha)) / r           # its contact parameter (law of gearing)
    r_tip = F.tip_diameter(m, z, x) / 2.0
    if math.hypot(*flank_point(phi_t)) >= r_tip:
        raise ValueError("form circle above tip circle")
    # The working involute is generated for phi > phi_t (contact moves up the cutter flank, through the
    # pitch point at phi = pi m / (4 r) + ...); find where it reaches the tip circle.
    lo, hi = phi_t, phi_t + math.pi / 2
    if math.hypot(*flank_point(hi)) < r_tip:
        raise ValueError("involute does not reach the tip circle")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if math.hypot(*flank_point(mid)) < r_tip:
            lo = mid
        else:
            hi = mid
    phi_tip = 0.5 * (lo + hi)
    fillet = [fillet_point(phi_b + (phi_t - phi_b) * k / n_fillet) for k in range(n_fillet + 1)]
    inv_pts = [flank_point(phi_t + (phi_tip - phi_t) * k / n_involute) for k in range(n_involute + 1)]
    return Flank(fillet, inv_pts, math.hypot(*inv_pts[0]), math.hypot(*fillet[0]), r_tip, phi_b)


def gear_outline_segments(m: float, z: int, x: float, alpha: float, thinning: float = 0.0,
                          rotation: float = 0.0, n_fillet: int = 16, n_involute: int = 24) -> dict:
    """Closed outline as Geometry-IR profile segments (spline flanks + exact tip/root arcs).

    Tooth spaces are centred at angles 90deg + k*360/z (+ rotation). ``thinning`` reduces the
    circular tooth thickness at the reference circle (mm) — a backlash allowance chosen by the designer."""
    fl = generate_flank(m, z, x, alpha, n_fillet, n_involute)
    r = m * z / 2.0
    dth = thinning / (2.0 * r)
    # right flank of the space (angles < 90deg), bottom -> tip; thinning rotates it toward the tooth
    right = [_rot(p, -dth) for p in fl.fillet + fl.involute[1:]]
    pitch = 2.0 * math.pi / z
    segments: list[dict] = []

    def mirror(p: tuple[float, float]) -> tuple[float, float]:  # about the 90deg line (x -> -x)
        return (-p[0], p[1])

    start = None
    for k in range(z):
        a = rotation + k * pitch
        r_side = [_rot(p, a) for p in right]               # bottom -> tip
        l_side = [_rot(mirror(p), a) for p in right]       # bottom -> tip (mirror)
        nf = len(fl.fillet)
        # CCW walk: right flank tip->bottom, root arc, left flank bottom->tip, tip arc to next space.
        rt = list(reversed(r_side))
        if start is None:
            start = rt[0]
        inv_part = rt[: len(rt) - nf + 1]
        fil_part = rt[len(rt) - nf:]
        segments.append({"kind": "spline", "points": [list(p) for p in inv_part[1:]]})
        segments.append({"kind": "spline", "points": [list(p) for p in fil_part[1:]]})
        b_r, b_l = r_side[0], l_side[0]
        ang_mid = math.atan2(b_r[1] + b_l[1], b_r[0] + b_l[0])
        if math.dist(b_r, b_l) > 1e-9:
            segments.append({"kind": "arc", "through": [fl.r_root * math.cos(ang_mid), fl.r_root * math.sin(ang_mid)],
                             "to": list(b_l)})
        segments.append({"kind": "spline", "points": [list(p) for p in l_side[1:nf]]})
        segments.append({"kind": "spline", "points": [list(p) for p in l_side[nf:]]})
        # tip land from this space's left tip to the next space's right tip
        nxt_tip = _rot(right[-1], a + pitch)
        t0 = l_side[-1]
        a0, a1 = math.atan2(t0[1], t0[0]), math.atan2(nxt_tip[1], nxt_tip[0])
        sweep = (a1 - a0) % (2 * math.pi)
        if sweep <= 1e-9 or sweep >= pitch:
            raise ValueError("pointed or overlapping teeth: no tip land")
        am = a0 + sweep / 2
        segments.append({"kind": "arc", "through": [fl.r_tip * math.cos(am), fl.r_tip * math.sin(am)],
                         "to": list(nxt_tip)})
    assert start is not None
    # close exactly onto the start point
    segments[-1]["to"] = list(start)
    return {"start": list(start), "segments": segments, "flank": fl}
