"""Spur gear engineering formulas — pure, deterministic, SI-free (mm, N, MPa, rad, rev/min).

Every function cites its source. Sources (see docs/SOURCES_AND_LICENSES.md):
  [KHK]  KHK Gear Technical Reference (KHK Co.), Eqs. 3.2, 3.9, 3.10; Tables 4.3-4.4.
  [SHI]  Budynas & Nisbett, Shigley's Mechanical Engineering Design, 8th ed., McGraw-Hill:
         Ch. 13 Eqs. 13-10..13-13; Ch. 14 Eqs. 14-4..14-8, Table 14-2, §14-1.
  [GEOM] Relations derived directly from involute / line-of-action geometry (no constants);
         each is cross-checked numerically in tests/test_gear_formulas.py.
Nothing here is an AGMA/ISO rating; Lewis is preliminary sizing only [SHI §14-1].
"""

from __future__ import annotations

import bisect
import math

REF_KHK = "KHK Gear Technical Reference"
REF_SHI = "Shigley's Mechanical Engineering Design, 8th ed. (Budynas & Nisbett)"

# Basic rack, 20 deg full depth: addendum 1.00 m, dedendum 1.25 m [SHI Table 13-1 via Ex. 14-2 / Prob. 13-4].
ADDENDUM_COEFF = 1.0
DEDENDUM_COEFF = 1.25
# Basic-rack tip (fillet) radius 0.300/P, i.e. 0.300 m in metric [SHI Ex. 14-2: r_f = 0.300/P].
RACK_TIP_RADIUS_COEFF = 0.300


# ------------------------------------------------------------------ involute function
def inv(alpha: float) -> float:
    """inv(a) = tan(a) - a   [KHK Eq. 3.2]."""
    return math.tan(alpha) - alpha


def inv_inverse(c: float, tol: float = 1e-14, max_iter: int = 60) -> tuple[float, int]:
    """Solve tan(a) - a = c for a in (0, pi/2). Newton with f' = tan^2(a), cube-root seed
    a0 = (3c)^(1/3) (from tan a - a ~ a^3/3), bisection fallback. Returns (alpha, iterations)."""
    if c < 0:
        raise ValueError("inv_inverse requires c >= 0")
    if c == 0:
        return 0.0, 0
    lo, hi = 1e-12, math.pi / 2 - 1e-9
    a = min(max((3.0 * c) ** (1.0 / 3.0), lo), hi)
    for i in range(1, max_iter + 1):
        f = inv(a) - c
        d = math.tan(a) ** 2
        if f > 0:
            hi = a
        else:
            lo = a
        step = f / d if d > 1e-300 else float("inf")
        nxt = a - step
        if not lo < nxt < hi:
            nxt = 0.5 * (lo + hi)
        if abs(nxt - a) < tol:
            return nxt, i
        a = nxt
    raise ArithmeticError("inv_inverse did not converge")


# ------------------------------------------------------------------ basic geometry (mm)
def reference_diameter(m: float, z: int) -> float:
    return m * z  # d = m z [SHI 13; KHK §3]


def base_diameter(m: float, z: int, alpha: float) -> float:
    return m * z * math.cos(alpha)  # d_b = d cos(a) [SHI Prob. 13-4]


def tip_diameter(m: float, z: int, x: float) -> float:
    """d_a = d + 2 m (1 + x). Assumes NO tip shortening [KHK Table 4.3 structure]; with
    x1 + x2 != y the tip clearance shrinks, which is checked separately (root_clearance)."""
    return m * z + 2.0 * m * (ADDENDUM_COEFF + x)


def root_diameter(m: float, z: int, x: float) -> float:
    return m * z - 2.0 * m * (DEDENDUM_COEFF - x)  # d_f = d - 2 m (1.25 - x) [KHK Table 4.3]


def tooth_thickness_ref(m: float, x: float, alpha: float) -> float:
    """Circular tooth thickness on the reference circle, zero backlash: s = m (pi/2 + 2 x tan a) [KHK]."""
    return m * (math.pi / 2.0 + 2.0 * x * math.tan(alpha))


def tooth_thickness_at(r_y: float, m: float, z: int, x: float, alpha: float, thinning: float = 0.0) -> float:
    """s_y = 2 r_y (s/d + inv a - inv a_y), cos a_y = r_b / r_y  [involute relation, KHK/GEOM].
    ``thinning`` is a circumferential reduction of s at the reference circle (backlash allowance)."""
    d = m * z
    rb = 0.5 * d * math.cos(alpha)
    if r_y < rb:
        raise ValueError("radius below base circle")
    a_y = math.acos(rb / r_y)
    s = tooth_thickness_ref(m, x, alpha) - thinning
    return 2.0 * r_y * (s / d + inv(alpha) - inv(a_y))


# ------------------------------------------------------------------ pair geometry
def working_pressure_angle(z1: int, z2: int, x1: float, x2: float, alpha: float) -> tuple[float, int]:
    """inv a_w = 2 tan a (x1 + x2)/(z1 + z2) + inv a   [KHK Table 4.4 item 5]."""
    c = 2.0 * math.tan(alpha) * (x1 + x2) / (z1 + z2) + inv(alpha)
    return inv_inverse(c)


def centre_distance_standard(m: float, z1: int, z2: int) -> float:
    return m * (z1 + z2) / 2.0  # a0 [KHK Table 4.4]


def centre_distance_working(m: float, z1: int, z2: int, alpha: float, alpha_w: float) -> float:
    """a_w = a0 cos a / cos a_w   (a cos a = a' cos a') [KHK Table 4.4; arXiv 2401.08266 Eq. 7]."""
    return centre_distance_standard(m, z1, z2) * math.cos(alpha) / math.cos(alpha_w)


def centre_distance_modification(z1: int, z2: int, alpha: float, alpha_w: float) -> float:
    """y = (z1+z2)/2 (cos a / cos a_w - 1)  [KHK Table 4.4]."""
    return (z1 + z2) / 2.0 * (math.cos(alpha) / math.cos(alpha_w) - 1.0)


def shift_sum_for_centre_distance(m: float, z1: int, z2: int, alpha: float, a_w: float) -> float:
    """Inverse calculation: a_w -> a_w pressure angle -> x1 + x2  [KHK §4, 'inverse calculation']."""
    a0 = centre_distance_standard(m, z1, z2)
    cos_aw = a0 * math.cos(alpha) / a_w
    if not 0 < cos_aw <= 1:
        raise ValueError("centre distance not reachable with this module/tooth count")
    aw = math.acos(cos_aw)
    return (z1 + z2) * (inv(aw) - inv(alpha)) / (2.0 * math.tan(alpha))


def transverse_contact_ratio(m: float, z1: int, z2: int, x1: float, x2: float, alpha: float) -> float:
    """eps_a = [sqrt(ra1^2-rb1^2) + sqrt(ra2^2-rb2^2) - a_w sin a_w] / (pi m cos a)  [KHK; SHI m_c = L_ab/p_b]."""
    aw, _ = working_pressure_angle(z1, z2, x1, x2, alpha)
    a_w = centre_distance_working(m, z1, z2, alpha, aw)
    ra1, ra2 = tip_diameter(m, z1, x1) / 2, tip_diameter(m, z2, x2) / 2
    rb1, rb2 = base_diameter(m, z1, alpha) / 2, base_diameter(m, z2, alpha) / 2
    lab = math.sqrt(ra1**2 - rb1**2) + math.sqrt(ra2**2 - rb2**2) - a_w * math.sin(aw)
    return lab / (math.pi * m * math.cos(alpha))


def involute_interference_margin(m: float, z_driver_tip: int, z_other: int, x_tip: float, x_other: float,
                                 alpha: float) -> float:
    """Margin (mm) along the line of action before the tip of gear 'tip' passes the interference point
    (tangency of the line of action with the OTHER gear's base circle):
        margin = a_w sin a_w - sqrt(r_a,tip^2 - r_b,tip^2)        >= 0 means no involute interference [GEOM].
    Valid for any profile shift; reduces to SHI Eqs. 13-11/13-12 limits for x = 0."""
    aw, _ = working_pressure_angle(z_driver_tip, z_other, x_tip, x_other, alpha)
    a_w = centre_distance_working(m, z_driver_tip, z_other, alpha, aw)
    ra = tip_diameter(m, z_driver_tip, x_tip) / 2
    rb = base_diameter(m, z_driver_tip, alpha) / 2
    return a_w * math.sin(aw) - math.sqrt(ra**2 - rb**2)


def root_clearance(m: float, z1: int, z2: int, x1: float, x2: float, alpha: float) -> tuple[float, float]:
    """Radial clearance between each tip circle and the mating root circle at a_w [GEOM]:
    c1 = a_w - r_a1 - r_f2, c2 = a_w - r_a2 - r_f1."""
    aw, _ = working_pressure_angle(z1, z2, x1, x2, alpha)
    a_w = centre_distance_working(m, z1, z2, alpha, aw)
    ra1, ra2 = tip_diameter(m, z1, x1) / 2, tip_diameter(m, z2, x2) / 2
    rf1, rf2 = root_diameter(m, z1, x1) / 2, root_diameter(m, z2, x2) / 2
    return a_w - ra1 - rf2, a_w - ra2 - rf1


# ------------------------------------------------------------------ undercut / interference limits
def min_teeth_no_undercut(alpha: float, k: float = 1.0) -> float:
    """Theoretical z_min = 2k / sin^2(a)  (rack cutter, x = 0)  [KHK Eq. 3.9; SHI Eq. 13-13]."""
    return 2.0 * k / math.sin(alpha) ** 2


def min_shift_no_undercut(z: int, alpha: float, k: float = 1.0) -> float:
    """x_min = k - (z/2) sin^2(a)  [KHK Eq. 3.10]."""
    return k - 0.5 * z * math.sin(alpha) ** 2


def shigley_min_pinion_for_ratio(ratio: float, alpha: float, k: float = 1.0) -> float:
    """Smallest pinion without interference for gear ratio m_G = N_G/N_P (x = 0, full depth)  [SHI Eq. 13-11]."""
    s2 = math.sin(alpha) ** 2
    return 2.0 * k / ((1.0 + 2.0 * ratio) * s2) * (ratio + math.sqrt(ratio**2 + (1.0 + 2.0 * ratio) * s2))


def shigley_max_gear_for_pinion(n_p: int, alpha: float, k: float = 1.0) -> float:
    """Largest gear that can run with pinion n_p without interference (x = 0)  [SHI Eq. 13-12].
    Returns +inf when the denominator is <= 0 (unlimited)."""
    s2 = math.sin(alpha) ** 2
    den = 4.0 * k - 2.0 * n_p * s2
    return math.inf if den <= 0 else (n_p**2 * s2 - 4.0 * k**2) / den


# ------------------------------------------------------------------ Lewis bending (preliminary)
# [SHI Table 14-2] Lewis form factor Y, 20 deg normal pressure angle, full-depth teeth, P = 1.
LEWIS_Y_20FD: tuple[tuple[int, float], ...] = (
    (12, 0.245), (13, 0.261), (14, 0.277), (15, 0.290), (16, 0.296), (17, 0.303), (18, 0.309), (19, 0.314),
    (20, 0.322), (21, 0.328), (22, 0.331), (24, 0.337), (26, 0.346), (28, 0.353), (30, 0.359), (34, 0.371),
    (38, 0.384), (43, 0.397), (50, 0.409), (60, 0.422), (75, 0.435), (100, 0.447), (150, 0.460),
    (300, 0.472), (400, 0.480),
)


def lewis_form_factor(z: int) -> tuple[float, bool]:
    """Y from [SHI Table 14-2], linear interpolation between rows (as Shigley does in Ex. 14-4).
    Returns (Y, interpolated). Raises ValueError outside 12..400 teeth."""
    zs = [r[0] for r in LEWIS_Y_20FD]
    if z < zs[0] or z > zs[-1]:
        raise ValueError(f"Lewis Y table covers 12..400 teeth, got {z}")
    i = bisect.bisect_left(zs, z)
    if zs[i] == z:
        return LEWIS_Y_20FD[i][1], False
    (z0, y0), (z1, y1) = LEWIS_Y_20FD[i - 1], LEWIS_Y_20FD[i]
    return y0 + (y1 - y0) * (z - z0) / (z1 - z0), True


# Barth / Shigley velocity factors, metric forms with V in m/s; K_v >= 1 (AGMA 2001-D04 convention) [SHI 14-6a..d].
VELOCITY_FACTORS = {
    "cast_iron_cast_profile": lambda v: (3.05 + v) / 3.05,      # Eq. 14-6a
    "cut_or_milled": lambda v: (6.1 + v) / 6.1,                 # Eq. 14-6b
    "hobbed_or_shaped": lambda v: (3.56 + math.sqrt(v)) / 3.56,  # Eq. 14-6c
    "shaved_or_ground": lambda v: math.sqrt((5.56 + math.sqrt(v)) / 5.56),  # Eq. 14-6d
}
VELOCITY_FACTORS_US = {  # V in ft/min [SHI Eqs. 14-4a, 14-4b, 14-5a, 14-5b]
    "cast_iron_cast_profile": lambda v: (600 + v) / 600,
    "cut_or_milled": lambda v: (1200 + v) / 1200,
    "hobbed_or_shaped": lambda v: (50 + math.sqrt(v)) / 50,
    "shaved_or_ground": lambda v: math.sqrt((78 + math.sqrt(v)) / 78),
}


def pitch_line_velocity(d_mm: float, n_rpm: float) -> float:
    """V = pi d n / 60000  [m/s, d in mm]  (SHI Ex. 14-1 in SI units)."""
    return math.pi * d_mm * n_rpm / 60000.0


def tangential_load(torque_nmm: float, d_mm: float) -> float:
    """W_t = 2 T / d  [N]  (statics)."""
    return 2.0 * torque_nmm / d_mm


def lewis_bending_stress(w_t: float, face_mm: float, m: float, y: float, k_v: float) -> float:
    """sigma = K_v W_t / (F m Y)  [MPa]  [SHI Eq. 14-8]. Preliminary estimate only [SHI §14-1]."""
    if k_v < 1.0:
        raise ValueError("K_v must be >= 1 (AGMA 2001-D04 convention)")
    return k_v * w_t / (face_mm * m * y)


def face_width_guideline(m: float) -> tuple[float, float]:
    """3p <= F <= 5p with p = pi m  [SHI §14-1]."""
    p = math.pi * m
    return 3.0 * p, 5.0 * p
