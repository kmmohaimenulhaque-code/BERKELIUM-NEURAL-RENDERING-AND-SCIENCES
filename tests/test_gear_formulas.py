"""Oracle tests: every expected value comes from a cited worked example (see formulas.py)."""
import math

import pytest

from berkelium.cem.library.gear import formulas as F

D20 = math.radians(20)


def test_involute_and_inverse():
    assert F.inv(D20) == pytest.approx(0.014904, abs=1e-6)                       # KHK Eq. 3.2
    for c in (1e-6, 0.0149, 0.0343, 0.2, 1.0):
        a, it = F.inv_inverse(c)
        assert F.inv(a) == pytest.approx(c, rel=1e-12) and it < 60


def test_khk_shifted_pair_table_4_4():
    aw, _ = F.working_pressure_angle(12, 24, 0.6, 0.36, D20)
    assert F.inv(aw) == pytest.approx(0.034316, abs=1e-6)
    assert math.degrees(aw) == pytest.approx(26.0886, abs=1e-4)
    assert F.centre_distance_modification(12, 24, D20, aw) == pytest.approx(0.8333, abs=1e-4)
    assert F.centre_distance_working(3, 12, 24, D20, aw) == pytest.approx(56.4999, abs=1e-4)
    assert F.centre_distance_standard(3, 12, 24) == 54.0
    # inverse calculation recovers x1 + x2
    assert F.shift_sum_for_centre_distance(3, 12, 24, D20, 56.4999) == pytest.approx(0.96, abs=1e-4)


def test_undercut_limits():
    assert F.min_teeth_no_undercut(D20) == pytest.approx(17.097, abs=1e-3)       # KHK 3.9 / SHI 13-13
    assert F.min_shift_no_undercut(12, D20) == pytest.approx(0.298, abs=1e-3)    # KHK 3.10
    for ratio, exp in [(2, 14.16), (2.5, 14.64), (3, 14.98), (4, 15.44), (5, 15.74)]:   # SHI 13-11 (sol. 13-8, 13-10)
        assert F.shigley_min_pinion_for_ratio(ratio, D20) == pytest.approx(exp, abs=1e-2)
    for n_p, exp in [(13, 16), (14, 26), (15, 45), (16, 101), (17, 1309)]:              # SHI 13-12 (sol. 13-8)
        assert math.floor(F.shigley_max_gear_for_pinion(n_p, D20)) == pytest.approx(exp, abs=1)
    assert math.isinf(F.shigley_max_gear_for_pinion(18, D20))


def test_interference_margin_consistent_with_shigley():
    # SHI 13-12: pinion 15 runs with up to 45 teeth; 46+ interferes (x = 0)
    assert F.involute_interference_margin(1, 45, 15, 0, 0, D20) >= 0
    assert F.involute_interference_margin(1, 47, 15, 0, 0, D20) < 0


def test_contact_ratio_shigley_13_4_data():
    # P = 3 teeth/in -> m = 25.4/3 mm; N = 21/28; analytic value 1.604 (manual's graphical value 1.55)
    m = 25.4 / 3
    assert F.transverse_contact_ratio(m, 21, 28, 0, 0, D20) == pytest.approx(1.604, abs=1e-3)
    assert F.base_diameter(m, 21, D20) / 25.4 == pytest.approx(6.578, abs=1e-3)


def test_lewis_shigley_example_14_1():
    # P = 8, F = 1.5 in, N = 16, n = 1200 rpm, sigma_all = 10 kpsi, cut/milled -> W_t = 365 lbf, H = 6.95 hp
    d_in = 16 / 8
    V = math.pi * d_in * 1200 / 12                       # ft/min
    assert V == pytest.approx(628, abs=0.5)
    kv = F.VELOCITY_FACTORS_US["cut_or_milled"](V)
    assert kv == pytest.approx(1.52, abs=5e-3)
    Y, interp = F.lewis_form_factor(16)
    assert Y == 0.296 and not interp
    wt_lbf = 1.5 * Y * 10000 / (kv * 8)
    assert wt_lbf == pytest.approx(365, rel=5e-3)
    assert wt_lbf * V / 33000 == pytest.approx(6.95, rel=5e-3)
    # metric form gives the same stress
    m_mm = 25.4 / 8
    sigma = F.lewis_bending_stress(wt_lbf * 4.4482216152605, 1.5 * 25.4, m_mm, Y, kv)
    assert sigma == pytest.approx(10000 * 0.006894757293168361, rel=1e-6)


def test_lewis_table_bounds_and_interp():
    assert F.lewis_form_factor(22) == (0.331, False)
    y, interp = F.lewis_form_factor(23)
    assert interp and y == pytest.approx((0.331 + 0.337) / 2)
    with pytest.raises(ValueError):
        F.lewis_form_factor(11)
    with pytest.raises(ValueError):
        F.lewis_bending_stress(1, 1, 1, 0.3, 0.9)


def test_tooth_thickness_relation():
    m, z, x = 2.0, 20, 0.0
    r = m * z / 2
    assert F.tooth_thickness_at(r, m, z, x, D20) == pytest.approx(math.pi * m / 2)
    ra = F.tip_diameter(m, z, x) / 2
    assert 0 < F.tooth_thickness_at(ra, m, z, x, D20) < math.pi * m / 2
