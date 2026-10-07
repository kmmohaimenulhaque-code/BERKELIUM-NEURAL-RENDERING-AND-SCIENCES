"""ADR-020 acausal law networks: dimension checks, structural analysis, inversion, diagnostics, derivatives."""
import math

import pytest

from berkelium.evidence import Claim, resolve
from berkelium.laws import LawError, LawSet, Relation, Var, as_law, optimise, propagate, sensitivities, solve
from berkelium.laws.library import beam, nozzle, rocket_chamber, vessel, wall
from berkelium.units import Quantity as Q


def q(v, u="1"):
    return Q.of(v, u)


NOZ = {"gamma": q(1.4), "p0": q(1, "MPa"), "T0": q(3000, "K"), "R": q(287, "J/(kg*K)"), "At": q(100, "mm^2")}


def test_dimension_errors_rejected_at_definition():
    with pytest.raises(LawError, match="dimension"):
        LawSet("bad", {"x": Var("x", "mm"), "y": Var("y", "N")}, [Relation("r", "x == y")])
    with pytest.raises(LawError, match="undeclared"):
        LawSet("bad", {"x": Var("x", "mm")}, [Relation("r", "x == z")])


def test_isentropic_reference_values_naca1135():
    s = solve(nozzle(), {**NOZ, "M_e": q(2.0)})
    assert s.status == "solved"
    assert abs(s.get("eps", "1") - 1.6875) < 1e-4 and abs(s.get("pe", "kPa") / 1000 - 0.1278) < 1e-4


def test_inversion_detects_two_branches_and_bounds_select_one():
    s = solve(nozzle(), {**NOZ, "eps": q(1.6875)})
    assert s.status == "ambiguous" and len(s.roots["M_e"]) == 2
    assert abs(s.roots["M_e"][0] - 0.3722) < 1e-3 and abs(s.roots["M_e"][1] - 2.0) < 1e-9
    nz = nozzle()
    nz.vars["M_e"] = Var("M_e", "1", 1.0)
    assert abs(solve(nz, {**NOZ, "eps": q(1.6875)}).get("M_e", "1") - 2.0) < 1e-9


def test_causality_is_derived_not_written():
    """The same relations size h from a stress OR check stress from h."""
    k = {"P": q(200, "N"), "L": q(100, "mm"), "E": q(207, "GPa"), "a": q(2), "rho": q(7800, "kg/m^3")}
    size = solve(beam(), {**k, "sigma": q(150, "MPa")})
    h = size.get("h", "mm")
    assert abs(h - (6 * 200 * 0.1 * 2 / 150e6) ** (1 / 3) * 1e3) < 1e-9
    check = solve(beam(), {**k, "h": q(h, "mm")})
    assert abs(check.get("sigma", "MPa") - 150) < 1e-9
    assert [b for b, _ in size.plan] != [b for b, _ in check.plan]


def test_underdetermined_partial_and_overdetermined_conflict():
    s = solve(beam(), {"P": q(200, "N"), "L": q(100, "mm"), "E": q(207, "GPa"), "b": q(5, "mm"), "a": q(2)})
    assert s.status == "underdetermined" and "rho" in s.diagnostics[0] or "mass" in s.diagnostics[0]
    assert abs(s.get("delta", "mm") - 200 * 0.1 ** 3 / (3 * 207e9 * 0.005 * 0.01 ** 3 / 12) * 1e3) < 1e-9
    c = solve(beam(), {"P": q(1, "N"), "L": q(100, "mm"), "E": q(207, "GPa"), "a": q(2), "b": q(5, "mm"),
                       "h": q(12, "mm"), "rho": q(1, "kg/m^3")})
    assert c.status == "conflict"


def test_validity_domain_flags_short_beam():
    k = {"P": q(1, "N"), "L": q(30, "mm"), "E": q(207, "GPa"), "a": q(1), "rho": q(1, "kg/m^3")}
    s = solve(beam(), {**k, "b": q(10, "mm")})                 # solving FOR h: the only root is invalid
    assert s.status == "invalid_domain" and "violate validity" in s.diagnostics[0]
    s = solve(beam(), {**k, "h": q(10, "mm")})                 # h given: computed, but flagged
    assert any(c.kind == "validity" and c.status == "invalid_domain" for c in s.checks)


def test_sensitivities_and_uncertainty_match_theory():
    bb = beam()
    bb.relations = [r for r in bb.relations if r.id != "aspect"]
    k = {"P": q(200, "N"), "L": q(100, "mm"), "E": q(207, "GPa"), "b": q(5, "mm"), "h": q(10, "mm"),
         "rho": q(7800, "kg/m^3")}
    el = sensitivities(bb, k, ["delta"])["delta"]
    for name, exact in {"P": 1, "L": 3, "E": -1, "b": -1, "h": -3}.items():
        assert abs(el[name] - exact) < 1e-6
    u = propagate(bb, k, {"E": 0.05, "h": 0.02}, ["delta"], n=200)["delta"]
    assert abs(u["sigma_linear_si"] / u["value_si"] - math.hypot(0.05, 0.06)) < 1e-6
    assert abs(u["mc_std_si"] / u["value_si"] - math.hypot(0.05, 0.06)) < 0.015


def test_optimum_equals_closed_form_and_names_active_constraint():
    s, info = optimise(beam(), {"P": q(200, "N"), "L": q(100, "mm"), "E": q(207, "GPa"), "a": q(2),
                                "rho": q(7800, "kg/m^3"), "delta_max": q(0.5, "mm"), "sigma_allow": q(150, "MPa")},
                       {"h": (4.0, 40.0)}, "mass")
    assert abs(s.get("h", "mm") - (6 * 200 * 0.1 * 2 / 150e6) ** (1 / 3) * 1e3) < 1e-6
    assert info["active_constraints"] == ["strength_req"]


def test_cross_domain_composition_and_second_heat_domain():
    rc = rocket_chamber()
    k = {"gamma": q(1.2), "R": q(350, "J/(kg*K)"), "T0": q(3200, "K"), "p0": q(2, "MPa"), "pe": q(101.325, "kPa"),
         "F": q(1000, "N"), "CR": q(4), "nu": q(0.3), "ro": q(25, "mm"), "rho_s": q(7800, "kg/m^3")}
    s = solve(rc, k)
    assert s.status == "solved" and abs(s.get("ri", "mm") - s.get("d_t", "mm")) < 1e-9   # CR = 4 => D_c = 2 d_t
    assert abs(s.get("F", "N") - s.get("mdot", "kg/s") * s.get("ve", "m/s")) < 1e-6
    w = solve(wall(), {"T_in": q(473.15, "K"), "T_amb": q(298.15, "K"), "T_s": q(318.15, "K"),
                       "k_ins": q(0.04, "W/(m*K)"), "h_o": q(10, "W/(m^2*K)")})
    assert abs(w.get("t_ins", "mm") - 31.0) < 1e-9


def test_law_network_enters_evidence_calculus_and_detects_misused_model():
    exact = as_law(vessel(), "svm", "MPa", "vessel.svm", id="lame")
    thin = vessel()
    rel = [Relation("thin", "sig_t == p*ri/(ro - ri)") if r.id == "lame_hoop_inner" else r for r in thin.relations]
    thin_law = as_law(LawSet("thin", thin.vars, rel), "svm", "MPa", "vessel.svm", id="thin", cost=0.5)
    pt = {"p": q(10, "MPa"), "ri": q(10, "mm"), "ro": q(20, "mm"), "nu": q(0.3)}
    r = resolve(Claim("vessel.svm", "<=", 25, "MPa"), pt, [exact], exhaustive=True)
    assert r.verdict == "pass" and abs(r.evidence[0].estimate.value - 23.1325) < 1e-3
    r = resolve(Claim("vessel.svm", "<=", 25, "MPa"), pt, [thin_law, exact], exhaustive=True)
    assert r.verdict == "conflict"
