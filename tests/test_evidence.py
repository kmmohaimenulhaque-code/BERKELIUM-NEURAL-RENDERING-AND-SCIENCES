"""ADR-019 evidence calculus: decision rules, validity domains, falsification, calibration, generality."""
import pytest

from berkelium.evidence import Claim, Est, Law, calibrate, calibrated, resolve
from berkelium.evidence.library import beam_laws, pipe_laws, point
from berkelium.units import UnitError

X = point(L=(100, "mm"), h=(10, "mm"))
C = Claim("q", "<=", 1.0, "mm")


def law(id, value, mf, cost=1.0, validity=(), unit="mm"):
    return Law(id, "q", unit, "analytic", cost, lambda p: Est(value, unit, 0.0, mf), list(validity),
               {"s": "L / h"})


def test_cheapest_decisive_law_wins_and_unknown_bound_only_corroborates():
    r = resolve(C, X, [law("unknown", 0.5, None, 0.1), law("cheap", 0.5, 0.1, 1), law("dear", 0.5, 0.0, 9)])
    assert r.verdict == "pass" and r.decided_by == "cheap" and r.cost == 1.1
    assert r.evidence[0].reason.startswith("interval unknown")


def test_straddling_interval_escalates_then_insufficient():
    r = resolve(C, X, [law("rough", 0.95, 0.2)])
    assert r.verdict == "indeterminate"
    r = resolve(C, X, [law("rough", 0.95, 0.2), law("fine", 0.95, 0.01, 5)])
    assert r.verdict == "pass" and r.decided_by == "fine"
    assert resolve(C, X, []).verdict == "insufficient_evidence"


def test_validity_is_dimension_checked_and_recorded():
    r = resolve(C, X, [law("bad_pred", 0.5, 0.0, validity=["L >= 10"]), law("ok", 0.5, 0.0, 2, ["s >= 5"])])
    assert r.evidence[0].status == "invalid_domain" and "not evaluable" in r.evidence[0].reason
    assert r.decided_by == "ok"
    r = resolve(C, X, [law("slender_only", 0.5, 0.0, validity=["s >= 20"])])
    assert r.verdict == "insufficient_evidence" and "outside validity" in r.evidence[0].reason


def test_disjoint_intervals_are_a_conflict_not_a_decision():
    r = resolve(C, X, [law("a", 0.5, 0.01), law("b", 0.8, 0.01, 2)], exhaustive=True)
    assert r.verdict == "conflict" and r.conflicts == [("a", "b")] and r.decided_by is None
    r = resolve(C, X, [law("a", 0.5, 0.01)], measurements=[("lab", Est(0.9, "mm", 0.0, 0.05))])
    assert r.verdict == "conflict"
    with pytest.raises(UnitError):
        resolve(C, X, [], measurements=[("lab", Est(0.9, "m", 0.0, 0.05))])


def test_calibration_learns_bound_and_box_and_refuses_outside():
    ref = Law("ref", "q", "mm", "numerical", 9, lambda p: Est(1.0 + 0.02 * p["h"].to("mm"), "mm", 0.001, 0.0),
              [], {"s": "L / h"})
    cheap = Law("cheap", "q", "mm", "analytic", 1, lambda p: Est(1.0, "mm", 0.0, None), [], {"s": "L / h"})
    pts = [point(L=(100, "mm"), h=(h, "mm")) for h in (5, 10)]
    cal = calibrate(cheap, ref, pts, ("s",))
    assert cal.n == 2 and cal.box == ((10.0, 20.0),) and 0.16 < cal.rel_bound < 0.17
    cc = calibrated(cheap, cal)
    assert cc.check(point(L=(100, "mm"), h=(8, "mm")))[0]
    ok, why = cc.check(point(L=(100, "mm"), h=(2, "mm")))
    assert not ok and "s <=" in why


def test_second_domain_with_no_new_machinery():
    q = dict(rho=(998, "kg/m^3"), mu=(1.002e-3, "Pa*s"), D=(0.05, "m"), L=(10, "m"), eps=(0, "m"))
    c = Claim("pipe.pressure_drop", "<=", 300.0, "Pa")
    assert resolve(c, point(V=(0.03, "m/s"), **q), pipe_laws()).decided_by == "hagen_poiseuille"
    assert resolve(c, point(V=(0.5, "m/s"), **q), pipe_laws()).decided_by == "colebrook"
    assert resolve(c, point(V=(0.06, "m/s"), **q), pipe_laws()).verdict == "insufficient_evidence"


def test_beam_hierarchy_against_real_fem():
    p = point(L=(100, "mm"), b=(5, "mm"), h=(10, "mm"), P=(100, "N"), E=(200, "GPa"), nu=(0.3, "1"))
    r = resolve(Claim("beam.tip_deflection", "<=", 1.0, "mm"), p, beam_laws(), exhaustive=True)
    assert r.verdict == "pass" and r.decided_by == "fem_p2_gci" and not r.conflicts
    fem = next(e for e in r.evidence if e.source == "fem_p2_gci").estimate
    assert fem.converged and 0.39 < fem.value < 0.41 and fem.numerical_error < 1e-3
    assert r.digest() == resolve(Claim("beam.tip_deflection", "<=", 1.0, "mm"), p, beam_laws(),
                                 exhaustive=True).digest()
