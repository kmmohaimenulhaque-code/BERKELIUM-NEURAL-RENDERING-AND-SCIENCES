"""ADR-021: engineering memory + relation discovery with an independent, physics-aware promotion gate."""
import numpy as np

from berkelium.discovery import Gate, discover, pi_groups, vanishes_at_zero
from berkelium.memory import EngineeringMemory
from berkelium.units import Quantity


def test_memory_is_content_addressed_append_only_and_persistent(tmp_path):
    m = EngineeringMemory(tmp_path / "mem.jsonl")
    calls = []
    v1, c1 = m.memo({"src": "fem", "x": 1}, lambda: calls.append(1) or {"y": 2.0}, {"producer": "t"})
    v2, c2 = m.memo({"x": 1, "src": "fem"}, lambda: calls.append(1) or {"y": 99.0}, {"producer": "t"})
    assert (v1, c1, c2, len(calls)) == ({"y": 2.0}, False, True, 1)      # same computation never runs twice
    m.put("rejection", {"f": "c*s"}, {"why": "held-out"}, {})
    again = EngineeringMemory(tmp_path / "mem.jsonl")
    assert len(again) == 2 and again.query("rejection")[0].value == {"why": "held-out"}
    try:
        m.put("opinion", {}, {}, {})
        raise AssertionError("unknown kind accepted")
    except ValueError:
        pass


def test_pi_groups_are_dimensionless_and_complete():
    units = {"delta": "mm", "P": "N", "L": "mm", "E": "GPa", "b": "mm", "h": "mm", "nu": "1"}
    gs = pi_groups(units)
    assert len(gs) == 5                                   # 7 variables, dimension-matrix rank 2
    for g in gs:
        q = Quantity.of(1.0)
        for n, e in g.items():
            q = Quantity(q.si * Quantity.of(1.0, units[n]).si ** e,
                         tuple(a + e * b for a, b in zip(q.dim, Quantity.of(1, units[n]).dim, strict=True)))
        assert all(d == 0 for d in q.dim)


def _data(f, rng, n):
    G = np.column_stack([rng.uniform(0.05, 0.2, n), rng.uniform(1, 2, n), rng.uniform(0.2, 0.4, n)])
    return G, f(G)


def test_discovery_recovers_planted_law_and_limits_falsify():
    rng = np.random.default_rng(0)
    law = lambda G: 0.7 * G[:, 0] ** 2 + 0.3 * G[:, 0] * G[:, 2]  # noqa: E731
    Gt, yt = _data(law, rng, 20)
    Gh, yh = _data(law, rng, 8)
    rep = {}
    best, _ = discover(Gt, yt, Gh, yh, ("s", "a", "nu"), limits=("s",), report=rep)
    assert set(best.terms) == {(2, 0, 0), (1, 0, 1)} and np.allclose(sorted(best.coef), [0.3, 0.7])
    assert rep["falsified_by_limits"] > 0 and vanishes_at_zero(best.terms, 0)
    assert not vanishes_at_zero(((2, 0, 0), (0, -1, 3)), 0)


def test_gate_rejects_poor_or_underevidenced_candidates():
    rng = np.random.default_rng(1)
    law = lambda G: 0.7 * G[:, 0] ** 2  # noqa: E731
    Gt, yt = _data(law, rng, 20)
    Gh, yh = _data(law, rng, 6)
    weak, _ = discover(Gt, yt, Gh, yh, ("s", "a", "nu"), exponents=(-1,), max_terms=1, max_factors=1)
    g = Gate(k=2.0, tol=0.0)
    assert not g.passes(weak, np.full(6, 1e-4), 20) and "held-out" in g.log[0]
    good, _ = discover(Gt, yt, Gh, yh, ("s", "a", "nu"))
    assert Gate().passes(good, np.full(6, 1e-6), 20)
    assert not Gate().passes(good, np.full(2, 1e-6), 20)            # too few held-out points
