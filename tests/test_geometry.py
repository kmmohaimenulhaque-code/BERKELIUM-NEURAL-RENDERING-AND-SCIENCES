import math

import pytest

from berkelium.geometry.backend import (
    OperationFailed,
    UnsupportedConversion,
    UnsupportedOperation,
    plan_backend,
)
from berkelium.geometry.ir import GeometryGraph, GeometryIRError

occt = pytest.importorskip("berkelium.geometry.occt")
mfb = pytest.importorskip("berkelium.geometry.manifold_backend")


def G(ops, outputs):
    return GeometryGraph.model_validate({"ops": ops, "outputs": outputs})


def test_ir_structure_errors():
    with pytest.raises(ValueError):
        G([{"id": "a", "op": "box", "size": [1, 1, 1]}], {"o": "missing"})
    with pytest.raises(ValueError):  # extrude needs a profile, not a solid
        G([{"id": "a", "op": "box", "size": [1, 1, 1]}, {"id": "e", "op": "extrude", "profile": "a", "height": 1}],
          {"o": "e"})
    with pytest.raises(ValueError):
        G([{"id": "a", "op": "box", "size": [1, 1, 1]}, {"id": "a", "op": "sphere", "radius": 1}], {"o": "a"})


def test_expression_resolution():
    g = G([{"id": "c", "op": "cylinder", "radius": "d/2", "height": "2*h"},
           {"id": "r", "op": "rotate", "input": "c", "angle": "360/n"}], {"o": "r"})
    from berkelium.units import Quantity
    r = g.resolve({"d": Quantity.of(20, "mm"), "h": Quantity.of(0.5, "cm"), "n": 8})
    ops = r.by_id()
    assert ops["c"].radius == pytest.approx(10) and ops["c"].height == pytest.approx(10)
    assert ops["r"].angle == pytest.approx(45)
    r2 = G([{"id": "c", "op": "cylinder", "radius": 1, "height": 1},
            {"id": "r", "op": "rotate", "input": "c", "angle": "0.5[rad]"}], {"o": "r"}).resolve({})
    assert r2.by_id()["r"].angle == pytest.approx(math.degrees(0.5))
    with pytest.raises(GeometryIRError):
        g.resolve({})


CUBE_MINUS_CYL = [{"id": "b", "op": "box", "size": [20, 20, 10], "center": True},
                  {"id": "c", "op": "cylinder", "radius": 5, "height": 20, "center": True},
                  {"id": "d", "op": "difference", "base": "b", "tools": ["c"]}]


def test_occt_exact_measurements():
    b = occt.OCCTBackend()
    r = b.execute(G(CUBE_MINUS_CYL, {"o": "d"}))
    m = b.measure(r.bodies["o"])
    assert m.volume_mm3 == pytest.approx(4000 - math.pi * 25 * 10, rel=1e-9)
    assert m.n_solids == 1 and m.closed and m.valid
    assert m.bbox_min == pytest.approx((-10, -10, -5), abs=1e-6)


def test_manifold_agrees_within_tolerance():
    g = G(CUBE_MINUS_CYL, {"o": "d"})
    mb, ob = mfb.ManifoldBackend(), occt.OCCTBackend()
    vm = mb.measure(mb.execute(g).bodies["o"]).volume_mm3
    vo = ob.measure(ob.execute(g).bodies["o"]).volume_mm3
    assert vm == pytest.approx(vo, rel=2e-3)


def test_revolve_axis_convention_matches():
    prof = [{"id": "p", "op": "profile", "start": [2, 0], "segments": [
        {"kind": "line", "to": [6, 0]}, {"kind": "line", "to": [6, 4]}, {"kind": "line", "to": [2, 4]},
        {"kind": "line", "to": [2, 0]}]}, {"id": "r", "op": "revolve", "profile": "p"}]
    g = G(prof, {"o": "r"})
    for b in (occt.OCCTBackend(), mfb.ManifoldBackend()):
        m = b.measure(b.execute(g).bodies["o"])
        assert m.volume_mm3 == pytest.approx(math.pi * (36 - 4) * 4, rel=3e-3)  # mesh: chord-tolerance polygon
        assert m.bbox_min[1] == pytest.approx(0, abs=1e-6) and m.bbox_max[1] == pytest.approx(4, abs=1e-6)


def test_patterns_and_deterministic_step():
    g = G([{"id": "c", "op": "cylinder", "radius": 1, "height": 2},
           {"id": "t", "op": "translate", "input": "c", "offset": [10, 0, 0]},
           {"id": "p", "op": "polar_pattern", "input": "t", "count": 6}], {"o": "p"})
    b = occt.OCCTBackend()
    body = b.execute(g).bodies["o"]
    m = b.measure(body)
    assert m.n_solids == 6 and m.volume_mm3 == pytest.approx(6 * math.pi * 2, rel=1e-9)
    s1, lossy, _ = b.export(body, "step")
    assert not lossy and s1 == b.export(body, "step")[0]
    stl, lossy, tol = b.export(body, "stl")
    assert lossy and tol > 0 and len(stl) > 0


def test_field_planning_and_no_fake_brep():
    g = G([{"id": "f", "op": "implicit", "expr": "sqrt(x^2+y^2+z^2) - 5", "bounds_min": [-6, -6, -6],
            "bounds_max": [6, 6, 6], "voxel": 0.25}], {"o": "f"})
    with pytest.raises(UnsupportedOperation):
        occt.OCCTBackend().execute(g)
    b = plan_backend(g)
    assert b.name == "manifold"
    r = b.execute(g)
    assert r.tolerance_mm >= 0.25 and r.approximations
    assert b.measure(r.bodies["o"]).volume_mm3 == pytest.approx(4 / 3 * math.pi * 125, rel=0.02)
    with pytest.raises(UnsupportedConversion):
        b.export(r.bodies["o"], "step")


def test_typed_failures():
    bad = G([{"id": "p", "op": "profile", "start": [0, 0], "segments": [{"kind": "line", "to": [1, 0]},
                                                                         {"kind": "line", "to": [1, 1]}]},
             {"id": "e", "op": "extrude", "profile": "p", "height": 1}], {"o": "e"})
    with pytest.raises(OperationFailed):
        occt.OCCTBackend().execute(bad)
    with pytest.raises(OperationFailed):
        mfb.ManifoldBackend().execute(bad)
