import math

import pytest

from berkelium.cem.library.gear import formulas as F
from berkelium.cem.library.gear.cem import Parameters, Requirements
from berkelium.cem.library.gear.profile import generate_flank
from berkelium.cem.protocol import default_registry
from berkelium.designs.procedural import lantern_proposal
from berkelium.pipeline import ArtifactStore, run

CEM = default_registry().get("gear.spur_pair@0.1")
A20 = math.radians(20)


def statuses(rs, target=None):
    return {(r.target, r.fidelity.method): r.status for r in rs if target is None or r.target == target}


@pytest.mark.parametrize("m,z,x", [(2, 20, 0.0), (3, 12, 0.6), (1, 40, -0.3), (2, 18, 0.0)])
def test_generated_flank_is_exact_involute_with_correct_thickness(m, z, x):
    fl = generate_flank(m, z, x, A20)
    rb = F.base_diameter(m, z, A20) / 2
    th = [math.atan2(p[1], p[0]) for p in fl.involute]
    rr = [math.hypot(*p) for p in fl.involute]
    for t, r in zip(th, rr):  # theta + inv(alpha_r) is constant along an involute
        assert t + F.inv(math.acos(rb / r)) == pytest.approx(th[0] + F.inv(math.acos(rb / rr[0])), abs=1e-12)
    assert fl.r_root == pytest.approx(F.root_diameter(m, z, x) / 2, abs=1e-9)
    assert fl.r_tip == pytest.approx(F.tip_diameter(m, z, x) / 2)
    radii = [math.hypot(*p) for p in fl.fillet + fl.involute]
    assert all(b >= a - 1e-9 for a, b in zip(radii, radii[1:]))  # no undercut loop


def test_derive_hits_centre_distance_with_profile_shift():
    d = CEM.derive(Requirements(ratio=2, module=3, center_distance=56.5, min_pinion_teeth=12))
    p = d.parameters
    assert (p.z1, p.z2) == (12, 24) and p.x1 + p.x2 == pytest.approx(0.96, abs=1e-3)
    assert CEM.quantities(p)["center_distance"].to("mm") == pytest.approx(56.5, abs=1e-9)
    assert any("split equally" in a for a in d.assumptions)


def test_derive_lewis_module_selection():
    p = CEM.derive(Requirements(ratio=3.1, pinion_torque=5e4, pinion_speed=1200, allowable_bending_stress=120)).parameters
    lewis = [r for r in CEM.check(p) if r.fidelity.method == "lewis_barth_preliminary"]
    assert lewis and all(r.status == "pass" for r in lewis)
    smaller = Parameters(**(p.model_dump() | {"module": 2.0, "face_width": 8 * math.pi}))
    assert any(r.status == "fail" for r in CEM.check(smaller) if r.fidelity.method == "lewis_barth_preliminary")


def test_checks_flag_undercut_and_pointed_teeth():
    s = statuses(CEM.check(Parameters(module=2, z1=12, z2=40, face_width=25)))
    assert s[("pair.pinion", "rack_cutter_undercut")] == "fail"
    s = statuses(CEM.check(Parameters(module=2, z1=12, z2=40, x1=1.3, face_width=25)))
    assert s[("pair.pinion", "involute_tooth_thickness")] == "fail"


def test_lewis_not_evaluated_without_load_or_with_shift():
    rs = CEM.check(Parameters(module=2, z1=20, z2=40, face_width=25))
    assert {r.status for r in rs if r.fidelity.method == "lewis_barth_preliminary"} == {"not_evaluated"}
    rs = CEM.check(Parameters(module=2, z1=20, z2=40, x1=0.2, face_width=25, pinion_torque=1e4, pinion_speed=100))
    assert {r.status for r in rs if r.fidelity.method == "lewis_barth_preliminary"} == {"not_evaluated"}


@pytest.mark.parametrize("mode", ["valid", "boundary", "invalid"])
def test_sampler_modes_are_honest(mode):
    import numpy as np
    rng = np.random.default_rng(7)
    for _ in range(5):
        p = CEM.sample(rng, mode)
        has_fail = any(r.status == "fail" for r in CEM.check(p))
        assert has_fail == (mode == "invalid")


GEAR_PROPOSAL = {
    "specification": {"requirements": [
        {"id": "r_cd", "quantity": "center_distance", "applies_to": "pair", "comparator": "==",
         "target": {"value": 56.5, "unit": "mm"}, "tolerance": {"value": 0.01, "unit": "mm"}}]},
    "structure": {"components": [{"id": "pair", "kind": "cem", "cem": "gear.spur_pair@0.1", "requirements": {
        "ratio": 2, "module": {"value": 3, "unit": "mm"}, "center_distance": {"value": 5.65, "unit": "cm"},
        "min_pinion_teeth": 12}}]}}


@pytest.mark.occt
def test_full_pipeline_gear_pair(tmp_path):
    pytest.importorskip("berkelium.geometry.occt")
    res = run(GEAR_PROPOSAL, artifacts=ArtifactStore(tmp_path))
    rec = res.record
    by = {(r.target, r.level, r.fidelity.method): r.status for r in rec.evaluation.validation.results}
    assert by[("pair.mesh", 4, "boolean_common@occt")] == "pass"
    assert by[("r_cd", 2, "requirement_compare")] == "pass"
    assert rec.evaluation.validation.summary in ("pass", "warn")
    assert rec.evaluation.validation.physically_validated is False
    assert any(r.level == 7 and r.status == "not_evaluated" for r in rec.evaluation.validation.results)
    steps = [a for a in rec.evaluation.artifacts if a.format == "step"]
    assert len(steps) == 2 and not any(a.lossy for a in steps)
    assert all(a.lossy for a in rec.evaluation.artifacts if a.format == "glb")
    again = run(GEAR_PROPOSAL, artifacts=ArtifactStore(tmp_path)).record
    assert again.content_hash == rec.content_hash      # deterministic record and artifacts


@pytest.mark.occt
def test_mesh_interference_positive_control():
    pytest.importorskip("berkelium.geometry.occt")
    from berkelium.cem.protocol import GeometryContext
    from berkelium.geometry.ir import GeometryGraph
    from berkelium.geometry.occt import OCCTBackend
    p = Parameters(module=2, z1=20, z2=40, face_width=10)
    g = CEM.expand(p).graph.model_dump()
    # rotate the pinion by half a tooth pitch: teeth now collide
    g["ops"].append({"id": "pin_rot", "op": "rotate", "input": "pinion_blank", "angle": 180 / 20})
    g["outputs"]["pinion"] = "pin_rot"
    b = OCCTBackend()
    r = b.execute(GeometryGraph.model_validate(g))
    meas = {k: b.measure(v) for k, v in r.bodies.items()}
    out = CEM.geometry_validators()[0](GeometryContext(b, r.bodies, meas, p))
    assert out[0].status == "fail" and out[0].measured.value > 1.0


def test_pipeline_rejects_invalid_proposal_and_unknown_cem():
    res = run({"structure": {"components": [{"id": "x", "kind": "cem", "cem": "gear.spur_pair@0.1"}]},
               "evaluation": {}})
    assert res.record is None and res.schema_errors
    rec = run({"structure": {"components": [{"id": "x", "kind": "cem", "cem": "rocket.nozzle@0.1",
                                              "requirements": {"thrust": 1}}]}}, realize=False).record
    assert rec.evaluation.validation.summary == "fail"
    assert rec.evaluation.diagnostics[0].code == "UNKNOWN_CEM"


def test_procedural_design_both_backends_agree():
    vols = {}
    for be in ("occt", "manifold"):
        rec = run(lantern_proposal(), backend=be).record
        assert rec.evaluation.validation.summary == "pass"
        vols[be] = next(m.value.value for m in rec.evaluation.measurements if m.name == "volume")
    assert vols["occt"] == pytest.approx(vols["manifold"], rel=2e-3)
