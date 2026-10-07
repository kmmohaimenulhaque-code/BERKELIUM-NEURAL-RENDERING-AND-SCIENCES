"""PHYSICS-6/7: natural pipeline path CEM -> geometry -> STEP -> Gmsh -> FEM -> GCI -> L7 -> DesignRecord."""
import pytest

from berkelium.cem.library.beam.cem import CantileverBeamCEM, Requirements, default_analysis
from berkelium.physics.materials import LIBRARY
from berkelium.physics.mesh import gmsh_version
from berkelium.pipeline import run
from berkelium.schema.design import DesignProposal

REQ = dict(tip_load_N=200, length_mm=100, max_tip_deflection_mm=0.5, allowable_stress_MPa=150,
           youngs_modulus_GPa=207, aspect_ratio=2.0)


def proposal(limit_mm: float, analyses=True):
    cem = CantileverBeamCEM()
    p = cem.derive(Requirements(**REQ)).parameters
    case = default_analysis("beam", p, "carbon_steel")
    return DesignProposal.model_validate({
        "intent": {"summary": "cantilever bracket"},
        "specification": {
            "requirements": [{"id": "stiff", "quantity": "sim.beam_elastic.tip_deflection", "comparator": ">=",
                              "target": {"value": -limit_mm, "unit": "mm"}}],
            "materials": [LIBRARY["carbon_steel"].model_dump(mode="json")],
            "analyses": [case.model_dump(mode="json")] if analyses else []},
        "structure": {"components": [{"id": "beam", "kind": "cem", "cem": "structure.cantilever_beam@0.1",
                                      "requirements": REQ}]}})


def test_beam_derivation_meets_its_own_analytic_targets():
    cem = CantileverBeamCEM()
    d = cem.derive(Requirements(**REQ))
    q = cem.quantities(d.parameters)
    assert q["tip_deflection_eb"].to("mm") <= 0.5 and q["root_stress_eb"].to("MPa") <= 150
    assert d.parameters.height_mm == 11.7          # stress-governed: (6 P L a / sigma)^(1/3) = 11.70 mm


def test_record_without_analysis_is_not_physically_validated():
    rec = run(proposal(0.5, analyses=False), realize=False).record
    assert rec.evaluation.validation.physically_validated is False
    assert not rec.evaluation.simulation


def test_model_cannot_smuggle_simulation_results():
    bad = proposal(0.5).model_dump(mode="json")
    bad["evaluation"] = {"simulation": [{"case_id": "x"}]}
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="evaluation"):
        DesignProposal.model_validate(bad)


@pytest.mark.skipif(gmsh_version() is None, reason="gmsh executable not installed")
def test_end_to_end_l7_pass_then_fail(tmp_path):
    from berkelium.pipeline import ArtifactStore
    res = run(proposal(0.5), artifacts=ArtifactStore(tmp_path))
    rec = res.record
    sim = rec.evaluation.simulation[0]
    est = {e.id: e for e in sim.estimates}
    assert sim.status == "converged" and sim.fidelity == "numerical"
    assert abs(est["root_reaction"].value - 200) < 1e-6
    l7 = [r for r in rec.evaluation.validation.results if r.level == 7]
    assert {r.status for r in l7} == {"pass"} and rec.evaluation.validation.physically_validated
    assert sim.fields and (tmp_path / sim.fields[0].sha256[:2]).exists()
    eb = 0.4125   # Euler-Bernoulli for the derived section; 3-D FE includes shear -> slightly larger
    assert 0 < (-est["tip_deflection"].value - eb) / eb < 0.03
    # a requirement inside the numerical error band / beyond the result flips to indeterminate / fail
    d = -est["tip_deflection"].value
    rec2 = run(proposal(d * 0.9), artifacts=ArtifactStore(tmp_path)).record
    req = next(r for r in rec2.evaluation.validation.results if r.validator.startswith("requirement.l7"))
    assert req.status == "fail" and rec2.evaluation.validation.physically_validated is False
