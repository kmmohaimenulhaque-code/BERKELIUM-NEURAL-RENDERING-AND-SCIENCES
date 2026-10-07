"""Experiment E2 (ADR-020): are acausal law networks a sufficient substrate for CEM-style engineering
across domains? Deterministic; writes docs/experiments/laws_e2.json.

Each check compares the domain-free machinery against an INDEPENDENT reference: published tables, closed
forms evaluated directly, the hand-written CEM, or verified FEM.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())

from berkelium.evidence import Claim, Est, Law, resolve
from berkelium.laws import LawSet, Relation, Var, as_law, optimise, propagate, sensitivities, solve
from berkelium.laws.library import beam, nozzle, rocket_chamber, vessel, wall
from berkelium.units import Quantity as Q


def q(v, u="1"):
    return Q.of(v, u)


def main(out="docs/experiments/laws_e2.json", fem=True):
    t0, rep = time.time(), {}
    # A. reference values ------------------------------------------------------------------------------
    s = solve(nozzle(), {"gamma": q(1.4), "M_e": q(2.0), "p0": q(1, "MPa"), "T0": q(3000, "K"),
                         "R": q(287, "J/(kg*K)"), "At": q(100, "mm^2")})
    from berkelium.cem.library.beam.cem import CantileverBeamCEM, Requirements
    cem = CantileverBeamCEM().derive(Requirements(tip_load_N=200, length_mm=100, max_tip_deflection_mm=0.5,
                                                  allowable_stress_MPa=150, youngs_modulus_GPa=207))
    b_opt, info = optimise(beam(), {"P": q(200, "N"), "L": q(100, "mm"), "E": q(207, "GPa"), "a": q(2),
                                    "rho": q(7800, "kg/m^3"), "delta_max": q(0.5, "mm"), "sigma_allow": q(150, "MPa")},
                           {"h": (4.0, 40.0)}, "mass")
    h_closed = max((4 * 200 * 0.1 ** 3 * 2 / (207e9 * 0.5e-3)) ** 0.25, (6 * 200 * 0.1 * 2 / 150e6) ** (1 / 3)) * 1e3
    rep["A_references"] = {
        "nozzle_area_ratio_M2": {"computed": s.get("eps", "1"), "NACA1135": 1.6875},
        "nozzle_pressure_ratio_M2": {"computed": s.get("pe", "kPa") / 1000, "NACA1135": 0.1278},
        "beam_min_mass_height_mm": {"computed": b_opt.get("h", "mm"), "closed_form": h_closed,
                                    "hand_written_cem_unrounded": [d.value.value for d in cem.derivations],
                                    "active": info["active_constraints"], "evaluations": info["evaluations"]}}
    # B. inversion and ambiguity -------------------------------------------------------------------------
    inv = {"gamma": q(1.4), "eps": q(1.6875), "p0": q(1, "MPa"), "T0": q(3000, "K"), "R": q(287, "J/(kg*K)"),
           "At": q(100, "mm^2")}
    amb = solve(nozzle(), inv)
    nz = nozzle()
    nz.vars["M_e"] = Var("M_e", "1", 1.0)                 # declare: supersonic exit
    sup = solve(nz, inv)
    rep["B_inversion"] = {"unbounded": {"status": amb.status, "roots": amb.roots.get("M_e")},
                          "supersonic_declared": {"status": sup.status, "M_e": sup.get("M_e", "1")}}
    # C. specification diagnostics -----------------------------------------------------------------------
    under = solve(beam(), {"P": q(200, "N"), "L": q(100, "mm"), "E": q(207, "GPa"), "rho": q(7800, "kg/m^3")})
    over = solve(beam(), {"P": q(200, "N"), "L": q(100, "mm"), "E": q(207, "GPa"), "a": q(2), "b": q(5, "mm"),
                          "h": q(12, "mm"), "rho": q(7800, "kg/m^3")})
    rep["C_diagnostics"] = {"underdetermined": {"status": under.status, "msg": under.diagnostics},
                            "inconsistent": {"status": over.status, "msg": over.diagnostics}}
    # D. sensitivities (exact elasticities known for the beam) -------------------------------------------
    bk = {"P": q(200, "N"), "L": q(100, "mm"), "E": q(207, "GPa"), "b": q(5, "mm"), "h": q(10, "mm"),
          "rho": q(7800, "kg/m^3")}
    bb = beam()
    bb.relations = [r for r in bb.relations if r.id != "aspect"]
    el = sensitivities(bb, bk, ["delta", "sigma"])
    rep["D_sensitivities"] = {"computed": el, "exact_delta": {"P": 1, "L": 3, "E": -1, "b": -1, "h": -3},
                              "exact_sigma": {"P": 1, "L": 1, "b": -1, "h": -2}}
    # E. uncertainty -----------------------------------------------------------------------------------
    u = propagate(bb, bk, {"E": 0.05, "h": 0.02}, ["delta"], n=400)["delta"]
    rep["E_uncertainty"] = {"rel_sigma_linear": u["sigma_linear_si"] / u["value_si"],
                            "rel_sigma_mc": u["mc_std_si"] / u["value_si"],
                            "analytic_first_order": math.hypot(0.05, 3 * 0.02), "mc_failed": u["mc_failed"]}
    # F. cross-domain composition -------------------------------------------------------------------------
    rc = rocket_chamber()
    rck = {"gamma": q(1.2), "R": q(350, "J/(kg*K)"), "T0": q(3200, "K"), "p0": q(2, "MPa"), "pe": q(101.325, "kPa"),
           "F": q(1000, "N"), "CR": q(4), "nu": q(0.3), "S_allow": q(150, "MPa"), "n_sf": q(2),
           "rho_s": q(7800, "kg/m^3")}
    rsol, rinfo = optimise(rc, rck, {"ro": (5.0, 200.0)}, "mass_per_len")
    rep["F_rocket_chamber"] = {
        "status": rsol.status, "plan": [b for b, _ in rsol.plan],
        "values": {k: round(rsol.get(k, u2), 6) for k, u2 in (("M_e", "1"), ("d_t", "mm"), ("d_e", "mm"),
                   ("mdot", "kg/s"), ("ri", "mm"), ("ro", "mm"), ("w", "mm"), ("svm", "MPa"))},
        "active": rinfo["active_constraints"], "relations": len(rc.relations), "new_relations_for_composition": 2}
    tw = solve(wall(), {"T_in": q(473.15, "K"), "T_amb": q(298.15, "K"), "T_s": q(318.15, "K"),
                        "k_ins": q(0.04, "W/(m*K)"), "h_o": q(10, "W/(m^2*K)")})
    rep["F_wall_insulation_mm"] = {"computed": tw.get("t_ins", "mm"),
                                   "closed_form": 0.04 * ((473.15 - 298.15) / (10 * 20) - 1 / 10) * 1e3}
    # G. evidence bridge + falsification against verified FEM ----------------------------------------------
    if fem:
        from berkelium.physics.run import run_case
        from berkelium.physics.schema import AnalysisCase, Material
        from tests._geom import quarter_tube_step

        def fem_run(p):
            mat = Material(id="m", source="E2", elastic={"youngs_modulus": {"value": 200, "unit": "GPa"},
                                                         "poisson_ratio": p["nu"].to("1")})
            c = AnalysisCase(id="lame", physics="structural_linear_static", target="t", material="m",
                             regions=[{"id": "in", "field": "sqrt(x^2+y^2) - 10.01"}, {"id": "sx", "field": "abs(x) - 0.001"},
                                      {"id": "sy", "field": "abs(y) - 0.001"}, {"id": "z0", "field": "abs(z) - 0.001"},
                                      {"id": "zH", "field": "abs(z - 10) - 0.001"}],
                             conditions=[{"kind": "fixed", "region": "sx", "components": ["x"]},
                                         {"kind": "fixed", "region": "sy", "components": ["y"]},
                                         {"kind": "fixed", "region": "z0", "components": ["z"]},
                                         {"kind": "fixed", "region": "zH", "components": ["z"]},
                                         {"kind": "pressure", "region": "in", "magnitude": {"value": p["p"].to("MPa"), "unit": "MPa"}}],
                             outputs=[{"id": "vm", "kind": "average", "field": "von_mises", "region": "in", "unit": "MPa"}],
                             mesh={"generator": "gmsh", "element_order": 2, "size": {"value": 2.5, "unit": "mm"}},
                             convergence={"levels": 3, "refinement_ratio": 1.5})
            r = run_case(c, {"m": mat}, step=quarter_tube_step(p["ri"].to("mm"), p["ro"].to("mm"), 10))
            e = r.estimates[0]
            return Est(e.value, "MPa", e.discretization_error or 0.0, 0.0, e.status == "converged")
        fem_law = Law("fem_lame_quarter", "vessel.svm", "MPa", "numerical", 1000, fem_run)
        exact = as_law(vessel(), "svm", "MPa", "vessel.svm", id="lame_network")
        thin = vessel()
        thin.relations = [Relation("thin_wall_hoop", "sig_t == p*ri/(ro - ri)", references=("thin-wall formula",))
                          if r.id == "lame_hoop_inner" else r for r in thin.relations]
        thin = LawSet("thin_wall_misused", thin.vars, thin.relations)
        wrong_nu = lambda p: exact.run({**p, "nu": q(0.45)})  # noqa: E731
        pt = {"p": q(10, "MPa"), "ri": q(10, "mm"), "ro": q(20, "mm"), "nu": q(0.3)}
        laws = [exact, as_law(thin, "svm", "MPa", "vessel.svm", id="thin_wall_misused", cost=0.5),
                Law("lame_wrong_nu", "vessel.svm", "MPa", "analytic", 0.6, wrong_nu), fem_law]
        r = resolve(Claim("vessel.svm", "<=", 25.0, "MPa"), pt, laws, exhaustive=True)
        rep["G_falsification"] = {"verdict": r.verdict, "conflicts": r.conflicts,
                                  "estimates": {e.source: (e.estimate.value, e.estimate.interval()) for e in r.evidence
                                                if e.estimate}}
    rep["seconds"] = round(time.time() - t0, 1)
    Path(out).write_text(json.dumps(rep, indent=2, default=str) + "\n")
    print(json.dumps(rep, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []), fem="--no-fem" not in sys.argv)
