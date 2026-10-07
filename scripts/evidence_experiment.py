"""Experiment E1 (ADR-019): does the evidence calculus cut high-fidelity cost WITHOUT changing decisions,
and does it detect inconsistency? Deterministic; writes docs/experiments/evidence_e1.json.

  A. calibrate Euler-Bernoulli against verified FEM on a design-of-experiments (8 beams)
  B. minimise beam height s.t. tip deflection <= limit: FEM-only bisection vs evidence-resolved bisection;
     final design re-verified by FEM in both
  C. falsification probes: contradicting measurement, mis-implemented law, out-of-domain point
  D. pipe (second domain, same code): laminar / turbulent / transitional => insufficient_evidence
"""
import json
import sys
import time
from pathlib import Path

from berkelium.evidence import Claim, Est, Law, calibrate, calibrated, resolve
from berkelium.evidence.library import beam_laws, pipe_laws, point

E, NU, P, L, A = 200.0, 0.3, 500.0, 200.0, 2.0


def beam(h, L=L, a=A, P=P):
    return point(L=(L, "mm"), b=(h / a, "mm"), h=(h, "mm"), P=(P, "N"), E=(E, "GPa"), nu=(NU, "1"))


def search(laws, limit, lo=10.0, hi=30.0, tol=0.1):
    claim = Claim("beam.tip_deflection", "<=", limit, "mm")
    log, fem = [], 0
    while hi - lo > tol:
        h = round((lo + hi) / 2, 4)
        r = resolve(claim, beam(h), laws)
        fem += sum(e.source.startswith("fem") and e.status == "evaluated" for e in r.evidence)
        log.append({"h": h, "verdict": r.verdict, "decided_by": r.decided_by})
        if r.verdict == "pass":
            hi = h
        elif r.verdict == "fail":
            lo = h
        else:
            raise RuntimeError(f"search stopped: {r.verdict} at h={h}")
    final = resolve(claim, beam(hi), laws, verify_with="fem_p2_gci", exhaustive=True)
    fem += 1
    return {"h_min_mm": hi, "steps": log, "fem_runs": fem, "final_verdict": final.verdict,
            "final_conflicts": final.conflicts}


def main(out="docs/experiments/evidence_e1.json"):
    t0 = time.time()
    laws = beam_laws()
    eb, fem = laws[0], laws[2]
    eb_g = Law(eb.id, eb.quantity, eb.unit, eb.fidelity, eb.cost, eb.run, eb.validity,
               {**eb.groups, "poisson": "nu"}, eb.references)
    doe = [beam(L / s, a=a) for s in (6, 10, 15, 20) for a in (1.0, 2.0)]
    cal = calibrate(eb_g, fem, doe, ("slenderness", "aspect", "poisson"))
    eb_cal = calibrated(eb_g, cal)
    rep = {"A_calibration": {"n": cal.n, "box": dict(zip(cal.groups, cal.box, strict=True)),
                             "rel_bound": cal.rel_bound, "samples": cal.samples}}
    limit = 1.0
    base = search([fem], limit)
    smart = search([eb_cal, fem], limit)
    rep["B_search"] = {"limit_mm": limit, "fem_only": base, "evidence": smart,
                       "same_answer": abs(base["h_min_mm"] - smart["h_min_mm"]) < 1e-9,
                       "fem_runs_saved": base["fem_runs"] - smart["fem_runs"]}
    c = Claim("beam.tip_deflection", "<=", limit, "mm")
    meas = ("lab_measurement_hypothetical", Est(1.30, "mm", 0.0, 0.02))
    r1 = resolve(c, beam(20.0), laws, measurements=[meas], exhaustive=True)
    bug = Law("eb_with_unit_bug", eb.quantity, "mm", "analytic", 0.5,
              lambda p: Est(eb.run(p).value * 1.25, "mm", 0.0, 0.05 * eb.run(p).value), eb.validity, eb.groups)
    r2 = resolve(c, beam(20.0), [bug, fem], exhaustive=True)
    r3 = resolve(c, beam(70.0, L=200.0), [eb_cal, fem])      # L/h = 2.86: outside EB validity and calibration
    rep["C_falsification"] = {
        "contradicting_measurement": {"verdict": r1.verdict, "conflicts": r1.conflicts},
        "mis_implemented_law": {"verdict": r2.verdict, "conflicts": r2.conflicts},
        "out_of_domain": {"verdict": r3.verdict, "decided_by": r3.decided_by,
                          "skipped": [(e.source, e.reason) for e in r3.evidence if e.status != "evaluated"]}}
    pc = Claim("pipe.pressure_drop", "<=", 300.0, "Pa")
    pipes = {}
    for name, V in (("laminar", 0.03), ("turbulent", 0.5), ("transitional", 0.06)):
        q = point(rho=(998, "kg/m^3"), mu=(1.002e-3, "Pa*s"), D=(0.05, "m"), L=(10, "m"), V=(V, "m/s"), eps=(0, "m"))
        r = resolve(pc, q, pipe_laws())
        pipes[name] = {"verdict": r.verdict, "decided_by": r.decided_by}
    rep["D_pipe"] = pipes
    rep["seconds"] = round(time.time() - t0, 1)
    Path(out).write_text(json.dumps(rep, indent=2, default=list) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "A_calibration"} | {
        "A": {k: rep["A_calibration"][k] for k in ("n", "box", "rel_bound")}}, indent=1, default=list)[:3500])


if __name__ == "__main__":
    main(*sys.argv[1:])
