"""Experiment E3 (ADR-021): can Berkelium discover a useful relation from its own verified evidence, promote
it only through an independent gate, and thereby resolve future claims more cheaply WITHOUT changing verdicts?

  stage "evidence": verified FEM runs (memoised in engineering memory; resumable within a time budget)
  stage "analyze" : Pi groups -> sparse discovery on train family -> held-out gate -> promote/reject ->
                    capability benchmark on fresh claims (before vs after) -> extrapolation refusal
Writes docs/experiments/discovery_e3.json; memory at docs/experiments/memory_e3.jsonl.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
from berkelium.discovery import Gate, discover, pi_groups  # noqa: E402
from berkelium.evidence import Claim, Est, Law, resolve  # noqa: E402
from berkelium.evidence.library import _eb, _fem, point  # noqa: E402
from berkelium.memory import EngineeringMemory  # noqa: E402

MEM = EngineeringMemory("docs/experiments/memory_e3.jsonl")
P, E, L = 100.0, 200.0, 100.0
DOE = [(s, a, nu) for nu in (0.2, 0.3, 0.4) for s in (5, 7, 10, 14, 20) for a in (1.0, 2.0)]
rng = np.random.default_rng(3)
BENCH = [(round(float(rng.uniform(5.5, 19)), 3), round(float(rng.uniform(1, 2)), 3),
          round(float(rng.uniform(0.22, 0.38)), 3)) for _ in range(10)]
MARGINS = [0.005, -0.005, 0.01, -0.01, 0.02, -0.02, 0.04, -0.04, 0.003, -0.003]


def pt(s, a, nu):
    h = L / s
    return point(L=(L, "mm"), b=(h / a, "mm"), h=(h, "mm"), P=(P, "N"), E=(E, "GPa"), nu=(nu, "1"))


def fem_ev(s, a, nu):
    key = {"source": "fem_p2_gci@beam.tip_deflection", "P_N": P, "E_GPa": E, "L_mm": L, "slenderness": s,
           "aspect": a, "nu": nu}
    v, cached = MEM.memo(key, lambda: (lambda e: {"delta_mm": e.value, "err_mm": e.numerical_error,
                                                   "converged": e.converged, "note": e.note})(_fem(pt(s, a, nu))),
                         {"producer": "berkelium.evidence.library._fem", "version": "0.1"})
    return v, cached


def stage_evidence(budget=250.0):
    t0, made = time.time(), 0
    for s, a, nu in DOE + BENCH:
        if time.time() - t0 > budget:
            break
        _, c = fem_ev(s, a, nu)
        made += 0 if c else 1
    todo = sum(MEM.get("evidence", {"source": "fem_p2_gci@beam.tip_deflection", "P_N": P, "E_GPa": E, "L_mm": L,
                                    "slenderness": s, "aspect": a, "nu": nu}) is None for s, a, nu in DOE + BENCH)
    print(f"computed {made} new FEM evidence records in {time.time() - t0:.0f}s; remaining {todo}; memory {len(MEM)}")


def rows(points):
    G, y, ye = [], [], []
    for s, a, nu in points:
        v, _ = fem_ev(s, a, nu)
        assert v["converged"], (s, a, nu)
        d_eb = _eb(pt(s, a, nu)).value
        G.append([1 / s, a, nu])
        y.append(v["delta_mm"] / d_eb - 1)
        ye.append(v["err_mm"] / d_eb)
    return np.array(G), np.array(y), np.array(ye)


def stage_analyze(out="docs/experiments/discovery_e3.json"):
    rep = {"pi_groups": pi_groups({"delta": "mm", "P": "N", "L": "mm", "E": "GPa", "b": "mm", "h": "mm", "nu": "1"}),
           "groups_used": ["h/L", "h/b", "nu"], "target": "delta_FEM / delta_EB - 1"}
    train = [p for p in DOE if p[2] != 0.3]
    held = [p for p in DOE if p[2] == 0.3]
    Gt, yt, _ = rows(train)
    Gh, yh, yhe = rows(held)
    groups = ("s", "a", "nu")
    # Pass 1 (as first run): data-only gate. Kept in the report because it is the failure that motivated pass 2.
    naive, _ = discover(Gt, yt, Gh, yh, groups)
    rep["data_only_discovery"] = {"chosen": naive.formula({"s": "h/L", "a": "h/b", "nu": "nu"}),
                                  "heldout_max_abs": naive.heldout_max_abs,
                                  "slender_limit_ok": bool(__import__("berkelium.discovery", fromlist=["x"])
                                                           .vanishes_at_zero(naive.terms, 0)),
                                  "verdict": "would pass the data gate, but violates the slender limit "
                                             "(Euler-Bernoulli must become exact as h/L -> 0)"}
    lim = {}
    best, cands = discover(Gt, yt, Gh, yh, groups, limits=("s",), report=lim)
    rep["limit_check"] = lim
    gate = Gate(k=2.0, tol=1e-4)
    ok = gate.passes(best, yhe, len(train))
    rep["discovery"] = {"n_candidates": len(cands), "chosen": best.formula({"s": "h/L", "a": "h/b", "nu": "nu"}),
                        "complexity": best.complexity, "train_rmse": best.train_rmse,
                        "heldout_max_abs": best.heldout_max_abs, "heldout_rmse": best.heldout_rmse,
                        "max_heldout_evidence_err": float(yhe.max()),
                        "top5": [(c.formula({"s": "h/L", "a": "h/b", "nu": "nu"}), c.heldout_rmse, c.complexity)
                                 for c in cands[:5]],
                        "timoshenko_theory": "(0.6 + 0.55*nu)*(h/L)^2 (Cowper kappa)", "promoted": ok,
                        "gate_log": list(gate.log)}
    # negative control: a deliberately impoverished hypothesis space must be rejected by the same gate
    weak, _ = discover(Gt, yt, Gh, yh, groups, exponents=(1,), max_terms=1, max_factors=1, limits=("s",))
    g2 = Gate(k=2.0, tol=1e-4)
    rep["negative_control"] = {"candidate": weak.formula({"s": "h/L", "a": "h/b", "nu": "nu"}),
                               "promoted": g2.passes(weak, yhe, len(train)), "gate_log": list(g2.log)}
    G_all = np.vstack([Gt, Gh])
    box = {g: (float(G_all[:, i].min()), float(G_all[:, i].max())) for i, g in enumerate(groups)}
    resid = np.abs(np.concatenate([best.predict(Gt) - yt, best.predict(Gh) - yh]))
    bound = float(resid.max() + max(yhe.max(), rows(train)[2].max()))
    if ok:
        MEM.put("relation", {"quantity": "beam.tip_deflection", "formula": best.formula()},
                {"text": f"delta_3d == delta_eb*(1 + {best.formula({'s': 'h/L', 'a': 'h/b', 'nu': 'nu'})})",
                 "validity_box": box, "model_form_rel": bound, "groups": groups, "terms": best.terms,
                 "coef": best.coef, "evidence_ids": sorted(MEM.rid("evidence", {
                     "source": "fem_p2_gci@beam.tip_deflection", "P_N": P, "E_GPa": E, "L_mm": L, "slenderness": s,
                     "aspect": a, "nu": nu}) for s, a, nu in DOE)},
                {"producer": "berkelium.discovery", "gate": f"k={gate.k}, tol={gate.tol}", "train": "nu in {0.2,0.4}",
                 "heldout": "nu = 0.3"})
    else:
        MEM.put("rejection", {"quantity": "beam.tip_deflection", "formula": best.formula()}, {"log": gate.log}, {})
    rep["promotion"] = {"validity_box": box, "model_form_rel": bound}

    def disc_run(p):
        s, a, nu = 1 / (p["L"].to("mm") / p["h"].to("mm")), p["h"].to("mm") / p["b"].to("mm"), p["nu"].to("1")
        d = _eb(p).value * (1 + float(best.predict(np.array([[s, a, nu]]))[0]))
        return Est(d, "mm", 0.0, bound * d, note="discovered relation (promoted)")
    in_box = [f"h/L >= {box['s'][0]!r}", f"h/L <= {box['s'][1]!r}", f"h/b >= {box['a'][0]!r}",
              f"h/b <= {box['a'][1]!r}", f"nu >= {box['nu'][0]!r}", f"nu <= {box['nu'][1]!r}"]
    disc = Law("discovered_beam", "beam.tip_deflection", "mm", "surrogate", 2.0, disc_run, in_box)
    eb = Law("euler_bernoulli", "beam.tip_deflection", "mm", "analytic", 1.0, _eb, ["L/h >= 5"])

    def fem_cached(p):
        s = round(p["L"].to("mm") / p["h"].to("mm"), 3)
        a, nu = round(p["h"].to("mm") / p["b"].to("mm"), 3), round(p["nu"].to("1"), 3)
        v, _ = fem_ev(s, a, nu)
        return Est(v["delta_mm"], "mm", v["err_mm"], 0.0, v["converged"])
    fem = Law("fem_p2_gci", "beam.tip_deflection", "mm", "numerical", 1000.0, fem_cached)
    bench = []
    for (s, a, nu), m in zip(BENCH, MARGINS, strict=True):
        p = pt(s, a, nu)
        truth_d = fem_cached(p)
        claim = Claim("beam.tip_deflection", "<=", truth_d.value * (1 + m), "mm")
        truth = resolve(claim, p, [fem])
        before = resolve(claim, p, [eb, fem])
        after = resolve(claim, p, [eb, disc, fem])
        n_fem = lambda r: sum(e.source == "fem_p2_gci" and e.status == "evaluated" for e in r.evidence)  # noqa: E731
        bench.append({"point": [s, a, nu], "margin": m, "truth": truth.verdict, "before": before.verdict,
                      "after": after.verdict, "after_by": after.decided_by, "fem_before": n_fem(before),
                      "fem_after": n_fem(after)})
    rep["benchmark"] = {"cases": bench, "fem_runs_before": sum(b["fem_before"] for b in bench),
                        "fem_runs_after": sum(b["fem_after"] for b in bench),
                        "verdict_disagreements": sum(b["after"] != b["truth"] for b in bench if b["truth"] in ("pass", "fail")),
                        "conflicts_after": sum(b["after"] == "conflict" for b in bench)}
    r = resolve(Claim("beam.tip_deflection", "<=", 1.0, "mm"), pt(3.0, 1.0, 0.3), [eb, disc])
    rep["extrapolation_refused"] = {"verdict": r.verdict, "reasons": [(e.source, e.reason) for e in r.evidence]}
    rep["memory_records"] = {k: len(MEM.query(k)) for k in MEM.KINDS}
    Path(out).write_text(json.dumps(rep, indent=2, default=list) + "\n")
    print(json.dumps({k: rep[k] for k in ("data_only_discovery", "limit_check", "negative_control", "promotion")},
                     indent=1, default=list)[:3000])
    print("chosen:", rep["discovery"]["chosen"], "| heldout max", rep["discovery"]["heldout_max_abs"],
          "| promoted", rep["discovery"]["promoted"], rep["discovery"]["top5"][:3])
    print(json.dumps({k: v for k, v in rep["benchmark"].items() if k != "cases"}), rep["extrapolation_refused"]["verdict"])


if __name__ == "__main__":
    stage_evidence() if sys.argv[1] == "evidence" else stage_analyze()
