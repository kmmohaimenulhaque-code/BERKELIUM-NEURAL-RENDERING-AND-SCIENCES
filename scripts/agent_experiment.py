"""Experiment E4 (ADR-022): does ClaimEnv separate grounded engineering behaviour from fluent-but-ungrounded
behaviour, and does knowledge PROMOTED INTO MEMORY (E3) measurably change what a policy can do?

Tasks: every verified FEM point in engineering memory (E3) x claim margins; truth = verified FEM verdict.
Tools: Euler-Bernoulli (no error bound), the discovered relation RECONSTRUCTED FROM ITS MEMORY RECORD, FEM (cached
evidence). Writes docs/experiments/agent_e4.json and trajectories to docs/experiments/trajectories_e4.jsonl
(training data for SFT / preference learning: every step is labelled by the verified outcome).
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
from berkelium.agent import AlwaysHighest, CheapestSufficient, Overconfident, Task, run_episode  # noqa: E402
from berkelium.evidence import Claim, Est, Law  # noqa: E402
from berkelium.evidence.library import _eb, point  # noqa: E402
from berkelium.memory import EngineeringMemory  # noqa: E402

MEM = EngineeringMemory("docs/experiments/memory_e3.jsonl")
MARGINS = (0.003, -0.003, 0.01, -0.01, 0.03, -0.03)


def tools(with_memory: bool):
    ev = {(r.key["slenderness"], r.key["aspect"], r.key["nu"]): r.value for r in MEM.query("evidence")}

    def key(p):
        return (round(p["L"].to("mm") / p["h"].to("mm"), 3), round(p["h"].to("mm") / p["b"].to("mm"), 3),
                round(p["nu"].to("1"), 3))

    def fem(p):
        v = ev[key(p)]
        return Est(v["delta_mm"], "mm", v["err_mm"], 0.0, v["converged"])
    out = [Law("euler_bernoulli", "beam.tip_deflection", "mm", "analytic", 1.0, _eb, ["L/h >= 5"]),
           Law("fem_p2_gci", "beam.tip_deflection", "mm", "numerical", 1000.0, fem)]
    rel = MEM.query("relation", lambda r: r.key["quantity"] == "beam.tip_deflection")
    if with_memory and rel:
        v = rel[0].value
        terms, coef, box, bound = v["terms"], v["coef"], v["validity_box"], v["model_form_rel"]

        def disc(p):
            g = [p["h"].to("mm") / p["L"].to("mm"), p["h"].to("mm") / p["b"].to("mm"), p["nu"].to("1")]
            y = sum(c * np.prod([gi ** e for gi, e in zip(g, t, strict=True)]) for c, t in zip(coef, terms, strict=True))
            d = _eb(p).value * (1 + y)
            return Est(d, "mm", 0.0, bound * d)
        out.append(Law(f"memory:{rel[0].id[:12]}", "beam.tip_deflection", "mm", "surrogate", 2.0, disc,
                       [f"h/L >= {box['s'][0]!r}", f"h/L <= {box['s'][1]!r}", f"h/b >= {box['a'][0]!r}",
                        f"h/b <= {box['a'][1]!r}", f"nu >= {box['nu'][0]!r}", f"nu <= {box['nu'][1]!r}"]))
    return out, ev


def tasks(with_memory: bool):
    ts, ev = tools(with_memory)
    out = []
    for (s, a, nu), v in sorted(ev.items()):
        h = 100.0 / s
        p = point(L=(100.0, "mm"), b=(h / a, "mm"), h=(h, "mm"), P=(100.0, "N"), E=(200.0, "GPa"), nu=(nu, "1"))
        for m in MARGINS:
            c = Claim("beam.tip_deflection", "<=", v["delta_mm"] * (1 + m), "mm")
            lo, hi = v["delta_mm"] - v["err_mm"], v["delta_mm"] + v["err_mm"]
            truth = c.judge((lo, hi))
            truth = truth if truth != "indeterminate" else "insufficient_evidence"
            out.append(Task(f"beam_s{s}_a{a}_nu{nu}_m{m}", c, p, ts, truth, family=f"nu={nu}"))
    return out


def main():
    rep, traj_path = {}, Path("docs/experiments/trajectories_e4.jsonl")
    lines = []
    for mem in (False, True):
        T = tasks(mem)
        for pol in (AlwaysHighest(), CheapestSufficient(), Overconfident()):
            agg = defaultdict(float)
            outcomes = defaultdict(int)
            for t in T:
                ep = run_episode(t, pol)
                agg["return"] += ep["return"]
                agg["cost"] += ep["cost"]
                outcomes[ep["outcome"]] += 1
                ep["memory"] = mem
                lines.append(json.dumps(ep, sort_keys=True, default=str))
            n = len(T)
            rep[f"{pol.name}|memory={mem}"] = {"tasks": n, "mean_return": agg["return"] / n,
                                               "mean_cost": agg["cost"] / n, "outcomes": dict(outcomes)}
    traj_path.write_text("\n".join(lines) + "\n")
    rep["truth_distribution"] = dict(__import__("collections").Counter(t.truth for t in tasks(True)))
    rep["trajectories"] = {"file": str(traj_path), "episodes": len(lines)}
    Path("docs/experiments/agent_e4.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
