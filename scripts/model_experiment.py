"""Experiment E5 (ADR-023): can a domain-free constructor BUILD correct models (and correctly refuse) on scenarios
with independent references, including hard negatives, held-out domains and a cross-domain composition — and
how does first-match retrieval fail? Writes docs/experiments/model_e5.json + trajectories_e5.jsonl."""
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.getcwd())
from berkelium.agent.model_env import run_model_episode  # noqa: E402
from berkelium.agent.model_policies import ConstructorPolicy, GreedyRetrieval  # noqa: E402
from berkelium.agent.model_tasks import tasks  # noqa: E402


def main():
    t0 = time.time()
    T = tasks(4, seed=0)
    rep, lines = {"tasks": len(T), "truth": dict(Counter(t.truth_status for t in T)),
                  "splits": dict(Counter(t.split for t in T))}, []
    for pol_cls in (ConstructorPolicy, GreedyRetrieval):
        by = defaultdict(Counter)
        ret = defaultdict(float)
        for t in T:
            ep = run_model_episode(t, pol_cls())
            by[(t.split, "all")][ep["outcome"]] += 1
            by[(t.domain, t.truth_status)][ep["outcome"]] += 1
            ret[t.split] += ep["return"]
            lines.append(json.dumps(ep, sort_keys=True, default=str))
        rep[pol_cls.name] = {"by_split": {k[0]: dict(v) for k, v in by.items() if k[1] == "all"},
                             "by_domain_status": {f"{k[0]}|{k[1]}": dict(v) for k, v in sorted(by.items()) if k[1] != "all"},
                             "mean_return": {s: ret[s] / rep["splits"][s] for s in rep["splits"]}}
    Path("docs/experiments/trajectories_e5.jsonl").write_text("\n".join(lines) + "\n")
    rep["seconds"] = round(time.time() - t0, 1)
    Path("docs/experiments/model_e5.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps({k: rep[k] for k in ("tasks", "truth", "splits", "seconds")}))
    for p in ("constructor", "greedy_retrieval"):
        print(p, rep[p]["by_split"], {k: round(v, 3) for k, v in rep[p]["mean_return"].items()})
    print(json.dumps(rep["constructor"]["by_domain_status"]))
    print(json.dumps(rep["greedy_retrieval"]["by_domain_status"]))


if __name__ == "__main__":
    main()
