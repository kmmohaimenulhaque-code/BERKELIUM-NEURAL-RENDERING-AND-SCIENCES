"""Phase 14 — failure analysis -> targeted curriculum (ADR-024).

Reads ModelEnv trajectories (from model_llm_eval.py or model_experiment.py), assigns every NON-correct episode a
mechanistic failure mode by replaying what the agent saw and did, clusters them, and writes a curriculum spec that
model_build_training.py consumes (--curriculum): which scenario families to oversample and by how much.

Failure modes (first matching rule, deterministic):
  invalid_action        emitted something that is not a legal action
  context_violation     declared while its model used a fragment whose context does not hold
  ignored_validity      declared 'determined' while its last solve reported validity failures
  ignored_closure       declared 'determined' while its last solve reported closure (definition) violations
  declared_without_solve declared with no solve after its last model change
  wrong_status          a different status than truth (e.g. determined vs ambiguous)
  wrong_value           right status, wrong number
  budget_exhausted      never declared
  other
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

LEGAL = {"search", "add", "remove", "solve", "declare"}


def mode(ep) -> str:
    tr = ep["trajectory"]
    if any(s["action"].get("type") not in LEGAL for s in tr):
        return "invalid_action"
    if ep["outcome"] == "budget_exhausted":
        return "budget_exhausted"
    last_obs, act = tr[-1]["obs"], tr[-1]["action"]
    ls = last_obs.get("last_solve") or {}
    changed_after_solve = False
    for s in tr:
        t = s["action"].get("type")
        if t in ("add", "remove"):
            changed_after_solve = True
        if t == "solve":
            changed_after_solve = False
    if ls.get("context_violations"):
        return "context_violation"
    if act.get("status") == "determined" and ls.get("validity_failures"):
        return "ignored_validity"
    if act.get("status") == "determined" and ls.get("closure_violations"):
        return "ignored_closure"
    if not ls or changed_after_solve:
        return "declared_without_solve"
    if act.get("status") != ep["truth"]:
        return "wrong_status"
    if ep["outcome"] in ("wrong", "ungrounded"):
        return "wrong_value"
    return "other"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("trajectories")
    ap.add_argument("--policy", default=None)
    ap.add_argument("--out", default="out/curriculum.json")
    ap.add_argument("--max-weight", type=int, default=4)
    a = ap.parse_args(argv)
    eps = [json.loads(x) for x in Path(a.trajectories).read_text().splitlines() if x.strip()]
    eps = [e for e in eps if a.policy is None or e["policy"] == a.policy]
    fails = [e for e in eps if e["outcome"] != "correct"]
    clusters = defaultdict(list)
    for e in fails:
        clusters[mode(e)].append(e["task"].split("#")[0])
    fam_fail = Counter(e["task"].split("#")[0] for e in fails)
    fam_all = Counter(e["task"].split("#")[0] for e in eps)
    weights = {f: min(a.max_weight, 1 + round((a.max_weight - 1) * fam_fail[f] / fam_all[f])) for f in sorted(fam_all)}
    spec = {"source": a.trajectories, "policy": a.policy, "episodes": len(eps), "failures": len(fails),
            "clusters": {k: {"count": len(v), "families": dict(Counter(v))} for k, v in sorted(clusters.items())},
            "family_failure_rate": {f: fam_fail[f] / fam_all[f] for f in sorted(fam_all)},
            "oversample": weights,
            "note": "weights apply to TRAIN-domain families only; held-out domains stay excluded from training"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(spec, indent=2) + "\n")
    print(json.dumps({k: spec[k] for k in ("episodes", "failures", "clusters", "oversample")}, indent=1))


if __name__ == "__main__":
    main()
