"""Evaluate an LLM (via the model gateway) as a ModelEnv model-construction policy (ADR-023).

  export BERKELIUM_MODEL_URL=http://127.0.0.1:8000/v1 BERKELIUM_MODEL=agent_v2
  python scripts/model_llm_eval.py --split heldout --label agent_v2 --out out/eval/model_agent_v2.json

Tasks: the E5 benchmark (seed 0) — NEVER used for training (training data uses seed >= 1). heldout = vessel,
gas dynamics, rocket (domains absent from all training data); train = structures, heat, fluids (seen domains,
unseen instances). Aborts with exit 3 (no report) if the endpoint is lost.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.getcwd())
from berkelium.agent.model_env import run_model_episode  # noqa: E402
from berkelium.agent.model_policies import ConstructorPolicy, GreedyRetrieval, LLMModelPolicy  # noqa: E402
from berkelium.agent.model_tasks import tasks  # noqa: E402
from berkelium.ai.gateway import GatewayError, provider_from_env  # noqa: E402

ACTIONS = ("search", "add", "remove", "solve", "declare")


def summarize(eps):
    n = len(eps)
    out = Counter(e["outcome"] for e in eps)
    by = defaultdict(Counter)
    ans = 0
    for e in eps:
        by[f"{e['domain']}|{e['truth']}"][e["outcome"]] += 1
        last = e["trajectory"][-1]["action"]
        ans += last.get("type") == "declare" and last.get("status") == e["truth"]
    steps = sum(len(e["trajectory"]) for e in eps)
    return {"tasks": n, "mean_return": sum(e["return"] for e in eps) / n, "outcomes": dict(out),
            "grounded_correct_rate": out["correct"] / n, "wrong_rate": out["wrong"] / n,
            "ungrounded_rate": out["ungrounded"] / n, "status_only_accuracy": ans / n,
            "mean_steps": steps / n,
            "invalid_action_rate": sum(e2["action"].get("type") not in ACTIONS for e in eps for e2 in e["trajectory"])
            / max(1, steps), "by_domain_status": {k: dict(v) for k, v in sorted(by.items())}}


def main(argv=None, provider=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["heldout", "train", "all"], default="heldout")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--wait", type=float, default=0.0)
    a = ap.parse_args(argv)
    prov = provider or provider_from_env()
    try:
        prov.check(wait_s=a.wait)
    except GatewayError as e:
        print(f"model endpoint not ready: {e}", file=sys.stderr)
        return 3
    T = [t for t in tasks(4, seed=0) if a.split == "all" or t.split == a.split][: a.limit or None]
    eps, bad = [], 0
    for t in T:
        try:
            eps.append(run_model_episode(t, LLMModelPolicy(prov)))
            bad = 0
        except GatewayError as e:
            bad += 1
            if bad >= 3:
                print(f"evaluation aborted, endpoint lost: {e}", file=sys.stderr)
                return 3
    rep = {"label": a.label, "model": prov.model, "split": a.split, "benchmark": "modelenv E5 seed 0",
           "reward_version": eps[0]["reward_version"], "llm": summarize(eps),
           "reference_constructor": summarize([run_model_episode(t, ConstructorPolicy()) for t in T]),
           "reference_greedy": summarize([run_model_episode(t, GreedyRetrieval()) for t in T])}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=2) + "\n")
    Path(a.out).with_suffix(".trajectories.jsonl").write_text(
        "\n".join(json.dumps(e, sort_keys=True, default=str) for e in eps) + "\n")
    print(json.dumps({k: rep[k] for k in ("label", "model", "split")} | {"llm": {k: v for k, v in rep["llm"].items()
                     if k != "by_domain_status"}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
