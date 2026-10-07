"""Evaluate an LLM (via the model gateway) as a ClaimEnv policy on VERIFIED tasks (ADR-022).

  export BERKELIUM_MODEL_URL=http://127.0.0.1:8000/v1 BERKELIUM_MODEL=Qwen/Qwen3-32B
  python scripts/agent_llm_eval.py --split heldout --label base --out out/agent/base.json

Splits are by design family: train = nu 0.2/0.4; heldout = every other Poisson ratio (nu=0.3 and 10 fresh
points) — never used for training or discovery.
Aborts (exit 3, no report) if the endpoint is lost — a dead endpoint must never be scored as a 0 % model.
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
from agent_experiment import tasks  # noqa: E402

from berkelium.agent import CheapestSufficient, GatewayPolicy, Overconfident, run_episode  # noqa: E402
from berkelium.ai.gateway import GatewayError, provider_from_env  # noqa: E402


def select(split):
    ts = tasks(True)
    return [t for t in ts if t.family == split] if split != "all" else ts


def summarize(eps):
    n = len(eps)
    out = Counter(e["outcome"] for e in eps)
    verdict_ok = 0
    for e in eps:
        last = e["trajectory"][-1]["action"]
        if last.get("type") == "declare" and last.get("verdict") == e["truth"]:
            verdict_ok += 1
    return {"tasks": n, "mean_return": sum(e["return"] for e in eps) / n, "mean_cost": sum(e["cost"] for e in eps) / n,
            "outcomes": dict(out), "answer_only_accuracy": verdict_ok / n,
            "grounded_correct_rate": out["correct"] / n,
            "ungrounded_rate": out["ungrounded"] / n,
            "invalid_action_rate": sum(1 for e in eps for s in e["trajectory"] if s["action"].get("type") not in
                                       ("evaluate", "declare")) / max(1, sum(len(e["trajectory"]) for e in eps))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["heldout", "train", "all"], default="heldout")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--wait", type=float, default=0.0)
    a = ap.parse_args(argv)
    prov = provider_from_env()
    try:
        prov.check(wait_s=a.wait)
    except GatewayError as e:
        print(f"model endpoint not ready: {e}", file=sys.stderr)
        return 3
    T = select(a.split)[: a.limit or None]
    pol, eps, bad = GatewayPolicy(prov), [], 0
    for t in T:
        try:
            eps.append(run_episode(t, pol))
            bad = 0
        except GatewayError as e:
            bad += 1
            if bad >= 3:
                print(f"evaluation aborted, endpoint lost: {e}", file=sys.stderr)
                return 3
    rep = {"label": a.label, "model": prov.model, "split": a.split, "reward_version": eps[0]["reward_version"],
           "llm": summarize(eps),
           "reference_cheapest_sufficient": summarize([run_episode(t, CheapestSufficient()) for t in T]),
           "reference_overconfident": summarize([run_episode(t, Overconfident()) for t in T])}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=2) + "\n")
    Path(a.out).with_suffix(".trajectories.jsonl").write_text(
        "\n".join(json.dumps(e, sort_keys=True, default=str) for e in eps) + "\n")
    print(json.dumps({k: rep[k] for k in ("label", "model", "split", "llm")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
