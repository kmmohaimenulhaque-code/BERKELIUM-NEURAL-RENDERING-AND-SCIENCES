"""Build SFT + preference data from VERIFIED ClaimEnv trajectories (family-held-out).

  python scripts/agent_build_training.py --out out/agent

SFT   : every step of grounded, positive-return CheapestSufficient episodes on the TRAIN family (nu 0.2/0.4),
        deduplicated on (prompt, action). The prompt is byte-identical to inference (agent.messages).
PAIRS : at the SAME observation, chosen = grounded action, rejected = the Overconfident policy's ungrounded
        declare (TRL conversational DPO format). Chosen/rejected are labelled by the environment, not a model.
HELD OUT: every task with nu not in {0.2, 0.4} never appears in either file.
"""
import argparse
import hashlib
import json
from pathlib import Path

from berkelium.agent import messages

SRC = Path("docs/experiments/trajectories_e4.jsonl")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/agent")
    a = ap.parse_args(argv)
    eps = [json.loads(x) for x in SRC.read_text().splitlines() if x.strip()]
    eps = [e for e in eps if e["memory"] and e["family"] == "train"]
    act = lambda a_: json.dumps(a_, sort_keys=True)  # noqa: E731
    sft, seen = [], set()
    for e in eps:
        if e["policy"] != "cheapest_sufficient" or e["return"] <= 0:
            continue
        for i, st in enumerate(e["trajectory"]):
            m = messages(st["obs"])
            key = (m[1]["content"], act(st["action"]))
            if key in seen:
                continue
            seen.add(key)
            sft.append({"id": f"{e['task']}#{i}", "family": e["family"], "verified": True,
                        "provenance": {"env": e["reward_version"], "policy": e["policy"], "outcome": e["outcome"]},
                        "messages": m + [{"role": "assistant", "content": act(st["action"])}]})
    by_task = {}
    for e in eps:
        by_task.setdefault(e["task"], {})[e["policy"]] = e
    pairs = []
    for t, d in sorted(by_task.items()):
        g, o = d.get("cheapest_sufficient"), d.get("overconfident")
        if not g or not o or o["outcome"] != "ungrounded":
            continue
        for i, (sg, so) in enumerate(zip(g["trajectory"], o["trajectory"], strict=False)):
            if sg["obs"] == so["obs"] and sg["action"] != so["action"]:
                m = messages(sg["obs"])
                pairs.append({"id": f"{t}#{i}", "family": g["family"], "prompt": m,
                              "chosen": [{"role": "assistant", "content": act(sg["action"])}],
                              "rejected": [{"role": "assistant", "content": act(so["action"])}]})
                break
    h = lambda r: int(hashlib.sha256(r["id"].split("#")[0].encode()).hexdigest(), 16) % 10  # noqa: E731
    val = [r for r in sft if h(r) == 0]
    train = [r for r in sft if h(r) != 0]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.chat.jsonl", train), ("val.chat.jsonl", val), ("pairs.jsonl", pairs)):
        (out / name).write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
    man = {"source": str(SRC), "source_sha256": sha(SRC), "families": sorted({r["family"] for r in sft}),
           "sft_train": len(train), "sft_val": len(val), "pairs": len(pairs), "heldout_family": "heldout (every nu not in {0.2, 0.4})",
           "files": {n: sha(out / n) for n in ("train.chat.jsonl", "val.chat.jsonl", "pairs.jsonl")}}
    (out / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
