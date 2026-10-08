"""Build AGENT-V2 training data from VERIFIED ModelEnv trajectories (+ optionally merge the AGENT-V1 ClaimEnv data).

  python scripts/model_build_training.py --out out/agent_v2 [--merge out/agent]

SFT   : every step of CORRECT+GROUNDED constructor episodes on TRAIN domains (structures, heat, fluids),
        deduplicated on (prompt, action); prompt = model_messages(obs), byte-identical to inference.
PAIRS : first step where greedy retrieval diverges from the constructor at the SAME observation, for tasks where
        greedy ended wrong/ungrounded: chosen = constructor action, rejected = greedy action (labelled by the env).
SPLIT : val = every 6th sampled INSTANCE of each scenario (whole episodes, never individual steps); HELD-OUT domains
        (vessel, gas_dynamics, rocket) are excluded and the build ASSERTS it.
Lineage: manifest records source file hashes, seeds, policy versions, counts, and the merged V1 manifest.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
from berkelium.agent.model_env import REWARD_VERSION, run_model_episode  # noqa: E402
from berkelium.agent.model_policies import ConstructorPolicy, GreedyRetrieval, model_messages  # noqa: E402
from berkelium.agent.model_tasks import TRAIN_DOMAINS, tasks  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/agent_v2")
    ap.add_argument("--seed", type=int, default=1)          # seed 0 is the E5 benchmark; training uses others
    ap.add_argument("--n-per", type=int, default=12)
    ap.add_argument("--merge", default=None, help="AGENT-V1 ClaimEnv data dir to merge (its manifest is recorded)")
    ap.add_argument("--curriculum", default=None, help="failure_analysis.py spec: oversample failing families")
    a = ap.parse_args(argv)
    T = [t for t in tasks(a.n_per, seed=a.seed) if t.domain in TRAIN_DOMAINS]
    curriculum = json.loads(Path(a.curriculum).read_text()) if a.curriculum else None
    if curriculum:   # extra fresh instances (new seeds) of the families the evaluated model fails on
        w = curriculum["oversample"]
        for k in range(1, max(w.values(), default=1)):
            fam_, inst_ = lambda t: t.id.split("#")[0], lambda t: int(t.id.split("#")[1])  # noqa: E731
            T += [t.__class__(**{**t.__dict__, "id": f"{fam_(t)}#{1000 * k + inst_(t)}"})
                  for t in tasks(a.n_per, seed=a.seed + 1000 * k)
                  if t.domain in TRAIN_DOMAINS and w.get(fam_(t), 1) > k]
    act = lambda x: json.dumps(x, sort_keys=True)  # noqa: E731
    sft, seen, pairs = [], set(), []
    fam = lambda tid: tid.split("#")[0]  # noqa: E731
    for t in T:
        g = run_model_episode(t, ConstructorPolicy())
        if g["outcome"] != "correct":
            raise SystemExit(f"teacher failed on {t.id}: {g['outcome']} (refusing to build data)")
        for i, st in enumerate(g["trajectory"]):
            m = model_messages(st["obs"])
            key = (m[1]["content"], act(st["action"]))
            if key not in seen:
                seen.add(key)
                sft.append({"id": f"{t.id}#{i}", "family": fam(t.id), "domain": t.domain, "verified": True,
                            "provenance": {"env": REWARD_VERSION, "policy": "constructor", "seed": a.seed},
                            "messages": m + [{"role": "assistant", "content": act(st["action"])}]})
        b = run_model_episode(t, GreedyRetrieval())
        if b["outcome"] != "correct":
            for sg, sb in zip(g["trajectory"], b["trajectory"], strict=False):
                if sg["obs"] == sb["obs"] and sg["action"] != sb["action"]:
                    pairs.append({"id": t.id, "family": fam(t.id), "domain": t.domain, "prompt": model_messages(sg["obs"]),
                                  "chosen": [{"role": "assistant", "content": act(sg["action"])}],
                                  "rejected": [{"role": "assistant", "content": act(sb["action"])}],
                                  "rejected_outcome": b["outcome"]})
                    break
    assert all(r["domain"] in TRAIN_DOMAINS for r in sft + pairs), "held-out domain leaked into training data"
    inst = lambda r: int(r["id"].split("#")[1])  # noqa: E731   (task id = family#instance#step)
    val = [r for r in sft if inst(r) % 6 == 0]
    train = [r for r in sft if inst(r) % 6 != 0]
    ep = lambda r: "#".join(r["id"].split("#")[:2])  # noqa: E731
    assert not ({ep(r) for r in val} & {ep(r) for r in train}), "an episode is split across train/val"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    merged = None
    n_model = (len(train), len(val), len(pairs))
    if a.merge:
        mdir = Path(a.merge)
        merged = json.loads((mdir / "manifest.json").read_text())
        for name in ("train.chat.jsonl", "val.chat.jsonl", "pairs.jsonl"):
            extra = [json.loads(x) for x in (mdir / name).read_text().splitlines() if x.strip()]
            {"train.chat.jsonl": train, "val.chat.jsonl": val, "pairs.jsonl": pairs}[name].extend(extra)
    for name, rows in (("train.chat.jsonl", train), ("val.chat.jsonl", val), ("pairs.jsonl", pairs)):
        (out / name).write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
    man = {"dataset": "AGENT-V2" if a.merge else "MODELENV", "env": REWARD_VERSION, "seed": a.seed, "n_per": a.n_per,
           "train_domains": sorted(TRAIN_DOMAINS), "heldout_domains": "vessel, gas_dynamics, rocket (never included)",
           "sft_train": len(train), "sft_val": len(val), "pairs": len(pairs),
           "modelenv_rows": {"sft_train": n_model[0], "sft_val": n_model[1], "pairs": n_model[2]},
           "split_policy": "modelenv: instance % 6 == 0 -> val (episode-level); claimenv: as in merged manifest",
           "merged_from": merged, "curriculum": curriculum and {"source": a.curriculum,
                                                                  "oversample": curriculum["oversample"]},
           "files": {n: sha(out / n) for n in ("train.chat.jsonl", "val.chat.jsonl", "pairs.jsonl")}}
    (out / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    print(json.dumps({k: v for k, v in man.items() if k not in ("files", "merged_from")}))


if __name__ == "__main__":
    main()
