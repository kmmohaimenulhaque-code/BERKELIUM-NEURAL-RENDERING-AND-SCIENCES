"""Evaluate EVERY model generation in ONE command, with progress, resume and clean shutdown (ADR-025).

  python scripts/eval_generations.py --limit 2          # smoke run (~minutes): proves the whole pipeline
  python scripts/eval_generations.py                    # full run -> out/eval/summary.md

Generations (default): base, m1, agent-v1, agent-v2, agent-v2-dpo. Adapters come from their Hugging Face repos
(root files only, cached under models/adapters/) unless overridden: --adapter agent-v2=training/runs/qwen3-32b-lora-agent-v2
Suites (default): claim (ClaimEnv held-out), model (ModelEnv held-out domains), model_seen (ModelEnv seen domains,
unseen instances). Optional: planning (M1's own task: NL -> DesignProposal, gear-only prompt as trained; slow).

Backends
  local (default) transformers + PEFT in this process: the base model is loaded ONCE and every adapter is
                  switched in place. No server, no port, no vLLM. Same stack the adapters were trained with.
  vllm            starts scripts/serve_vllm.sh with ALL adapters in one server, waits visibly (fails fast if the
                  server dies), evaluates, and always stops the server at the end — also on Ctrl+C.
  --endpoint URL  use an already running OpenAI-compatible server instead (adapters served under these names).

Every finished episode is appended to out/eval/<suite>__<gen>.trajectories.jsonl immediately. Ctrl+C loses
nothing: rerun the same command and it resumes. Decoding is greedy (temperature 0), thinking off, at most
--max-new-tokens (default 256) tokens per action; actions are ~30-token JSON objects.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))

HF_ADAPTERS = {"m1": "Mhaquehaque/berkelium-qwen3-32b-lora",
               "agent-v1": "Mhaquehaque/berkelium-qwen3-32b-agent-v1",
               "agent-v2": "Mhaquehaque/berkelium-qwen3-32b-agent-v2",
               "agent-v2-dpo": "Mhaquehaque/berkelium-qwen3-32b-agent-v2-dpo"}
ROOT_FILES = ["adapter_config.json", "adapter_model.safetensors", "tokenizer*", "chat_template.jinja",
              "run_manifest.json"]
SUITES = ("claim", "model", "model_seen", "planning")


def say(*a):
    print(*a, flush=True)


def fmt_s(s):
    if s is None:
        return "?"
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


# ------------------------------------------------------------------------------------------------ tasks
def suite_tasks(suite: str, limit: int):
    if suite == "claim":
        from agent_llm_eval import select
        ts = select("heldout")
    elif suite in ("model", "model_seen"):
        from berkelium.agent.model_tasks import tasks
        ts = [t for t in tasks(4, seed=0) if t.split == ("heldout" if suite == "model" else "train")]
    else:
        raise ValueError(suite)
    return ts[:limit] if limit else ts


def references(suite: str, ts):
    if suite == "claim":
        from berkelium.agent import CheapestSufficient, Overconfident, run_episode
        return {"cheapest_sufficient": summarize("claim", [run_episode(t, CheapestSufficient()) for t in ts]),
                "overconfident": summarize("claim", [run_episode(t, Overconfident()) for t in ts])}
    from berkelium.agent.model_env import run_model_episode
    from berkelium.agent.model_policies import ConstructorPolicy, GreedyRetrieval
    return {"constructor": summarize(suite, [run_model_episode(t, ConstructorPolicy()) for t in ts]),
            "greedy_retrieval": summarize(suite, [run_model_episode(t, GreedyRetrieval()) for t in ts])}


def summarize(suite, eps):
    """Suite summary + a uniform wrong_rate (ClaimEnv calls a confidently wrong verdict 'wrong_grounded')."""
    if not eps:
        return {"tasks": 0}
    if suite == "claim":
        from agent_llm_eval import summarize as s
        m = s(eps)
        m["wrong_rate"] = m["outcomes"].get("wrong_grounded", 0) / m["tasks"]
        return m
    from model_llm_eval import summarize as s
    return s(eps)


# ---------------------------------------------------------------------------------------------- adapters
def resolve_adapters(gens, overrides):
    from huggingface_hub import snapshot_download
    out, info = {}, {}
    for g in gens:
        if g == "base":
            continue
        src = overrides.get(g, HF_ADAPTERS.get(g))
        if src is None:
            raise SystemExit(f"no source for generation {g!r}; pass --adapter {g}=PATH_OR_REPO")
        if Path(src, "adapter_config.json").exists():
            out[g], info[g] = str(src), {"source": "local", "path": str(src)}
            continue
        say(f"fetching adapter {g} from {src} (root files only)...")
        path = snapshot_download(src, local_dir=f"models/adapters/{g}", allow_patterns=ROOT_FILES)
        sha = None
        try:
            from huggingface_hub import HfApi
            sha = HfApi().model_info(src).sha
        except Exception:  # noqa: BLE001 — offline is fine, the files are what we evaluate
            pass
        out[g], info[g] = path, {"source": "hf", "repo": src, "sha": sha, "path": path}
    for g, p in out.items():
        info[g]["lora_r"] = json.loads(Path(p, "adapter_config.json").read_text()).get("r")
    return out, info


# ----------------------------------------------------------------------------------------------- backends
class LocalBackend:
    def __init__(self, base, adapters, max_new, batch, min_free_gb):
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            if free / 2**30 < min_free_gb:
                raise SystemExit(f"GPU busy: only {free / 2**30:.0f} of {total / 2**30:.0f} GB free (need {min_free_gb}).\n"
                                 f"A vLLM server from an earlier run is probably still alive. Stop it:\n"
                                 f"  pkill -f 'vllm serve'; sleep 15")
        else:
            say("WARNING: no GPU visible to torch - this will be extremely slow")
        from berkelium.ai.local_provider import LocalHFProvider
        say(f"loading {base} + {len(adapters)} adapter(s) in-process (first time downloads the base model)...")
        t = time.time()
        self.p = LocalHFProvider(base, adapters, max_new_tokens=max_new)
        say(f"model ready in {fmt_s(time.time() - t)}")
        self.batch = batch

    def generator(self, gen):
        self.p.use("base" if gen == "base" else gen)
        return lambda msgs: self.p.generate_many(msgs)

    def provider(self, gen):
        self.p.use("base" if gen == "base" else gen)
        return self.p

    def close(self):
        pass


class VllmBackend:
    def __init__(self, base, adapters, max_new, batch, endpoint, wait_s, outdir):
        from berkelium.ai.gateway import DecodingConfig, OpenAICompatibleProvider
        from berkelium.ai.waiting import make_waiter
        self.base, self.batch, self.proc = base, batch, None
        self.decoding = DecodingConfig(temperature=0.0, top_p=1.0, top_k=0, max_tokens=max_new, seed=0)
        self.url = endpoint or "http://127.0.0.1:8000/v1"
        log, pidfile = outdir / "logs" / "vllm.log", outdir / "logs" / "vllm.pid"
        if not endpoint:
            log.parent.mkdir(parents=True, exist_ok=True)
            pidfile.unlink(missing_ok=True)
            cmd = ["bash", "scripts/serve_vllm.sh", *[f"{g}={p}" for g, p in adapters.items()]]
            say(f"starting vLLM: {' '.join(cmd)}\n  log: {log}")
            self.proc = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT,
                                         env={**os.environ, "VLLM_PIDFILE": str(pidfile), "MODEL": base},
                                         start_new_session=True)
        self.Prov = OpenAICompatibleProvider
        probe = OpenAICompatibleProvider(self.url, base, timeout_s=600)
        probe.check(wait_s=wait_s, interval_s=10,
                    on_wait=make_waiter(self.url, None if endpoint else pidfile, None if endpoint else log))
        say(f"server ready: {probe.served_models()}")

    def provider(self, gen):
        return self.Prov(self.url, self.base if gen == "base" else gen, timeout_s=600)

    def generator(self, gen):
        from berkelium.agent.batch_runner import thread_generator
        p = self.provider(gen)
        return thread_generator(lambda m: p.generate(m, None, self.decoding).text, self.batch)

    def close(self):
        if self.proc and self.proc.poll() is None:
            say("stopping vLLM server...")
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)


# ------------------------------------------------------------------------------------------------- suites
def run_env_suite(backend, gen, suite, ts, outdir, batch):
    from berkelium.agent.batch_runner import run_batched
    traj = outdir / f"{suite}__{gen}.trajectories.jsonl"
    have = {}
    if traj.exists():
        for line in traj.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                have[e["task"]] = e
    todo = [t for t in ts if t.id not in have]
    tag = f"[{gen} | {suite}]"
    if todo:
        say(f"{tag} {len(have)} done earlier, running {len(todo)} episode(s)...")
        last = [0.0]

        def on_ep(ep):
            with traj.open("a") as f:
                f.write(json.dumps(ep, sort_keys=True, default=str) + "\n")
            have[ep["task"]] = ep

        def on_prog(p):
            if time.monotonic() - last[0] >= 15 or p["done"] == p["total"]:
                last[0] = time.monotonic()
                say(f"{tag} {p['done']}/{p['total']} episodes | round {p['round']} | {p['calls']} calls | "
                    f"{p['calls_per_s']:.2f} calls/s | ETA {fmt_s(p['eta_s'])}")
        run_batched("claim" if suite == "claim" else "model", todo, backend.generator(gen), f"llm:{gen}", batch,
                    on_episode=on_ep, on_progress=on_prog)
    else:
        say(f"{tag} already complete ({len(have)} episodes)")
    eps = [have[t.id] for t in ts if t.id in have]
    rep = {"generation": gen, "suite": suite, "tasks": len(ts), "llm": summarize(suite, eps),
           "reward_version": eps[0]["reward_version"] if eps else None}
    (outdir / f"{suite}__{gen}.json").write_text(json.dumps(rep, indent=2) + "\n")
    return rep


def run_planning(backend, gen, outdir, limit):
    """M1's trained task with the prompt it was trained on (gear-only CEM catalogue). Sequential: slow."""
    from huggingface_hub import hf_hub_download

    from berkelium.ai.evaluate import evaluate
    from berkelium.ai.gateway import DecodingConfig
    from berkelium.ai.orchestrator import Orchestrator
    from berkelium.cem.library.gear.cem import SpurGearPairCEM
    from berkelium.cem.protocol import CEMRegistry
    from berkelium.datasets.factory import load_split
    reg = CEMRegistry()
    reg.register(SpurGearPairCEM())
    ex = []
    for f in ("test.jsonl", "heldout_family.jsonl"):
        ex += load_split(hf_hub_download("Mhaquehaque/berkelium-engineering-dataset", f, repo_type="dataset"))
    ex = [e for e in ex if e["task"] in ("plan", "plan_procedural")][: limit or None]
    prov = backend.provider(gen)
    orch = Orchestrator(prov, registry=reg, constrained=isinstance(backend, VllmBackend),
                        decoding=DecodingConfig(temperature=0.0, top_p=1.0, top_k=0, max_tokens=4096, seed=0))
    n, t0, design = [0], time.time(), orch.design

    def counted(intent, max_repairs=2):
        r = design(intent, max_repairs=max_repairs)
        n[0] += 1
        say(f"[{gen} | planning] {n[0]}/{len(ex)} designs | ETA {fmt_s((time.time() - t0) / n[0] * (len(ex) - n[0]))}")
        return r
    orch.design = counted
    rep = evaluate(ex, orch, out=outdir / f"planning__{gen}.json", label=gen)
    return {"generation": gen, "suite": "planning", "tasks": len(ex), "llm": rep["metrics"],
            "constrained_decoding": orch.schema is not None}


# ---------------------------------------------------------------------------------------------- summary
COLS = [("grounded_correct_rate", "correct"), ("wrong_rate", "wrong"), ("ungrounded_rate", "ungrnd"),
        ("invalid_action_rate", "invalid"), ("mean_return", "return")]


def write_summary(outdir, reports, refs):
    lines = ["# Berkelium generation evaluation", "",
             "correct = grounded & correct; wrong = confidently wrong; ungrnd = right answer without evidence; "
             "invalid = unparsable actions. Higher correct, lower wrong is better.", ""]
    for suite in [s for s in SUITES if any(r["suite"] == s for r in reports)]:
        lines += [f"## {suite}", ""]
        if suite == "planning":
            lines += ["| gen | n | parse | schema | pass_final | req_met |", "|---|---|---|---|---|---|"]
            for r in [r for r in reports if r["suite"] == suite]:
                m = r["llm"]
                lines.append(f"| {r['generation']} | {m['n']} | {m['parse_rate']:.2f} | {m['schema_valid_rate']:.2f} | "
                             f"{m['validation_pass_final']:.2f} | {m['requirement_satisfaction']:.2f} |")
        else:
            lines += ["| gen | n | " + " | ".join(c for _, c in COLS) + " |", "|---" * (len(COLS) + 2) + "|"]
            rows = [(r["generation"], r["llm"]) for r in reports if r["suite"] == suite]
            rows += [(f"*{k}*", v) for k, v in refs.get(suite, {}).items()]
            for g, m in rows:
                lines.append(f"| {g} | {m['tasks']} | " + " | ".join(f"{m.get(k, 0):.2f}" for k, _ in COLS) + " |")
        lines.append("")
    (outdir / "summary.md").write_text("\n".join(lines) + "\n")
    (outdir / "summary.json").write_text(json.dumps({"reports": reports, "references": refs}, indent=2) + "\n")
    say("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["local", "vllm"], default="local")
    ap.add_argument("--endpoint", default=None, help="existing OpenAI-compatible server (implies --backend vllm)")
    ap.add_argument("--generations", default="base,m1,agent-v1,agent-v2,agent-v2-dpo")
    ap.add_argument("--suites", default="claim,model,model_seen")
    ap.add_argument("--adapter", action="append", default=[], help="NAME=LOCAL_DIR_OR_HF_REPO override")
    ap.add_argument("--base", default="Qwen/Qwen3-32B")
    ap.add_argument("--limit", type=int, default=0, help="first N tasks per suite (smoke run -> out/eval_smoke)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=256)   # actions are ~30-token JSON objects
    ap.add_argument("--min-free-gb", type=float, default=80.0)
    ap.add_argument("--server-wait", type=float, default=3600.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    gens = [g.strip() for g in a.generations.split(",") if g.strip()]
    suites = [s.strip() for s in a.suites.split(",") if s.strip()]
    bad = [s for s in suites if s not in SUITES]
    if bad:
        raise SystemExit(f"unknown suite(s) {bad}; choose from {SUITES}")
    outdir = Path(a.out or ("out/eval_smoke" if a.limit else "out/eval"))
    outdir.mkdir(parents=True, exist_ok=True)
    overrides = dict(x.split("=", 1) for x in a.adapter)
    adapters, ainfo = resolve_adapters(gens, overrides)
    tasks = {s: suite_tasks(s, a.limit) for s in suites if s != "planning"}
    say("tasks per suite: " + ", ".join(f"{s}={len(v)}" for s, v in tasks.items()))
    refs = {s: references(s, ts) for s, ts in tasks.items()}
    env = subprocess.run([sys.executable, "scripts/env_manifest.py"], capture_output=True, text=True).stdout
    (outdir / "eval_manifest.json").write_text(json.dumps({
        "generations": gens, "suites": suites, "adapters": ainfo, "base": a.base, "limit": a.limit,
        "backend": "endpoint" if a.endpoint else a.backend, "batch": a.batch,
        "decoding": {"greedy": True, "enable_thinking": False, "max_new_tokens": a.max_new_tokens},
        "task_ids": {s: [t.id for t in ts] for s, ts in tasks.items()},
        "benchmark_note": "ModelEnv seed 0 is never used for training (training seeds >= 1); ClaimEnv held-out = nu not in {0.2, 0.4}",
        "env": json.loads(env) if env.strip() else None}, indent=2) + "\n")
    backend, reports, t0 = None, [], time.time()
    try:
        if a.endpoint or a.backend == "vllm":
            backend = VllmBackend(a.base, adapters, a.max_new_tokens, a.batch, a.endpoint, a.server_wait, outdir)
        else:
            backend = LocalBackend(a.base, adapters, a.max_new_tokens, a.batch, a.min_free_gb)
        for gen in gens:
            for s in suites:
                if s == "planning":
                    if gen in ("base", "m1"):
                        reports.append(run_planning(backend, gen, outdir, a.limit))
                    continue
                reports.append(run_env_suite(backend, gen, s, tasks[s], outdir, a.batch))
        write_summary(outdir, reports, refs)
        say(f"\nDONE in {fmt_s(time.time() - t0)}. Commit: {outdir}/*.json {outdir}/summary.md {outdir}/eval_manifest.json")
        return 0
    except KeyboardInterrupt:
        say(f"\nStopped. Finished episodes are saved in {outdir}/. Run the SAME command again to resume.")
        return 130
    finally:
        if backend:
            backend.close()


if __name__ == "__main__":
    sys.exit(main())
