"""ADR-025: batched lockstep runner == per-episode runner; resume; fail-fast waiting; multi-adapter serving."""
import json
import os
import subprocess
import sys
import time

import pytest

from berkelium.agent import CheapestSufficient, run_episode
from berkelium.agent.batch_runner import parse_action, run_batched, thread_generator
from berkelium.agent.model_env import run_model_episode
from berkelium.agent.model_policies import ConstructorPolicy
from berkelium.agent.model_tasks import tasks as model_tasks
from berkelium.ai.gateway import GatewayError, OpenAICompatibleProvider
from berkelium.ai.waiting import make_waiter, pid_alive

sys.path.insert(0, "scripts")


def claim_tasks(n):
    from agent_llm_eval import select
    return select("heldout")[:n]


def replay_claim(batch):
    return [json.dumps(CheapestSufficient().act(json.loads(m[1]["content"]))) for m in batch]


class ReplayModel:
    """Teacher behind a text interface: one ConstructorPolicy per problem (keyed by the problem statement)."""

    def __init__(self):
        self.pol = {}

    def __call__(self, batch):
        out = []
        for m in batch:
            obs = json.loads(m[1]["content"])
            k = json.dumps(obs["problem"], sort_keys=True)
            out.append(json.dumps(self.pol.setdefault(k, ConstructorPolicy()).act(obs)))
        return out


def _strip(ep):
    return {k: v for k, v in ep.items() if k != "policy"}


def test_batched_claim_episodes_equal_sequential_ones():
    ts = claim_tasks(12)
    got = {e["task"]: e for e in run_batched("claim", ts, replay_claim, "x", batch_size=5)}
    for t in ts:
        assert _strip(got[t.id]) == _strip(run_episode(t, CheapestSufficient()))


def test_batched_model_episodes_equal_sequential_ones():
    ts = [t for t in model_tasks(1, seed=0)][:10]
    got = {e["task"]: e for e in run_batched("model", ts, ReplayModel(), "x", batch_size=4)}
    for t in ts:
        assert _strip(got[t.id]) == _strip(run_model_episode(t, ConstructorPolicy()))


def test_garbage_output_is_an_explicit_invalid_action_not_a_crash():
    assert parse_action("I think the beam is fine")["type"] == "invalid"
    assert parse_action('```json\n{"type": "solve"}\n```') == {"type": "solve"}
    eps = run_batched("claim", claim_tasks(2), lambda b: ["no json"] * len(b), "x", batch_size=2)
    assert all(e["outcome"] == "budget_exhausted" and e["return"] == -1.0 for e in eps)


def test_eval_resumes_after_interrupt_without_duplicates(tmp_path):
    import eval_generations as eg
    ts = claim_tasks(6)
    calls = {"n": 0}

    traj = tmp_path / "claim__g.trajectories.jsonl"

    class Backend:
        def generator(self, gen):
            def g(batch):
                calls["n"] += 1
                if calls["n"] > 0 and traj.exists() and traj.read_text().strip():
                    raise KeyboardInterrupt          # Ctrl+C after the first episodes reached disk
                return replay_claim(batch)
            return g
    with pytest.raises(KeyboardInterrupt):
        eg.run_env_suite(Backend(), "g", "claim", ts, tmp_path, batch=2)
    part = (tmp_path / "claim__g.trajectories.jsonl").read_text().splitlines()
    assert 0 < len(part) < 6
    calls["n"] = -100
    rep = eg.run_env_suite(Backend(), "g", "claim", ts, tmp_path, batch=2)
    lines = (tmp_path / "claim__g.trajectories.jsonl").read_text().splitlines()
    assert len(lines) == 6 and len({json.loads(x)["task"] for x in lines}) == 6
    assert rep["llm"]["tasks"] == 6 and rep["llm"]["wrong_rate"] == 0.0


def test_waiter_fails_fast_when_server_process_died(tmp_path):
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    pidfile, log = tmp_path / "vllm.pid", tmp_path / "vllm.log"
    pidfile.write_text(str(p.pid))
    log.write_text("INFO loading\nValueError: LoRA rank 32 is greater than max_lora_rank 16\n")
    assert pid_alive(pidfile) is False
    prov = OpenAICompatibleProvider("http://127.0.0.1:1/v1", "m")
    t = time.monotonic()
    with pytest.raises(GatewayError, match="max_lora_rank"):
        prov.check(wait_s=600, interval_s=0.01, on_wait=make_waiter(prov.base_url, pidfile, log))
    assert time.monotonic() - t < 5                     # not 10 minutes


def test_waiter_reports_progress_while_waiting(tmp_path, capsys):
    prov = OpenAICompatibleProvider("http://127.0.0.1:1/v1", "m")
    with pytest.raises(GatewayError):
        prov.check(wait_s=0.3, interval_s=0.05, on_wait=make_waiter(prov.base_url, every=0.0, out=sys.stdout))
    assert "waiting for http://127.0.0.1:1/v1" in capsys.readouterr().out


def test_thread_generator_preserves_order():
    gen = thread_generator(lambda m: m[0]["content"], concurrency=4)
    msgs = [[{"role": "user", "content": str(i)}] for i in range(20)]
    assert gen(msgs) == [str(i) for i in range(20)]


@pytest.mark.skipif(not os.path.exists("/bin/bash"), reason="bash required")
def test_serve_script_serves_all_adapters_with_computed_rank(tmp_path):
    for name, r in (("a16", 16), ("a32", 32)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "adapter_config.json").write_text(json.dumps({"r": r}))
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "vllm").write_text('#!/bin/sh\necho "$@"\n')
    (fake / "vllm").chmod(0o755)
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}", "VLLM_MODE": "path",
           "VLLM_PIDFILE": str(tmp_path / "pid"), "PYTHON": sys.executable}
    out = subprocess.run(["bash", "scripts/serve_vllm.sh", f"m1={tmp_path / 'a16'}", f"v2={tmp_path / 'a32'}"],
                         capture_output=True, text=True, env=env, check=True).stdout
    assert "--max-lora-rank 32" in out and "--max-loras 2" in out
    assert f"m1={tmp_path / 'a16'}" in out and f"v2={tmp_path / 'a32'}" in out
    assert (tmp_path / "pid").read_text().strip().isdigit()
    bad = subprocess.run(["bash", "scripts/serve_vllm.sh", f"x={tmp_path / 'missing'}"], capture_output=True,
                         text=True, env=env)
    assert bad.returncode == 2 and "no adapter_config.json" in bad.stderr


@pytest.mark.skipif(not os.environ.get("BERKELIUM_TEST_HF_MODEL"), reason="opt-in: set BERKELIUM_TEST_HF_MODEL")
def test_local_provider_switches_adapters(tmp_path):
    """Opt-in (downloads a model): base vs a random LoRA must differ; switching back restores base output."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    from berkelium.ai.local_provider import LocalHFProvider
    mid = os.environ["BERKELIUM_TEST_HF_MODEL"]
    torch.manual_seed(0)
    m = get_peft_model(AutoModelForCausalLM.from_pretrained(mid), LoraConfig(r=4, init_lora_weights=False,
                                                                             target_modules=["q_proj", "v_proj"]))
    m.save_pretrained(tmp_path / "lora")
    p = LocalHFProvider(mid, {"rand": str(tmp_path / "lora")}, max_new_tokens=8)
    msgs = [[{"role": "user", "content": "Say hi"}], [{"role": "user", "content": "Reply with the JSON {\"a\": 1}"}]]
    base = p.generate_many(msgs)
    p.use("rand")
    tuned = p.generate_many(msgs)
    p.use("base")
    assert p.generate_many(msgs) == base and tuned != base and len(base) == 2
