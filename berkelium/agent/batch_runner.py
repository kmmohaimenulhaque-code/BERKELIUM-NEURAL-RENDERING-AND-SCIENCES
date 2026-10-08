"""Lockstep batched episode runner for ClaimEnv / ModelEnv (ADR-025).

Why: the per-episode runners call the model once per step, sequentially, silently. On a 32B model that is hours
of apparent "hang". Here every live episode advances one step per round; the round's prompts are generated as
one batch (local transformers) or concurrently (vLLM batches them server-side). Every finished episode is handed
to ``on_episode`` immediately (the caller appends it to disk -> Ctrl+C loses nothing, reruns resume), and
``on_progress`` reports done/total, calls/s and an ETA.

Episode records are field-for-field the ones produced by ``env.run_episode`` / ``model_env.run_model_episode``,
so the existing summaries (scripts/agent_llm_eval.py, scripts/model_llm_eval.py) and failure analysis apply.
Batched greedy decoding is not guaranteed bit-identical across batch compositions (padding numerics); the batch
size is recorded in the eval manifest.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..ai.orchestrator import extract_json
from .env import REWARD_VERSION as CLAIM_RV
from .env import ClaimEnv
from .model_env import REWARD_VERSION as MODEL_RV
from .model_env import ModelEnv
from .model_policies import model_messages
from .policies import messages as claim_messages

Messages = list[dict]
GenerateMany = Callable[[list[Messages]], list[str]]


def parse_action(text: str) -> dict:
    """Model text -> action dict; anything unparsable is an explicit invalid action (kept for analysis)."""
    try:
        a = extract_json(text)
    except (ValueError, TypeError):
        return {"type": "invalid", "raw": (text or "")[:300]}
    return a if isinstance(a, dict) else {"type": "invalid", "raw": (text or "")[:300]}


@dataclass
class _Live:
    task: object
    env: object
    obs: dict
    traj: list = field(default_factory=list)
    total: float = 0.0
    last_info: dict = field(default_factory=dict)


def _claim_episode(lv: _Live, policy: str) -> dict:
    t = lv.task
    out = lv.traj[-1]["info"].get("outcome", "budget_exhausted") if lv.traj else "budget_exhausted"
    return {"task": t.id, "family": t.family, "policy": policy, "truth": t.truth, "return": lv.total,
            "outcome": out, "cost": lv.env.spent, "steps": len(lv.traj), "reward_version": CLAIM_RV,
            "trajectory": json.loads(json.dumps(lv.traj, default=str))}


def _model_episode(lv: _Live, policy: str) -> dict:
    t = lv.task
    return {"task": t.id, "domain": t.domain, "split": t.split, "policy": policy, "truth": t.truth_status,
            "return": lv.total, "outcome": lv.last_info.get("outcome", "budget_exhausted"), "steps": len(lv.traj),
            "reward_version": MODEL_RV, "trajectory": json.loads(json.dumps(lv.traj, default=str))}


SUITES = {
    "claim": {"env": ClaimEnv, "budget": 4, "messages": claim_messages, "finish": _claim_episode, "info_in_traj": True},
    "model": {"env": ModelEnv, "budget": 40, "messages": model_messages, "finish": _model_episode, "info_in_traj": False},
}


def run_batched(kind: str, tasks: list, generate_many: GenerateMany, policy_name: str, batch_size: int = 16,
                on_episode: Callable[[dict], None] | None = None,
                on_progress: Callable[[dict], None] | None = None) -> list[dict]:
    spec = SUITES[kind]
    live = []
    for t in tasks:
        env = spec["env"](t, spec["budget"])
        live.append(_Live(t, env, env.observe()))
    done_eps: list[dict] = []
    total, calls, rnd, t0 = len(live), 0, 0, time.monotonic()
    pending = list(live)
    while pending:
        rnd += 1
        pending.sort(key=lambda lv: len(json.dumps(lv.obs, default=str)))   # similar lengths -> less padding
        for i in range(0, len(pending), batch_size):
            chunk = pending[i:i + batch_size]
            texts = generate_many([spec["messages"](lv.obs) for lv in chunk])
            if len(texts) != len(chunk):
                raise RuntimeError(f"generator returned {len(texts)} outputs for {len(chunk)} prompts")
            calls += len(chunk)
            for lv, text in zip(chunk, texts, strict=True):
                act = parse_action(text)
                nxt, r, done, info = lv.env.step(act)
                step = {"obs": lv.obs, "action": act, "reward": r}
                if spec["info_in_traj"]:
                    step["info"] = {k: v for k, v in info.items() if k != "action"}
                lv.traj.append(step)
                lv.total += r
                lv.obs, lv.last_info = nxt, info
                if done:
                    ep = spec["finish"](lv, policy_name)
                    done_eps.append(ep)
                    if on_episode:
                        on_episode(ep)
            if on_progress:
                el = time.monotonic() - t0
                n = len(done_eps)
                on_progress({"done": n, "total": total, "round": rnd, "calls": calls, "elapsed_s": el,
                             "calls_per_s": calls / el if el > 0 else 0.0,
                             "eta_s": (el / n) * (total - n) if n else None})
        pending = [lv for lv in pending if not lv.env.done]
    return done_eps


def thread_generator(generate_one: Callable[[Messages], str], concurrency: int = 16) -> GenerateMany:
    """Wrap a single-request generator (e.g. an OpenAI-compatible endpoint) into a concurrent batch generator."""
    pool = ThreadPoolExecutor(max_workers=max(1, concurrency))

    def gen(batch: list[Messages]) -> list[str]:
        return list(pool.map(generate_one, batch))
    return gen
