"""Reference policies for ClaimEnv. ``GatewayPolicy`` lets Qwen (or any OpenAI-compatible model) act; the others
are deterministic baselines that bound what a learned policy must beat."""

from __future__ import annotations

import json


class AlwaysHighest:
    """Evaluate the most expensive (highest-fidelity) valid tool, then declare from it."""
    name = "always_highest"

    def act(self, obs):
        done = {e["tool"] for e in obs["evidence"]}
        for e in obs["evidence"]:
            if e.get("interval"):
                return _declare(obs, e)
        left = [t for t in sorted(obs["tools"], key=lambda t: -t["cost"]) if t["id"] not in done]
        return {"type": "evaluate", "tool": left[0]["id"]} if left else \
            {"type": "declare", "verdict": "insufficient_evidence"}


class CheapestSufficient:
    """The evidence calculus as a policy: cheapest first, declare at the first decisive interval, abstain on
    conflict or exhaustion."""
    name = "cheapest_sufficient"

    def act(self, obs):
        ivs = [e for e in obs["evidence"] if e.get("interval")]
        if any(a["interval"][1] < b["interval"][0] or b["interval"][1] < a["interval"][0]
               for i, a in enumerate(ivs) for b in ivs[i + 1:]):
            return {"type": "declare", "verdict": "insufficient_evidence"}
        for e in ivs:
            d = _declare(obs, e)
            if d["verdict"] != "insufficient_evidence":
                return d
        done = {e["tool"] for e in obs["evidence"]}
        left = [t for t in sorted(obs["tools"], key=lambda t: (t["cost"], t["id"])) if t["id"] not in done]
        if not left or obs["steps_left"] <= 1:
            return {"type": "declare", "verdict": "insufficient_evidence"}
        return {"type": "evaluate", "tool": left[0]["id"]}


class Overconfident:
    """Evaluates the cheapest tool and declares from its POINT value, ignoring uncertainty — the behaviour a
    fluent but ungrounded model exhibits. The environment must punish it even when it is right."""
    name = "overconfident"

    def act(self, obs):
        ev = [e for e in obs["evidence"] if e.get("status") == "evaluated"]
        if ev:
            c = obs["claim"]
            ok = ev[0]["value"] <= c["target"] if c["comparator"] in ("<=", "<") else ev[0]["value"] >= c["target"]
            return {"type": "declare", "verdict": "pass" if ok else "fail"}
        t = min(obs["tools"], key=lambda t: (t["cost"], t["id"]))
        return {"type": "evaluate", "tool": t["id"]}


class GatewayPolicy:
    """An LLM policy: the observation goes in as JSON, one JSON action comes out. Not evaluated in this sandbox
    (no model endpoint); run with BERKELIUM_MODEL_URL set."""
    name = "llm_gateway"
    SYSTEM = ("You are Berkelium's engineering agent. You decide engineering claims by gathering evidence with "
              "tools. A pass/fail verdict is only acceptable if an evaluated tool's INTERVAL decides it. Reply with "
              'exactly one JSON object: {"type":"evaluate","tool":ID} or {"type":"declare","verdict":'
              '"pass"|"fail"|"insufficient_evidence"}.')

    def __init__(self, provider=None):
        from ..ai.gateway import provider_from_env
        self.p = provider or provider_from_env()

    def act(self, obs):
        from ..ai.orchestrator import extract_json
        g = self.p.generate([{"role": "system", "content": self.SYSTEM},
                             {"role": "user", "content": json.dumps(obs, default=str)}], None, None)
        try:
            return extract_json(g.text)
        except ValueError:
            return {"type": "invalid"}


def _declare(obs, e):
    lo, hi = e["interval"]
    c = obs["claim"]
    if c["comparator"] in ("<=", "<"):
        v = "pass" if hi <= c["target"] else "fail" if lo > c["target"] else "insufficient_evidence"
    else:
        v = "pass" if lo >= c["target"] else "fail" if hi < c["target"] else "insufficient_evidence"
    return {"type": "declare", "verdict": v}
