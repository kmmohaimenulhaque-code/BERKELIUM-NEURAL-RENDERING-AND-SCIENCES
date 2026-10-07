"""ClaimEnv: an environment in which an agent must ENGINEER a decision, not describe one (ADR-022).

Episode: a Claim at a design point, a catalogue of evidence tools (Laws: id, fidelity, cost, validity
predicates — outputs hidden until evaluated) and a step budget. Actions (JSON-serialisable, so an LLM can emit
them): {"type": "evaluate", "tool": id} | {"type": "declare", "verdict": "pass"|"fail"|"insufficient_evidence"}.

Reward (policy, not physics — stated explicitly and versioned):
  grounded correct pass/fail          +1 - cost_weight * (cost spent / max tool cost)
  UNGROUNDED pass/fail (no gathered evidence interval decides it)   -1   even if it happens to be right
  wrong grounded pass/fail            -2   (cannot happen with correct tools; flags tool or truth bugs)
  insufficient_evidence while a tool that would decide was available and affordable   -0.5
  insufficient_evidence when truly undecidable with the catalogue                     +0.5
  conflict detected and declared insufficient_evidence                                +0.5
  budget exhausted without declaring  -1
Ground truth comes from the highest-fidelity verified evidence, never from a model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..evidence.core import Claim, Est, Law

REWARD_VERSION = "claimenv-r1"


@dataclass
class Task:
    id: str
    claim: Claim
    point: dict
    tools: list[Law]
    truth: str                      # pass | fail | insufficient_evidence (from verified reference evidence)
    family: str = ""


@dataclass
class ClaimEnv:
    task: Task
    budget: int = 4
    cost_weight: float = 0.5
    spent: float = 0.0
    steps: int = 0
    evidence: list[dict] = field(default_factory=list)
    done: bool = False

    def observe(self) -> dict:
        t = self.task
        return {"claim": {"quantity": t.claim.quantity, "comparator": t.claim.comparator, "target": t.claim.target,
                          "unit": t.claim.unit},
                "point": {k: {"si": v.si, "dim_LMTIONJ": list(v.dim)} for k, v in sorted(t.point.items())},
                "tools": [{"id": x.id, "fidelity": x.fidelity, "cost": x.cost, "validity": x.validity}
                          for x in t.tools if x.quantity == t.claim.quantity],
                "evidence": [dict(e) for e in self.evidence], "steps_left": self.budget - self.steps}

    def _decisive(self) -> set[str]:
        out = set()
        for e in self.evidence:
            if e.get("interval"):
                out.add(self.task.claim.judge(tuple(e["interval"])))
        return out & {"pass", "fail"}

    def _conflict(self) -> bool:
        iv = [e["interval"] for e in self.evidence if e.get("interval")]
        return any(a[1] < b[0] or b[1] < a[0] for i, a in enumerate(iv) for b in iv[i + 1:])

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        if self.done:
            raise RuntimeError("episode finished")
        self.steps += 1
        info: dict = {"action": action}
        if action.get("type") == "evaluate":
            tool = next((x for x in self.task.tools if x.id == action.get("tool")), None)
            if tool is None:
                info["error"] = "unknown tool"
            else:
                ok, why = tool.check(self.task.point)
                if not ok:
                    self.evidence.append({"tool": tool.id, "status": "invalid_domain", "reason": why})
                else:
                    try:
                        e: Est = tool.run(self.task.point)
                        self.spent += tool.cost
                        iv = e.interval()
                        self.evidence.append({"tool": tool.id, "status": "evaluated", "value": e.value,
                                              "unit": e.unit, "interval": list(iv) if iv else None,
                                              "fidelity": tool.fidelity})
                    except Exception as err:  # noqa: BLE001 — a failing tool is evidence too
                        self.evidence.append({"tool": tool.id, "status": "error", "reason": str(err)})
            if self.steps >= self.budget:
                self.done = True
                return self.observe(), -1.0, True, {**info, "outcome": "budget_exhausted"}
            return self.observe(), 0.0, False, info
        if action.get("type") == "declare":
            self.done = True
            v = action.get("verdict")
            decisive = self._decisive()
            maxcost = max(x.cost for x in self.task.tools)
            if v in ("pass", "fail"):
                if v not in decisive or self._conflict():
                    return self.observe(), -1.0, True, {**info, "outcome": "ungrounded",
                                                        "correct_by_luck": v == self.task.truth}
                if v != self.task.truth:
                    return self.observe(), -2.0, True, {**info, "outcome": "wrong_grounded"}
                r = 1.0 - self.cost_weight * min(self.spent / maxcost, 1.0)
                return self.observe(), r, True, {**info, "outcome": "correct", "cost": self.spent}
            if v == "insufficient_evidence":
                if self._conflict() or self.task.truth == "insufficient_evidence":
                    return self.observe(), 0.5, True, {**info, "outcome": "honest_abstention"}
                return self.observe(), -0.5, True, {**info, "outcome": "lazy_abstention"}
            return self.observe(), -1.0, True, {**info, "outcome": "invalid_action"}
        self.done = self.steps >= self.budget
        return self.observe(), -1.0 if self.done else 0.0, self.done, {**info, "error": "invalid action type"}


def run_episode(task: Task, policy, budget: int = 4) -> dict:
    env = ClaimEnv(task, budget)
    obs, traj, total = env.observe(), [], 0.0
    while not env.done:
        a = policy.act(obs)
        obs2, r, done, info = env.step(a)
        traj.append({"obs": obs, "action": a, "reward": r, "info": {k: v for k, v in info.items() if k != "action"}})
        total += r
        obs = obs2
    out = traj[-1]["info"].get("outcome", "budget_exhausted")
    return {"task": task.id, "family": task.family, "policy": policy.name, "truth": task.truth, "return": total,
            "outcome": out, "cost": env.spent, "steps": len(traj), "reward_version": REWARD_VERSION,
            "trajectory": json.loads(json.dumps(traj, default=str))}
