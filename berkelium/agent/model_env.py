"""ModelEnv (ADR-023): the agent must BUILD the model, not retrieve an answer.

Observation: problem (context facts, knowns with units, target), the agent's current model (fragment ids), the last
search result and the last solve report. Actions (JSON):
  {"type":"search","var":X}        -> every library fragment mentioning X (incl. wrong-context and quarantined
                                      ones, flagged) — the agent must choose
  {"type":"add","id":R} / {"type":"remove","id":R}
  {"type":"solve"}                 -> the deterministic core verifies the agent's OWN model: determinacy, roots,
                                      validity at the solution, context violations, definition closure
  {"type":"declare","status":S[,"value":v]}   S in determined | underdetermined | contradictory |
                                      outside_validity | ambiguous
Truth comes from each scenario's INDEPENDENT reference (explicit relation set + expected status), never from
the constructor being evaluated.
Reward (modelenv-r1): correct status (+ value within 1e-6 rel for 'determined') AND grounded (the agent's last
solve produced exactly that) -> +1 - 0.02*steps; correct but ungrounded -> -1; wrong -> -2; budget exhausted -> -1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..laws import LawSet, solve
from ..synthesis.library import FRAGMENTS_SRC, VARS, load
from ..units import Quantity

REWARD_VERSION = "modelenv-r1"
_OK, _QUAR = load()
QUARANTINED = {r for r, _ in _QUAR}
BY_ID = {f.relation.id: f for f in FRAGMENTS_SRC}


@dataclass
class ModelTask:
    id: str
    domain: str
    context: frozenset[str]
    known: dict[str, Quantity]
    target: str
    truth_status: str
    truth_value: float | None
    split: str


@dataclass
class ModelEnv:
    task: ModelTask
    budget: int = 40
    model: list[str] = field(default_factory=list)
    last_search: list[dict] = field(default_factory=list)
    last_solve: dict = field(default_factory=dict)
    steps: int = 0
    done: bool = False

    def observe(self) -> dict:
        t = self.task
        return {"problem": {"context": sorted(t.context), "target": t.target, "target_unit": VARS[t.target].unit,
                            "known": {k: {"value": v.to(VARS[k].unit), "unit": VARS[k].unit}
                                      for k, v in sorted(t.known.items())}},
                "model": list(self.model), "last_search": self.last_search, "last_solve": self.last_solve,
                "steps_left": self.budget - self.steps}

    def _solve(self) -> dict:
        t = self.task
        if not self.model:
            return {"status": "empty_model"}
        rels = [BY_ID[r].relation for r in self.model]
        ctx_bad = sorted(r for r in self.model if not BY_ID[r].context <= t.context)
        names = set().union(*(r.vars for r in rels)) | set(t.known)
        ls = LawSet("agent_model", {n: VARS[n] for n in names}, rels)
        s = solve(ls, {k: v for k, v in t.known.items()})
        rep = {"status": s.status, "diagnostics": s.diagnostics, "context_violations": ctx_bad,
               "validity_failures": [f"{c.id}: {c.detail}" for c in s.checks if c.kind == "validity" and c.status != "pass"]}
        if t.target in s.values:
            rep["target_value"] = s.values[t.target].to(VARS[t.target].unit)
        closure = []
        for f in _OK:
            r = f.relation
            if r.id not in self.model and r.fidelity in ("definition", "exact") and f.context <= t.context \
                    and r.vars <= set(s.values) and abs(r.residual(s.values)) > 1e-9 \
                    and all(_holds(pr, s.values) for pr in r.validity):   # domain-limited identities only count
                closure.append(r.id)                                         # inside their validity domain
        rep["closure_violations"] = closure
        return rep

    def step(self, a: dict):
        if self.done:
            raise RuntimeError("done")
        self.steps += 1
        typ = a.get("type")
        if typ == "search":
            v = a.get("var", "")
            self.last_search = [{"id": f.relation.id, "text": f.relation.text, "context": sorted(f.context),
                                 "validity": list(f.relation.validity), "fidelity": f.relation.fidelity,
                                 "model_form_rel": f.relation.model_form_rel, "quarantined": f.relation.id in QUARANTINED}
                                for f in FRAGMENTS_SRC if v in f.relation.vars]
        elif typ == "add":
            r = a.get("id")
            if r in BY_ID and r not in QUARANTINED and r not in self.model:
                self.model.append(r)
        elif typ == "remove":
            if a.get("id") in self.model:
                self.model.remove(a["id"])
        elif typ == "solve":
            self.last_solve = self._solve()
        elif typ == "declare":
            self.done = True
            return self.observe(), self._score(a), True, {"outcome": self._outcome}
        if self.steps >= self.budget:
            self.done = True
            self._outcome = "budget_exhausted"
            return self.observe(), -1.0, True, {"outcome": self._outcome}
        return self.observe(), 0.0, False, {}

    def _score(self, a: dict) -> float:
        t, st, ls = self.task, a.get("status"), self.last_solve
        correct = st == t.truth_status and (st != "determined" or (
            a.get("value") is not None and t.truth_value is not None
            and abs(float(a["value"]) - t.truth_value) <= 1e-6 * max(abs(t.truth_value), 1e-300)))
        grounded = bool(ls) and not ls.get("context_violations") and (
            (st == "determined" and ls.get("status") == "solved" and not ls.get("validity_failures")
             and not ls.get("closure_violations") and a.get("value") is not None and "target_value" in ls
             and abs(float(a["value"]) - ls["target_value"]) <= 1e-9 * max(abs(ls["target_value"]), 1e-300))
            or (st == "underdetermined" and ls.get("status") == "underdetermined")
            or (st == "contradictory" and bool(ls.get("closure_violations") or ls.get("status") == "conflict"))
            or (st == "outside_validity" and bool(ls.get("validity_failures") or ls.get("status") == "invalid_domain"))
            or (st == "ambiguous" and ls.get("status") == "ambiguous"))
        if correct and grounded:
            self._outcome = "correct"
            return 1.0 - 0.02 * self.steps
        if correct:
            self._outcome = "ungrounded"
            return -1.0
        self._outcome = "wrong"
        return -2.0


def _holds(pred: str, env) -> bool:
    from ..expr import eval_str
    try:
        return eval_str(pred, env) is True
    except Exception:  # noqa: BLE001
        return False


def run_model_episode(task: ModelTask, policy, budget: int = 40) -> dict:
    env = ModelEnv(task, budget)
    obs, traj, total, info = env.observe(), [], 0.0, {}
    while not env.done:
        act = policy.act(obs)
        nxt, r, done, info = env.step(act)
        traj.append({"obs": obs, "action": act, "reward": r})
        total += r
        obs = nxt
    return {"task": task.id, "domain": task.domain, "split": task.split, "policy": policy.name,
            "truth": task.truth_status, "return": total, "outcome": info.get("outcome", "budget_exhausted"),
            "steps": len(traj), "reward_version": REWARD_VERSION,
            "trajectory": json.loads(json.dumps(traj, default=str))}
