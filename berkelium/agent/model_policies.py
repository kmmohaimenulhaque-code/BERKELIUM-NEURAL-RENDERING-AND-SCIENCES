"""Policies for ModelEnv. ConstructorPolicy = the domain-free synthesiser acting THROUGH the env's actions (it
produces the teacher traces); GreedyRetrieval = first-match retrieval that ignores context/validity (the
hard-negative behaviour); LLMModelPolicy = Qwen via the gateway (not evaluated here)."""

from __future__ import annotations

import json

from ..synthesis import Problem, construct
from ..units import Quantity


def _problem(obs) -> Problem:
    p = obs["problem"]
    return Problem(frozenset(p["context"]), {k: Quantity.of(v["value"], v["unit"]) for k, v in p["known"].items()},
                   p["target"])


class ConstructorPolicy:
    name = "constructor"

    def __init__(self):
        self.plan = None

    def act(self, obs):
        if self.plan is None:
            c = construct(_problem(obs))
            self.result = c
            rels = list(c.chosen) or (list(min(c.candidates, key=lambda x: len(x.relations)).relations)
                                      if c.candidates else [])
            if not rels:   # nothing determines the target: show it with the admissible relations around it
                from ..synthesis import admissible
                adm = admissible(_problem(obs))
                tgt = obs["problem"]["target"]
                first = [f for f in adm if tgt in f.relation.vars]
                near = set().union(*(f.relation.vars for f in first)) if first else set()
                rels = [f.relation.id for f in adm if f in first or (f.relation.vars & near - {tgt}
                                                                     and f.relation.fidelity == "definition")]
            self.plan = [{"type": "search", "var": obs["problem"]["target"]}] + \
                [{"type": "add", "id": r} for r in rels] + [{"type": "solve"}]
        if self.plan:
            return self.plan.pop(0)
        c = self.result
        if c.status == "determined":
            return {"type": "declare", "status": "determined", "value": obs["last_solve"].get("target_value")}
        return {"type": "declare", "status": c.status}

    def reset(self):
        self.plan = None


class GreedyRetrieval:
    """Adds the first fragment mentioning each still-unknown variable (library order), ignoring context and
    validity; declares whatever the solve returns."""
    name = "greedy_retrieval"

    def __init__(self):
        self.phase = "search_target"

    def act(self, obs):
        ls, p = obs["last_solve"], obs["problem"]
        if self.phase == "search_target":
            self.phase = "add"
            self.pending = [p["target"]]
            return {"type": "search", "var": p["target"]}
        if self.phase == "add":
            for f in obs["last_search"]:
                if not f["quarantined"] and f["id"] not in obs["model"]:
                    self.phase = "solve"
                    return {"type": "add", "id": f["id"]}
            self.phase = "solve"
        if self.phase == "solve":
            self.phase = "after_solve"
            return {"type": "solve"}
        if ls.get("status") == "solved" and "target_value" in ls:
            return {"type": "declare", "status": "determined", "value": ls["target_value"]}
        if ls.get("status") == "underdetermined" and obs["steps_left"] > 3:
            import re
            m = re.search(r"specify \d+ of \[(.*?)\]", " ".join(ls.get("diagnostics", [])))
            cands = [x.strip(" '") for x in m.group(1).split(",")] if m else []
            cands = [c for c in cands if c not in p["known"]]
            if cands:
                self.phase = "add"
                return {"type": "search", "var": cands[0]}
        return {"type": "declare", "status": ls.get("status", "underdetermined").replace("solved", "determined")
                .replace("invalid_domain", "outside_validity").replace("conflict", "contradictory")}

    def reset(self):
        self.phase = "search_target"


class LLMModelPolicy:
    name = "llm_model_builder"
    SYSTEM = ("You are Berkelium's model-construction agent. Build the model for the problem by searching the relation "
              "library, adding fragments whose context matches the problem and whose validity holds, solving, and "
              "repairing. Only declare what your own last solve shows. Reply with ONE JSON action: "
              '{"type":"search","var":X} | {"type":"add","id":R} | {"type":"remove","id":R} | {"type":"solve"} | '
              '{"type":"declare","status":"determined"|"underdetermined"|"contradictory"|"outside_validity"|"ambiguous",'
              '"value":number_if_determined}')

    def __init__(self, provider=None):
        from ..ai.gateway import provider_from_env
        self.p = provider or provider_from_env()

    def act(self, obs):
        from ..ai.orchestrator import extract_json
        g = self.p.generate(model_messages(obs), None, None)
        try:
            return extract_json(g.text)
        except ValueError:
            return {"type": "invalid"}

    def reset(self):
        pass


def model_messages(obs: dict) -> list[dict]:
    return [{"role": "system", "content": LLMModelPolicy.SYSTEM},
            {"role": "user", "content": json.dumps(obs, default=str, sort_keys=True)}]
