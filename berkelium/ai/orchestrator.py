"""Orchestrator: intent -> model DesignProposal -> deterministic core -> repair loop (JSON Patch).
The model output is parsed and validated by the core; it can never write evaluation data."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field

import jsonpatch
from pydantic import ValidationError

from ..cem.protocol import CEMRegistry, default_registry
from ..pipeline import run
from ..schema.design import DesignProposal
from ..schema.export import json_schema
from .gateway import DecodingConfig, GatewayError, ModelProvider
from .prompts import plan_messages, report_digest, repair_messages

FORBIDDEN_KEYS = ("evaluation", "measurements", "validation", "simulation", "physically_validated")
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.S)


def extract_json(text: str):
    """Parse the first JSON value in model text (tolerates code fences and Qwen3 <think> blocks)."""
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    t = _FENCE.sub("", t.strip())
    dec = json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch in "{[":
            try:
                return dec.raw_decode(t[i:])[0]
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON value in model output")


def mentions_evaluation(obj) -> bool:
    if isinstance(obj, dict):
        return any(k in FORBIDDEN_KEYS or mentions_evaluation(v) for k, v in obj.items())
    if isinstance(obj, list):
        return any(mentions_evaluation(v) for v in obj)
    return False


@dataclass
class Attempt:
    kind: str                 # plan | repair
    raw: str
    parsed: bool
    schema_valid: bool
    summary: str | None
    error: str | None = None
    hallucinated_evaluation: bool = False


@dataclass
class DesignSession:
    intent: str
    attempts: list[Attempt] = field(default_factory=list)
    proposal: dict | None = None
    record: object | None = None

    @property
    def final_summary(self) -> str | None:
        return self.attempts[-1].summary if self.attempts else None


class Orchestrator:
    def __init__(self, provider: ModelProvider, registry: CEMRegistry | None = None, realize: bool = False,
                 backend: str | None = "manifold", constrained: bool = True, decoding: DecodingConfig | None = None):
        self.p = provider
        self.reg = registry or default_registry()
        self.realize, self.backend = realize, backend
        self.schema = json_schema("DesignProposal") if constrained else None
        self.decoding = decoding or DecodingConfig()

    def _evaluate(self, obj, kind, raw, sess) -> bool:
        halluc = mentions_evaluation(obj)
        try:
            prop = DesignProposal.model_validate(obj)
        except ValidationError as e:
            sess.attempts.append(Attempt(kind, raw, True, False, None, str(e).splitlines()[0], halluc))
            return False
        res = run(prop, registry=self.reg, realize=self.realize, backend=self.backend)
        sess.proposal, sess.record = prop.model_dump(mode="json"), res.record
        s = res.record.evaluation.validation.summary
        sess.attempts.append(Attempt(kind, raw, True, True, s, None, halluc))
        return s in ("pass", "warn")

    def design(self, intent: str, max_repairs: int = 2) -> DesignSession:
        sess = DesignSession(intent)
        try:
            g = self.p.generate(plan_messages(intent, self.reg), self.schema, self.decoding)
        except GatewayError as e:
            sess.attempts.append(Attempt("plan", "", False, False, None, str(e)))
            return sess
        try:
            obj = extract_json(g.text)
        except ValueError as e:
            sess.attempts.append(Attempt("plan", g.text, False, False, None, str(e)))
            return sess
        if self._evaluate(obj, "plan", g.text, sess):
            return sess
        for _ in range(max_repairs):
            if sess.proposal is None:
                break  # schema-invalid plans are not repairable by patch (no base document)
            digest = report_digest(sess.record)
            try:
                g = self.p.generate(repair_messages(sess.proposal, digest), None, self.decoding)
                patch = extract_json(g.text)
                patched = jsonpatch.apply_patch(copy.deepcopy(sess.proposal), patch)
            except (GatewayError, ValueError, jsonpatch.JsonPatchException, jsonpatch.JsonPointerException,
                    TypeError) as e:
                sess.attempts.append(Attempt("repair", getattr(g, "text", ""), False, False, None, str(e)))
                continue
            if self._evaluate(patched, "repair", g.text, sess):
                break
        return sess
