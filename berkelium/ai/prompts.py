"""Single source of truth for model-facing prompts — used by BOTH the dataset factory (training targets)
and the Model Gateway (inference), so training and serving never drift apart."""

from __future__ import annotations

import json

from ..cem.protocol import CEMRegistry
from ..schema.common import SCHEMA_VERSION

PROMPT_VERSION = "0.1.0"

SYSTEM_PLAN = """You are Berkelium's design planner. Convert the user's request into ONE JSON object that is a \
Berkelium DesignProposal (schema {schema}). Rules:
- Output only JSON. Never output measurements, validation results, simulation results or engineering verdicts: \
deterministic Berkelium code computes those.
- For a catalogued engineering object, add a component with kind "cem", the CEM reference, and put the user's \
needs in "requirements" (numbers as {{"value": v, "unit": "u"}}). Do not compute derived engineering values \
(tooth counts, shifts, stresses) that the CEM derives itself.
- For anything not catalogued, add a component with kind "procedural" and an authored "geometry" op graph.
- Copy every quantitative user requirement into specification.requirements so it can be checked.
- Put ambiguities in intent.open_questions and explicit assumptions in intent.assumptions.
Available CEMs:
{cems}"""

SYSTEM_REPAIR = """You are Berkelium's design repairer. You receive a DesignProposal and the deterministic \
validation report. Reply with ONLY a JSON array of RFC 6902 JSON Patch operations against the proposal that \
fixes the failures. Do not edit or add evaluation results; change only intent, specification or structure."""


def cem_catalogue(registry: CEMRegistry) -> str:
    lines = []
    for item in registry.summary():
        props = item["requirements_schema"].get("properties", {})
        fields = []
        for k, v in props.items():
            unit = v.get("unit") or next((x.get("unit") for x in v.get("anyOf", []) if isinstance(x, dict)
                                          and x.get("unit")), None)
            fields.append(f"{k}" + (f"[{unit}]" if unit else ""))
        req = item["requirements_schema"].get("required", [])
        lines.append(f"- {item['ref']} ({item['kind']}): {item['summary']} requirements: {', '.join(fields)}; "
                     f"required: {', '.join(req)}")
    return "\n".join(lines)


def plan_messages(intent: str, registry: CEMRegistry) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PLAN.format(schema=SCHEMA_VERSION, cems=cem_catalogue(registry))},
            {"role": "user", "content": intent}]


def report_digest(record) -> str:
    """Compact, deterministic summary of what failed — the repair model's evidence."""
    v = record.evaluation.validation
    lines = [f"summary: {v.summary}"]
    for r in v.results:
        if r.status in ("fail", "error"):
            m = f" measured={r.measured.value:.6g} {r.measured.unit}" if r.measured else ""
            lim = f" limit {r.comparator} {r.limit.value:.6g} {r.limit.unit}" if r.limit else ""
            lines.append(f"- L{r.level} {r.status} {r.target}: {r.message}{m}{lim}")
    for d in record.evaluation.diagnostics:
        if d.severity == "error":
            lines.append(f"- {d.code}: {d.message}")
    return "\n".join(lines)


def repair_messages(proposal: dict, digest: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_REPAIR},
            {"role": "user", "content": "PROPOSAL:\n" + json.dumps(proposal, sort_keys=True, separators=(",", ":"))
             + "\nVALIDATION:\n" + digest}]
