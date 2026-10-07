"""Baseline / post-training evaluation on a dataset split (architecture §10.3 metrics).

Requirement satisfaction is judged against the GROUND-TRUTH requirements of the example (taken from the
verified target), not against whatever requirements the model wrote for itself."""

from __future__ import annotations

import json
import platform
import time
from collections import Counter
from pathlib import Path

from .. import __version__
from ..pipeline import run
from ..schema.design import DesignProposal
from ..schema.hashing import sha256_of
from .orchestrator import Orchestrator


def _gt_requirements_met(session, example) -> bool:
    if session.proposal is None:
        return False
    gt = example["target"].get("specification", {}).get("requirements", [])
    if not gt:
        return session.final_summary in ("pass", "warn")
    prop = dict(session.proposal)
    gt_comp = {c["id"] for c in example["target"]["structure"]["components"]}
    comps = [c["id"] for c in prop["structure"]["components"]]
    remap = {}
    if len(gt_comp) == 1 and len(comps) >= 1:  # model may name its component differently
        remap = {next(iter(gt_comp)): comps[0]}
    reqs = [dict(r, applies_to=remap.get(r.get("applies_to"), r.get("applies_to"))) for r in gt]
    prop["specification"] = dict(prop.get("specification", {}), requirements=reqs)
    rec = run(DesignProposal.model_validate(prop), realize=False).record
    rr = [r for r in rec.evaluation.validation.results if r.validator.startswith("requirement.")]
    return bool(rr) and all(r.status == "pass" for r in rr)


class EndpointLost(RuntimeError):
    """Raised when the model endpoint keeps failing: a baseline built on transport errors would be
    recorded as a model scoring 0 %, which is a false measurement."""


def evaluate(examples: list[dict], orch: Orchestrator, max_repairs: int = 2, out: str | Path | None = None,
             label: str = "baseline", max_consecutive_transport_errors: int = 3) -> dict:
    rows = []
    consecutive = 0
    for ex in examples:
        if ex["task"] not in ("plan", "plan_procedural"):
            continue
        intent = ex["messages"][-1]["content"]
        t = time.perf_counter()
        s = orch.design(intent, max_repairs=max_repairs)
        if any(a.transport_error for a in s.attempts):
            consecutive += 1
            if consecutive >= max_consecutive_transport_errors:
                raise EndpointLost(f"{consecutive} consecutive examples hit endpoint errors; last: "
                                   f"{next(a.error for a in s.attempts if a.transport_error)}. No report written.")
        else:
            consecutive = 0
        first = s.attempts[0] if s.attempts else None
        gt_cems = sorted(c.get("cem") or "procedural" for c in ex["target"]["structure"]["components"])
        got_cems = sorted(c.get("cem") or "procedural" for c in (s.proposal or {}).get("structure", {})
                          .get("components", [])) if s.proposal else []
        rows.append({
            "id": ex["id"], "split": ex["split"], "task": ex["task"], "family_id": ex["family_id"],
            "parsed": bool(first and first.parsed), "schema_valid_first": bool(first and first.schema_valid),
            "pass_first": bool(first and first.summary in ("pass", "warn")),
            "pass_final": s.final_summary in ("pass", "warn"),
            "repairs_used": sum(a.kind == "repair" for a in s.attempts),
            "cem_selection_correct": got_cems == gt_cems,
            "requirements_met": _gt_requirements_met(s, ex),
            "hallucinated_evaluation": any(a.hallucinated_evaluation for a in s.attempts),
            "transport_error": any(a.transport_error for a in s.attempts),
            "seconds": round(time.perf_counter() - t, 3),
            "attempts": [a.__dict__ for a in s.attempts],
        })
    n = len(rows) or 1

    def rate(k):
        return round(sum(r[k] for r in rows) / n, 4)

    failed_first = [r for r in rows if not r["pass_first"] and r["schema_valid_first"]]
    metrics = {
        "n": len(rows), "parse_rate": rate("parsed"), "schema_valid_rate": rate("schema_valid_first"),
        "validation_pass_first": rate("pass_first"), "validation_pass_final": rate("pass_final"),
        "requirement_satisfaction": rate("requirements_met"), "cem_selection_accuracy": rate("cem_selection_correct"),
        "hallucinated_evaluation_rate": rate("hallucinated_evaluation"),
        "transport_error_rate": rate("transport_error"),   # must be 0 for a valid baseline
        "repair_success": round(sum(r["pass_final"] for r in failed_first) / len(failed_first), 4)
        if failed_first else None,
        "by_split": {s: {"n": c} for s, c in Counter(r["split"] for r in rows).items()},
    }
    report = {"label": label, "berkelium": __version__, "provider": orch.p.name, "model": orch.p.model,
              "constrained_decoding": orch.schema is not None, "decoding": orch.decoding.__dict__,
              "max_repairs": max_repairs, "python": platform.python_version(),
              "examples_hash": sha256_of([e["id"] for e in examples]), "metrics": metrics, "rows": rows}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
