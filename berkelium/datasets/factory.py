"""Verified dataset factory (architecture §10, ADR-009).

Every example's target is executed by the deterministic core; only verified examples are written to
training splits. Rejected candidates go to ``rejected.jsonl`` with the reason (never trained on).

Tasks
  plan    : intent -> DesignProposal (CEM requirements form; the model must not compute derived values)
  repair  : (proposal, validation digest) -> JSON Patch; target verified by re-running the core
Splits     : by family_id hash (train/val/test), plus ``heldout_family`` = procedural (non-CEM) designs.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonpatch
import numpy as np

from .. import __version__
from ..ai.prompts import PROMPT_VERSION, plan_messages, repair_messages, report_digest
from ..cem.library.gear import formulas as F
from ..cem.library.gear.cem import ALPHA, DEFAULT_CANDIDATE_MODULES, Parameters
from ..cem.protocol import CEMRegistry, default_registry
from ..designs.procedural import lantern_proposal
from ..pipeline import run
from ..schema.common import SCHEMA_VERSION
from ..schema.hashing import sha256_bytes, sha256_of

GENERATOR = f"berkelium.datasets.factory@{__version__}"
GEAR = "gear.spur_pair@0.1"


@dataclass
class FactoryConfig:
    seed: int = 0
    n_valid: int = 40
    n_boundary: int = 10
    n_invalid: int = 20          # become repair examples (broken -> verified fix)
    n_procedural: int = 6        # held-out family
    realize: bool = True         # realise geometry for every verified example
    backend: str | None = "manifold"


@dataclass
class Example:
    record: dict
    passed: bool
    reason: str = ""


def _q(v: float, u: str) -> dict:
    return {"value": float(v), "unit": u}


def family_id(cem: str, p: Parameters) -> str:
    return f"{cem}/z{p.z1}-{p.z2}/m{p.module:g}"


def split_for(fid: str) -> str:
    h = int(hashlib.sha256(fid.encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 80 else "val" if h < 90 else "test"


# ------------------------------------------------------------------ targets
def requirements_from_params(p: Parameters, rng: np.random.Generator) -> dict:
    """A user-level request that this parameter set satisfies (what a person would ask for)."""
    req: dict[str, Any] = {"ratio": round(p.z2 / p.z1, 6), "module": _q(p.module, "mm")}
    aw, _ = F.working_pressure_angle(p.z1, p.z2, p.x1, p.x2, ALPHA)
    if p.x1 or p.x2 or rng.random() < 0.3:
        req["center_distance"] = _q(round(F.centre_distance_working(p.module, p.z1, p.z2, ALPHA, aw), 4), "mm")
    if p.z1 < 18:
        req["min_pinion_teeth"] = p.z1
    if p.pinion_torque is not None:
        req["pinion_torque"] = _q(p.pinion_torque, "N*mm")
        req["pinion_speed"] = _q(p.pinion_speed, "rev/min")
        req["allowable_bending_stress"] = _q(p.allowable_bending_stress, "MPa")
    if rng.random() < 0.3:
        req["face_width"] = _q(round(p.face_width, 3), "mm")
    return req


def plan_target(req: dict, comp_id: str = "pair") -> dict:
    spec_reqs = [{"id": "r_ratio", "quantity": "ratio", "applies_to": comp_id, "comparator": "approx",
                  "target": _q(req["ratio"], "1"), "tolerance": _q(0.02 * req["ratio"], "1")}]
    if "center_distance" in req:
        spec_reqs.append({"id": "r_center_distance", "quantity": "center_distance", "applies_to": comp_id,
                          "comparator": "==", "target": req["center_distance"], "tolerance": _q(0.01, "mm")})
    if "module" in req:
        spec_reqs.append({"id": "r_module", "quantity": "module", "applies_to": comp_id, "comparator": "==",
                          "target": req["module"]})
    return {"schema_version": SCHEMA_VERSION, "kind": "design_proposal",
            "intent": {"summary": "spur gear pair", "assumptions": [], "open_questions": [], "extracted": []},
            "specification": {"requirements": spec_reqs, "parameters": [], "constraints": []},
            "structure": {"components": [{"id": comp_id, "kind": "cem", "cem": GEAR, "requirements": req}],
                          "relations": []}}


def params_proposal(p: Parameters, comp_id: str = "pair") -> dict:
    d = {k: v for k, v in p.model_dump().items() if v is not None and k != "pressure_angle"}
    return {"schema_version": SCHEMA_VERSION, "kind": "design_proposal",
            "structure": {"components": [{"id": comp_id, "kind": "cem", "cem": GEAR, "parameters": d}]}}


# ------------------------------------------------------------------ verification
def verify(target: dict, cfg: FactoryConfig, require_requirements: bool = True):
    res = run(target, realize=cfg.realize, backend=cfg.backend)
    rec = res.record
    if rec is None:
        return None, False, "schema invalid"
    v = rec.evaluation.validation
    if v.summary not in ("pass", "warn"):
        return rec, False, f"summary {v.summary}"
    if require_requirements:
        reqs = [r for r in v.results if r.validator.startswith("requirement.")]
        if any(r.status != "pass" for r in reqs):
            return rec, False, "requirement not satisfied"
    if cfg.realize and not any(r.level == 3 and r.status == "pass" for r in v.results):
        return rec, False, "geometry not realised"
    return rec, True, ""


def verification_block(rec, realized: bool) -> dict:
    v = rec.evaluation.validation
    return {"passed": True, "summary": v.summary, "report_hash": sha256_of(v), "record_hash": rec.content_hash,
            "counts": v.counts, "highest_level_evaluated": v.highest_level_evaluated, "geometry_realized": realized,
            "validators": sorted({r.validator for r in v.results})}


# ------------------------------------------------------------------ repair oracle
def repair_candidates(p: Parameters) -> list[Parameters]:
    """Deterministic fix strategies, tried in order; each must verify before it becomes a target."""
    c: list[Parameters] = []
    xmin1 = F.min_shift_no_undercut(p.z1, ALPHA)
    if p.x1 < xmin1:
        c.append(p.model_copy(update={"x1": math.ceil((xmin1 + 0.005) * 1000) / 1000}))
        c.append(p.model_copy(update={"z1": 18, "z2": max(18, round(18 * p.z2 / p.z1))}))
    if p.x1 > 0.8:
        c.append(p.model_copy(update={"x1": max(0.0, math.ceil((xmin1 + 0.005) * 1000) / 1000)}))
    for m in DEFAULT_CANDIDATE_MODULES:
        if m > p.module:
            c.append(p.model_copy(update={"module": m, "face_width": round(4 * math.pi * m, 3)}))
    out = []
    for q in c:
        try:
            out.append(Parameters.model_validate(q.model_dump()))
        except ValueError:
            continue
    return out


# ------------------------------------------------------------------ main
def build(out_dir: str | Path, cfg: FactoryConfig | None = None, registry: CEMRegistry | None = None) -> dict:
    cfg = cfg or FactoryConfig()
    reg = registry or default_registry()
    cem = reg.get(GEAR)
    rng = np.random.default_rng(cfg.seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    accepted: list[dict] = []
    rejected: list[dict] = []

    def emit(rec_dict: dict, ok: bool, reason: str) -> None:
        rec_dict["id"] = sha256_of({k: v for k, v in rec_dict.items() if k not in ("id", "verification")})[:24]
        (accepted if ok else rejected).append(rec_dict if ok else rec_dict | {"rejected_reason": reason})

    def base(task: str, mode: str, seed: int, fid: str, split: str) -> dict:
        return {"schema_version": SCHEMA_VERSION, "task": task, "prompt_version": PROMPT_VERSION,
                "source": {"generator": GENERATOR, "seed": cfg.seed, "item_seed": seed, "mode": mode},
                "cem_refs": [GEAR] if task != "plan_procedural" else [], "family_id": fid, "split": split}

    from .intents import gear_intent
    plan = [("valid", cfg.n_valid), ("boundary", cfg.n_boundary)]
    for mode, n in plan:
        for _ in range(n):
            item_seed = int(rng.integers(0, 2**31))
            r = np.random.default_rng(item_seed)
            p = cem.sample(r, mode)
            req = requirements_from_params(p, r)
            target = plan_target(req)
            fid = family_id(GEAR, p)
            ex = base("plan", mode, item_seed, fid, split_for(fid))
            ex["messages"] = plan_messages(gear_intent(req, r), reg)
            ex["target"] = target
            rec, ok, why = verify(target, cfg)
            if ok:
                ex["verification"] = verification_block(rec, cfg.realize)
            emit(ex, ok, why)

    for _ in range(cfg.n_invalid):
        item_seed = int(rng.integers(0, 2**31))
        r = np.random.default_rng(item_seed)
        bad = cem.sample(r, "invalid")
        broken = params_proposal(bad)
        brec = run(broken, realize=False).record
        fid = family_id(GEAR, bad)
        ex = base("repair", "invalid", item_seed, fid, split_for(fid))
        if brec is None or brec.evaluation.validation.summary not in ("fail", "error"):
            emit(ex | {"target": None}, False, "perturbation did not fail")
            continue
        fixed = None
        for cand in repair_candidates(bad):
            tgt = params_proposal(cand)
            rec, ok, _ = verify(tgt, cfg, require_requirements=False)
            if ok:
                fixed = (tgt, rec)
                break
        ex["messages"] = repair_messages(broken, report_digest(brec))
        if fixed is None:
            emit(ex | {"target": None}, False, "no verified repair found")
            continue
        ex["target"] = jsonpatch.make_patch(broken, fixed[0]).patch
        ex["verification"] = verification_block(fixed[1], cfg.realize) | {
            "broken_summary": brec.evaluation.validation.summary,
            "patched_hash": sha256_of(jsonpatch.apply_patch(copy.deepcopy(broken), ex["target"]))}
        emit(ex, True, "")

    for _ in range(cfg.n_procedural):
        item_seed = int(rng.integers(0, 2**31))
        r = np.random.default_rng(item_seed)
        h, rad = float(np.round(r.uniform(80, 200), 1)), float(np.round(r.uniform(30, 70), 1))
        t, slots = float(np.round(r.uniform(2, 5), 2)), int(r.integers(8, 25))
        prop = lantern_proposal(h, rad, t, slots).model_dump(mode="json")
        intent = (f"Model a lantern-like vessel about {h:g} mm tall and {2 * rad:g} mm across with a {t:g} mm wall, "
                  f"a bulging profile, a ring of {slots} tilted slots and a ring handle on top.")
        fid = "procedural/lantern"
        ex = base("plan_procedural", "valid", item_seed, fid, "heldout_family")
        ex["messages"] = plan_messages(intent, reg)
        ex["target"] = prop
        rec, ok, why = verify(prop, FactoryConfig(realize=True, backend="manifold"), require_requirements=False)
        if ok:
            ex["verification"] = verification_block(rec, True)
        emit(ex, ok, why)

    return write(out, accepted, rejected, cfg)


def write(out: Path, accepted: list[dict], rejected: list[dict], cfg: FactoryConfig) -> dict:
    files = {}
    by_split: dict[str, list[dict]] = {}
    for ex in sorted(accepted, key=lambda e: e["id"]):
        assert ex.get("verification", {}).get("passed") is True, "unverified example reached a split"
        by_split.setdefault(ex["split"], []).append(ex)
    for split, rows in sorted(by_split.items()):
        data = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows).encode()
        (out / f"{split}.jsonl").write_bytes(data)
        files[f"{split}.jsonl"] = {"sha256": sha256_bytes(data), "rows": len(rows)}
    rej = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n"
                  for r in sorted(rejected, key=lambda e: e["id"])).encode()
    (out / "rejected.jsonl").write_bytes(rej)
    fam = {s: sorted({r["family_id"] for r in rows}) for s, rows in by_split.items()}
    leaks = [f for s1 in fam for s2 in fam if s1 < s2 for f in set(fam[s1]) & set(fam[s2])]
    manifest = {
        "dataset": "berkelium-gear-v0", "schema_version": SCHEMA_VERSION, "generator": GENERATOR,
        "prompt_version": PROMPT_VERSION, "config": cfg.__dict__, "files": files,
        "rejected": {"rows": len(rejected), "sha256": sha256_bytes(rej),
                     "reasons": dict(Counter(r["rejected_reason"] for r in rejected))},
        "counts": {s: dict(Counter((r["task"], r["source"]["mode"]) for r in rows).most_common())
                   for s, rows in by_split.items()},
        "family_leakage": leaks,
        "examples": {r["id"]: sha256_of(r) for rows in by_split.values() for r in rows},
    }
    manifest["counts"] = {s: {f"{k[0]}:{k[1]}": v for k, v in c.items()} for s, c in manifest["counts"].items()}
    manifest["manifest_hash"] = sha256_of({k: v for k, v in manifest.items() if k != "manifest_hash"})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_split(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
