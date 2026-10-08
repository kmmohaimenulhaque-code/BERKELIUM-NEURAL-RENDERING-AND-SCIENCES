"""Model synthesis (ADR-023): build the model a problem needs, instead of retrieving one.

Problem = (context facts, known quantities, target variable [, claim]).
1. ADMISSIBLE fragments: dimension-valid (quarantine) and context-compatible (fragment.context <= problem.context).
2. CANDIDATE MODELS: backward chaining from the target: a relation containing the target is chosen; every other
   unknown in it becomes a sub-goal; alternatives multiply (AND-OR search, capped). Each candidate is a minimal
   relation set that structurally determines the target.
3. VERIFICATION: each candidate is solved by the domain-free law engine (`laws.solve`): determinacy, all roots
   (ambiguity), validity predicates at the solution.
4. CLOSURE CHECK: every admissible relation whose variables are all known/derived must hold at the solution;
   otherwise the specification is CONTRADICTORY.
5. HYPOTHESIS COMPETITION: valid candidates that disagree beyond their declared model-form bounds -> CONFLICT;
   agreeing candidates -> DETERMINED (reporting the lowest-model-form one); no valid candidate -> OUTSIDE_VALIDITY
   (if candidates exist but all are out of domain) or UNDERDETERMINED (naming what is missing).
Nothing here knows any domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

from ..laws import LawSet, solve
from ..units import Quantity
from .library import VARS, Fragment, load

Status = str  # determined | underdetermined | contradictory | outside_validity | ambiguous | conflict


@dataclass
class Problem:
    context: frozenset[str]
    known: dict[str, Quantity]
    target: str


@dataclass
class Candidate:
    relations: tuple[str, ...]
    status: str
    value: float | None = None
    unit: str = ""
    model_form_rel: float | None = 0.0
    detail: str = ""


@dataclass
class Construction:
    status: Status
    value: float | None
    unit: str
    chosen: tuple[str, ...] = ()
    candidates: list[Candidate] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    detail: str = ""


def admissible(problem: Problem, frags: list[Fragment] | None = None) -> list[Fragment]:
    frags = frags if frags is not None else load()[0]
    return [f for f in frags if f.context <= problem.context]


def _derivations(target: str, known: set[str], frags: list[Fragment], depth: int = 16, cap: int = 64
                 ) -> list[frozenset[str]]:
    by_var: dict[str, list[Fragment]] = {}
    for f in frags:
        if f.relation.kind != "==":
            continue
        for v in f.relation.vars:
            by_var.setdefault(v, []).append(f)

    def derive(v: str, used: frozenset[str], d: int) -> list[frozenset[str]]:
        if v in known:
            return [frozenset()]
        if d == 0:
            return []
        out: list[frozenset[str]] = []
        for f in by_var.get(v, []):
            rid = f.relation.id
            if rid in used:
                continue
            others = sorted(f.relation.vars - {v} - known)
            subs = [derive(u, used | {rid}, d - 1) for u in others]
            if any(not s for s in subs):
                continue
            for combo in product(*subs) if subs else [()]:
                out.append(frozenset({rid}).union(*combo))
                if len(out) >= cap:
                    return out
        return out
    found = derive(target, frozenset(), depth)
    uniq = sorted(set(found), key=lambda s: (len(s), sorted(s)))
    return [s for s in uniq if not any(o < s for o in uniq)]       # minimal sets only


def _missing(target: str, known: set[str], frags: list[Fragment]) -> list[str]:
    """Variables reachable from the target through admissible relations that nothing determines."""
    seen, stack, rels = set(), [target], {}
    for f in frags:
        for v in f.relation.vars:
            rels.setdefault(v, []).append(f)
    while stack:
        v = stack.pop()
        if v in seen or v in known:
            continue
        seen.add(v)
        for f in rels.get(v, []):
            stack.extend(f.relation.vars - seen)
    return sorted(v for v in seen if v != target)


def construct(problem: Problem, frags: list[Fragment] | None = None) -> Construction:
    adm = admissible(problem, frags)
    known = set(problem.known)
    unit = VARS[problem.target].unit
    if problem.target in known:
        return Construction("determined", problem.known[problem.target].to(unit), unit, detail="target is given")
    sets = _derivations(problem.target, known, adm)
    if not sets:
        return Construction("underdetermined", None, unit, missing=_missing(problem.target, known, adm),
                            detail="no admissible relation chain reaches the knowns; specify one of the missing")
    byid = {f.relation.id: f for f in adm}
    cands: list[Candidate] = []
    for s in sets:
        rels = [byid[r].relation for r in sorted(s)]
        names = set().union(*(r.vars for r in rels)) | known
        ls = LawSet("candidate", {n: VARS[n] for n in names if n in VARS}, rels)
        sol = solve(ls, {k: q for k, q in problem.known.items() if k in ls.vars})
        if problem.target not in sol.values:
            cands.append(Candidate(tuple(sorted(s)), sol.status, detail="; ".join(sol.diagnostics)))
            continue
        bad = [c for c in sol.checks if c.kind == "validity" and c.status != "pass"]
        mf = [r.model_form_rel for r in rels]
        cands.append(Candidate(tuple(sorted(s)), "invalid_domain" if bad else sol.status,
                               sol.values[problem.target].to(unit), unit,
                               None if any(m is None for m in mf) else sum(mf),
                               "; ".join(f"{c.id}: {c.detail}" for c in bad)))
        # closure: every admissible relation fully determined by known + derived values must hold
        env = {**sol.values}
        for f in adm:
            r = f.relation
            # only DEFINITIONS / EXACT identities can prove a specification contradictory; alternative models
            # of the same quantity are hypotheses and compete below instead
            if r.kind == "==" and r.id not in s and r.vars <= set(env) and r.fidelity in ("definition", "exact"):
                ok_r = abs(r.residual(env)) < 1e-9
                if not ok_r:
                    valid_here = all(_pred(p, env) for p in r.validity)
                    if valid_here:
                        cands[-1].status = "contradictory"
                        cands[-1].detail += f"; closure violated by {r.id} (residual {r.residual(env):.2e})"
    valid = [c for c in cands if c.status == "solved"]
    if any(c.status == "contradictory" for c in cands):
        c0 = next(c for c in cands if c.status == "contradictory")
        return Construction("contradictory", None, unit, c0.relations, cands, detail=c0.detail)
    if not valid:
        if any(c.status == "ambiguous" for c in cands):
            return Construction("ambiguous", None, unit, (), cands,
                                detail=next(c.detail for c in cands if c.status == "ambiguous"))
        if any(c.status == "invalid_domain" for c in cands):
            return Construction("outside_validity", None, unit, (), cands,
                                detail="every candidate model is outside its validity domain")
        return Construction("underdetermined", None, unit, (), cands, _missing(problem.target, known, adm))
    # hypothesis competition among valid models
    for i, a in enumerate(valid):
        for b in valid[i + 1:]:
            ba = abs(a.value) * (a.model_form_rel or 0.0)
            bb = abs(b.value) * (b.model_form_rel or 0.0)
            if abs(a.value - b.value) > ba + bb + 1e-9 * max(abs(a.value), abs(b.value), 1e-300):
                return Construction("conflict", None, unit, (), cands,
                                    detail=f"{a.relations} -> {a.value:.6g} vs {b.relations} -> {b.value:.6g}")
    best = min(valid, key=lambda c: (c.model_form_rel if c.model_form_rel is not None else 9e9, len(c.relations)))
    return Construction("determined", best.value, unit, best.relations, cands)


def _pred(p: str, env) -> bool:
    from ..expr import eval_str
    try:
        return eval_str(p, env) is True
    except Exception:  # noqa: BLE001
        return False
