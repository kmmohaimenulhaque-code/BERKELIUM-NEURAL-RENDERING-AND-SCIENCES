"""Acausal law networks (ADR-020) — the primitive underneath CEMs and the Evidence Calculus.

Descent. A CEM is a hand-written *computation*. Underneath every such computation is a set of *relations*
between physical quantities that have no direction: sigma = 6 P L / (b h^2) can size h from sigma, or check
sigma from h. Which variable is computed from which is not engineering knowledge; it is a consequence of what
is KNOWN in a given problem. So the primitive is the acausal relation, and "the computation" is derived.

Primitives
  Var        a named quantity with a unit (dimension) and optional bounds.
  Relation   'lhs == rhs' (law) or 'lhs <= rhs' / 'lhs >= rhs' (requirement/constraint), in Berkelium's safe
             expression language; dimension-checked at construction; carries validity predicates,
             fidelity, model-form bound, references, assumptions.
  LawSet     a set of Vars + Relations. Composition = union over shared variable names (shared names ARE
             the interface; unit clashes are rejected).

Machinery (domain-free)
  solve      structural analysis (bipartite matching of unknowns to equations, as in acausal modelling
             languages) -> block-lower-triangular order (Tarjan SCC) -> per-block numerical solve (bracketed
             Brent for scalar blocks, with ALL roots found and disambiguated by bounds/validity; Newton-type
             for coupled blocks) -> inequality and validity checks. Diagnoses UNDERDETERMINED (which
             variables to specify), OVERDETERMINED (extra equations become consistency checks -> CONFLICT if
             violated), AMBIGUOUS (several admissible roots), NO_ROOT, INVALID_DOMAIN.
  sensitivities  log-log elasticities d ln y / d ln x by re-solving (central differences).
  propagate  first-order (from elasticities) and deterministic Monte Carlo uncertainty.
  optimise   minimise/maximise an objective over free variables subject to the LawSet's inequalities.
  as_law     wraps a LawSet as an Evidence-Calculus Law, so composed analytic models enter the same evidence
             hierarchy as FEM and measurements.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..expr import ExprError, eval_str
from ..expr.ast import Binary, Call, Name, Node, Unary
from ..expr.evaluator import CONSTANTS, evaluate
from ..expr.parser import parse
from ..units import Quantity, UnitError

Status = Literal["solved", "underdetermined", "conflict", "ambiguous", "no_root", "invalid_domain", "error"]


class LawError(ValueError):
    pass


def _names(n: Node) -> set[str]:
    if isinstance(n, Name):
        return set() if n.id in CONSTANTS else {n.id}
    if isinstance(n, Binary):
        return _names(n.left) | _names(n.right)
    if isinstance(n, Unary):
        return _names(n.operand)
    if isinstance(n, Call):
        return set().union(*(_names(a) for a in n.args)) if n.args else set()
    return set()


@dataclass(frozen=True)
class Var:
    name: str
    unit: str = "1"
    lo: float | None = None          # bounds in ``unit``
    hi: float | None = None
    description: str = ""


def _scale(unit: str) -> float:
    return Quantity.of(1.0, unit).si


@dataclass(frozen=True)
class Relation:
    id: str
    text: str
    validity: tuple[str, ...] = ()
    fidelity: Literal["definition", "exact", "analytic", "empirical", "requirement"] = "analytic"
    model_form_rel: float | None = 0.0      # relative bound vs the idealised quantity; None = unknown
    references: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()

    @property
    def node(self) -> Binary:
        n = parse(self.text)
        if not (isinstance(n, Binary) and n.op in ("==", "<=", ">=")):
            raise LawError(f"{self.id}: relation must be 'a == b', 'a <= b' or 'a >= b'")
        return n

    @property
    def kind(self) -> str:
        return self.node.op

    @property
    def vars(self) -> set[str]:
        return _names(self.node)

    def sides(self, env: Mapping[str, Quantity]) -> tuple[Quantity, Quantity]:
        n = self.node
        a, b = evaluate(n.left, env), evaluate(n.right, env)
        a = a if isinstance(a, Quantity) else Quantity.of(float(a))
        b = b if isinstance(b, Quantity) else Quantity.of(float(b))
        if a.dim != b.dim:
            raise UnitError(f"{self.id}: dimension mismatch {a.dim} vs {b.dim}")
        return a, b

    def residual(self, env) -> float:
        a, b = self.sides(env)
        return (a.si - b.si) / max(abs(a.si), abs(b.si), 1e-300)


@dataclass
class LawSet:
    name: str
    vars: dict[str, Var]
    relations: list[Relation]

    def __post_init__(self):
        ids = [r.id for r in self.relations]
        if len(ids) != len(set(ids)):
            raise LawError(f"{self.name}: duplicate relation ids")
        order = {n: i for i, n in enumerate(sorted(self.vars))}

        def probe(v: Var, k: int) -> float:   # distinct, non-special, in-bounds values (no x - x = 0 traps)
            j = 0.0731 * order[v.name] % 1.0
            if v.lo is not None and v.hi is not None:
                return v.lo + (v.hi - v.lo) * (0.17 + 0.6 * ((j + 0.29 * k) % 1.0))
            base = (abs(v.lo) + 1.0) if v.lo is not None else 1.0
            return base * (1.2345 + j + 0.5 * k)
        for r in self.relations:
            missing = r.vars - set(self.vars)
            if missing:
                raise LawError(f"{r.id}: undeclared variables {sorted(missing)}")
            err = None
            for k in range(3):                       # dimensional consistency, once, at definition time
                env = {n: Quantity.of(probe(v, k), v.unit) for n, v in self.vars.items()}
                try:
                    r.sides(env)
                    err = None
                    break
                except UnitError as e:
                    raise LawError(f"{r.id}: {e}") from None
                except (ExprError, ZeroDivisionError, ValueError, OverflowError) as e:
                    err = e
            if err is not None:
                raise LawError(f"{r.id}: could not be evaluated for a dimension check: {err}")

    def compose(self, *others: LawSet, name: str | None = None) -> LawSet:
        vs = dict(self.vars)
        rel = list(self.relations)
        for o in others:
            for k, v in o.vars.items():
                if k in vs and Quantity.of(1, vs[k].unit).dim != Quantity.of(1, v.unit).dim:
                    raise LawError(f"interface clash on {k}: {vs[k].unit} vs {v.unit}")
                vs.setdefault(k, v)
            rel += [r for r in o.relations if r.id not in {x.id for x in rel}]
        return LawSet(name or "+".join([self.name, *(o.name for o in others)]), vs, rel)


# ------------------------------------------------------------------------------------------- structure
def _match(eqs: list[Relation], unknown: set[str]) -> dict[str, str]:
    """Maximum bipartite matching equation-id -> unknown (augmenting paths, deterministic order)."""
    adj = {e.id: sorted(e.vars & unknown) for e in eqs}
    owner: dict[str, str] = {}

    def aug(eid, seen):
        for v in adj[eid]:
            if v in seen:
                continue
            seen.add(v)
            if v not in owner or aug(owner[v], seen):
                owner[v] = eid
                return True
        return False
    for e in sorted(adj, key=lambda k: (len(adj[k]), k)):
        aug(e, set())
    return {eid: v for v, eid in owner.items()}


def _blocks(eqs: dict[str, Relation], match: dict[str, str], unknown: set[str]) -> list[list[str]]:
    """Tarjan SCC on 'equation i needs the unknown solved by equation j' -> topological blocks."""
    solver_of = {v: e for e, v in match.items()}
    deps = {e: sorted({solver_of[v] for v in eqs[e].vars & unknown if solver_of.get(v) not in (None, e)})
            for e in match}
    index, low, stack, on, out, i = {}, {}, [], set(), [], [0]

    def strong(v):
        index[v] = low[v] = i[0]
        i[0] += 1
        stack.append(v)
        on.add(v)
        for w in deps[v]:
            if w not in index:
                strong(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on.discard(w)
                comp.append(w)
                if w == v:
                    break
            out.append(sorted(comp))
    for v in sorted(deps):
        if v not in index:
            strong(v)
    return out          # Tarjan emits dependencies first


@dataclass
class Check:
    id: str
    kind: str
    status: Literal["pass", "fail", "invalid_domain", "not_evaluable"]
    detail: str
    margin: float | None = None      # relative slack for inequalities (>0 satisfied)


@dataclass
class Solution:
    status: Status
    values: dict[str, Quantity]
    plan: list[tuple[list[str], list[str]]] = field(default_factory=list)   # (relation ids, solved vars)
    checks: list[Check] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    roots: dict[str, list[float]] = field(default_factory=dict)

    def get(self, name: str, unit: str) -> float:
        return self.values[name].to(unit)


def _roots_1d(f, lo: float, hi: float, n: int = 400) -> list[float]:
    """All sign-change roots of f on [lo, hi] (log-spaced scan when the interval is positive and wide)."""
    from scipy.optimize import brentq
    if lo > 0 and hi / lo > 50:
        xs = np.geomspace(lo, hi, n)
    else:
        xs = np.linspace(lo, hi, n)
    vals = []
    for x in xs:
        try:
            v = f(x)
            vals.append(v if math.isfinite(v) else np.nan)
        except (ExprError, UnitError, ValueError, ZeroDivisionError, OverflowError):
            vals.append(np.nan)
    roots = []
    for k in range(len(xs) - 1):
        a, b = vals[k], vals[k + 1]
        if np.isnan(a) or np.isnan(b):
            continue
        if a == 0:
            roots.append(float(xs[k]))
        elif a * b < 0:
            r = brentq(f, xs[k], xs[k + 1], xtol=1e-300, rtol=1e-15, maxiter=300)
            if abs(f(r)) < 1e-9:     # reject poles (sign change through infinity)
                roots.append(float(r))
    if vals and not np.isnan(vals[-1]) and vals[-1] == 0:
        roots.append(float(xs[-1]))
    return sorted(set(roots))


def _bounds_si(v: Var) -> tuple[float, float]:
    s = _scale(v.unit)
    lo = v.lo * s if v.lo is not None else -1e12 * s
    hi = v.hi * s if v.hi is not None else 1e12 * s
    return lo, hi


def _search(f, lo: float, hi: float, tiny: float) -> list[float]:
    """Root search over [lo, hi] split at zero into log-spaced positive / negative ranges."""
    roots = []
    if hi > 0:
        a = max(lo, tiny)
        roots += _roots_1d(f, a, hi) if hi > a else []
        if lo <= 0 and _safe(f, 0.0) and abs(f(0.0)) < 1e-12:
            roots.append(0.0)
    if lo < 0:
        b = max(-hi, tiny)
        roots += [-x for x in (_roots_1d(lambda y: f(-y), b, -lo) if -lo > b else [])]
    return sorted(set(roots))


def _safe(f, x) -> bool:
    try:
        return math.isfinite(f(x))
    except Exception:  # noqa: BLE001
        return False


def solve(ls: LawSet, known: Mapping[str, Quantity]) -> Solution:
    for k in known:
        if k not in ls.vars:
            raise LawError(f"unknown variable {k!r}")
        if Quantity.of(1, ls.vars[k].unit).dim != known[k].dim:
            raise UnitError(f"{k}: given {known[k].dim}, declared {ls.vars[k].unit}")
    env: dict[str, Quantity] = dict(known)
    eqs = [r for r in ls.relations if r.kind == "=="]
    # only variables that appear in equations are unknowns; a limit that appears only in a requirement and
    # is not given simply leaves that requirement unevaluated
    unknown = set().union(*(e.vars for e in eqs)) - set(known) if eqs else set()
    match = _match(eqs, unknown)
    sol = Solution("solved", env)
    unmatched_vars = set(unknown - set(match.values()))
    byid = {e.id: e for e in eqs}
    blocked: set[str] = set(unmatched_vars)        # variables that cannot be determined (and dependants)
    if unmatched_vars:
        sol.status = "underdetermined"
        # every unknown connected (through equations) to an unmatched one shares the missing freedom
        changed = True
        while changed:
            changed = False
            for e in eqs:
                if e.vars & blocked and e.id in match and match[e.id] not in blocked:
                    blocked.add(match[e.id])
                    changed = True
        sol.diagnostics.append(f"underdetermined: {len(unmatched_vars)} degree(s) of freedom unspecified; "
                               f"specify {len(unmatched_vars)} of {sorted(blocked)} (everything else is solved)")
    for block in _blocks(byid, match, unknown):
        vs = [match[e] for e in block]
        if set(vs) & blocked:
            continue
        sol.plan.append((block, vs))
        if len(block) == 1:
            r, v = byid[block[0]], vs[0]
            var = ls.vars[v]
            unit = Quantity.of(1, var.unit)

            def f(x, r=r, v=v, unit=unit):
                return r.residual({**env, v: Quantity(x, unit.dim)})
            lo, hi = _bounds_si(var)
            roots = _search(f, lo, hi, 1e-12 * _scale(var.unit))
            sol.roots[v] = [x / _scale(var.unit) for x in roots]
            ok = [x for x in roots if _admissible(ls, {**env, v: Quantity(x, unit.dim)}, v)]
            if not roots:
                sol.status = "no_root"
                sol.diagnostics.append(f"{r.id}: no value of {v} in [{var.lo}, {var.hi}] {var.unit} satisfies it")
                return sol
            if not ok:
                sol.status = "invalid_domain"
                sol.diagnostics.append(f"{v}: roots {sol.roots[v]} {var.unit} violate validity/bounds")
                return sol
            if len(ok) > 1:
                sol.status = "ambiguous"
                sol.diagnostics.append(f"{v}: {len(ok)} admissible roots {[x / _scale(var.unit) for x in ok]} "
                                       f"{var.unit}; add a bound or validity condition to select one")
                return sol
            env[v] = Quantity(ok[0], unit.dim)
        else:
            from scipy.optimize import root
            dims = [Quantity.of(1, ls.vars[v].unit).dim for v in vs]
            x0 = [math.sqrt(max(_bounds_si(ls.vars[v])[0], 1e-300) * _bounds_si(ls.vars[v])[1])
                  if ls.vars[v].lo is not None and ls.vars[v].hi is not None else _scale(ls.vars[v].unit) for v in vs]

            def F(x, block=block, vs=vs, dims=dims):
                e2 = {**env, **{v: Quantity(xi, d) for v, xi, d in zip(vs, x, dims, strict=True)}}
                return [byid[b].residual(e2) for b in block]
            res = root(F, x0, method="hybr", options={"xtol": 1e-14})
            if not res.success or max(abs(t) for t in F(res.x)) > 1e-9:
                sol.status = "no_root"
                sol.diagnostics.append(f"coupled block {block} did not converge: {res.message}")
                return sol
            for v, xi, d in zip(vs, res.x, dims, strict=True):
                env[v] = Quantity(float(xi), d)
    # overdetermined equations become consistency checks
    for e in eqs:
        if e.id not in match and e.vars <= set(env):
            res = e.residual(env)
            st = "pass" if abs(res) < 1e-9 else "fail"
            sol.checks.append(Check(e.id, "consistency", st, f"redundant equation residual {res:.2e}"))
            if st == "fail":
                sol.status = "conflict"
                sol.diagnostics.append(f"overdetermined and inconsistent: {e.id} residual {res:.3e}")
    for r in ls.relations:
        if r.kind == "==" and not r.vars <= set(env):
            continue
        if r.kind != "==" and not r.vars <= set(env):
            sol.checks.append(Check(r.id, r.kind, "not_evaluable", f"unspecified: {sorted(r.vars - set(env))}"))
            continue
        if r.kind != "==":
            a, b = r.sides(env)
            slack = (b.si - a.si) if r.kind == "<=" else (a.si - b.si)
            m = slack / max(abs(a.si), abs(b.si), 1e-300)
            sol.checks.append(Check(r.id, r.kind, "pass" if slack >= -1e-12 * max(abs(a.si), abs(b.si)) else "fail",
                                    f"{a.si:.6g} {r.kind} {b.si:.6g} (SI)", m))
        for p in r.validity:
            try:
                ok = eval_str(p, env) is True
                sol.checks.append(Check(r.id, "validity", "pass" if ok else "invalid_domain", p))
            except (ExprError, UnitError) as e:
                sol.checks.append(Check(r.id, "validity", "not_evaluable", f"{p}: {e}"))
    return sol


def _admissible(ls: LawSet, env, v: str) -> bool:
    var = ls.vars[v]
    x = env[v].to(var.unit)
    if (var.lo is not None and x < var.lo - 1e-12 * abs(var.lo)) or (var.hi is not None and x > var.hi + 1e-12 * abs(var.hi)):
        return False
    for r in ls.relations:
        for p in r.validity:
            names = _names(parse(p))
            if v in names and names <= set(env):   # only predicates that depend on the variable being chosen
                try:
                    if eval_str(p, env) is not True:
                        return False
                except (ExprError, UnitError):
                    return False
    return True


# ------------------------------------------------------------------------------ derived capabilities
def sensitivities(ls: LawSet, known: Mapping[str, Quantity], outputs: list[str], step: float = 1e-6
                  ) -> dict[str, dict[str, float]]:
    """Elasticities d ln(out) / d ln(in) for every known input (dimensionless, comparable across domains)."""
    base = solve(ls, known)
    if base.status != "solved":
        raise LawError(f"base point not solved: {base.status} {base.diagnostics}")
    out: dict[str, dict[str, float]] = {o: {} for o in outputs}
    for k, q in known.items():
        if q.si == 0:
            continue
        up = solve(ls, {**known, k: Quantity(q.si * (1 + step), q.dim)})
        dn = solve(ls, {**known, k: Quantity(q.si * (1 - step), q.dim)})
        if up.status != "solved" or dn.status != "solved":
            continue
        for o in outputs:
            out[o][k] = (math.log(abs(up.values[o].si)) - math.log(abs(dn.values[o].si))) / \
                (math.log(1 + step) - math.log(1 - step))
    return out


def propagate(ls: LawSet, known: Mapping[str, Quantity], rel_sigma: Mapping[str, float], outputs: list[str],
              n: int = 400, seed: int = 0) -> dict[str, dict[str, float]]:
    """Independent Gaussian relative uncertainties on inputs -> outputs: first-order and Monte Carlo."""
    el = sensitivities(ls, known, outputs)
    base = solve(ls, known)
    rng = np.random.default_rng(seed)
    draws = {o: [] for o in outputs}
    failed = 0
    for _ in range(n):
        kk = {k: Quantity(q.si * (1 + rel_sigma.get(k, 0.0) * rng.standard_normal()), q.dim) for k, q in known.items()}
        s = solve(ls, kk)
        if s.status != "solved":
            failed += 1
            continue
        for o in outputs:
            draws[o].append(s.values[o].si)
    res = {}
    for o in outputs:
        y = base.values[o].si
        lin = abs(y) * math.sqrt(sum((el[o].get(k, 0.0) * s) ** 2 for k, s in rel_sigma.items()))
        d = np.array(draws[o])
        res[o] = {"value_si": y, "sigma_linear_si": lin, "mc_mean_si": float(d.mean()), "mc_std_si": float(d.std()),
                  "mc_p05_si": float(np.percentile(d, 5)), "mc_p95_si": float(np.percentile(d, 95)),
                  "mc_failed": failed, "n": n}
    return res


def optimise(ls: LawSet, fixed: Mapping[str, Quantity], free: Mapping[str, tuple[float, float]], objective: str,
             minimise: bool = True, grid: int = 41) -> tuple[Solution, dict]:
    """Global-then-local search over free variables (bounds in their declared units) honouring every
    inequality. Deterministic: a coarse grid seeds bounded Nelder-Mead on a constraint-violation penalty."""
    from itertools import product

    from scipy.optimize import minimize
    names = sorted(free)
    units = [Quantity.of(1, ls.vars[k].unit) for k in names]
    evals = [0]

    def evaluate_x(x):
        evals[0] += 1
        s = solve(ls, {**fixed, **{k: Quantity.of(v, ls.vars[k].unit) for k, v in zip(names, x, strict=True)}})
        if s.status != "solved":
            return math.inf, s
        viol = sum(max(0.0, -c.margin) for c in s.checks if c.kind in ("<=", ">=") and c.margin is not None)
        bad = any(c.status == "invalid_domain" for c in s.checks)
        f = s.values[objective].si * (1 if minimise else -1)
        return (f, s) if viol == 0 and not bad else (math.inf if bad else abs(f) * (1 + 1e3 * viol) + 1e3 * viol, s)
    pts = [np.linspace(*free[k], grid if len(names) == 1 else max(5, int(grid ** (1 / len(names))))) for k in names]
    best = min((evaluate_x(np.array(p))[0], tuple(p)) for p in product(*pts))
    if not math.isfinite(best[0]):
        return Solution("no_root", {}, diagnostics=["no feasible point on the search grid"]), {"evaluations": evals[0]}
    r = minimize(lambda x: evaluate_x(x)[0], np.array(best[1]), method="Nelder-Mead",
                 bounds=[free[k] for k in names], options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 4000})
    f, s = evaluate_x(r.x)
    active = [c.id for c in s.checks if c.kind in ("<=", ">=") and c.margin is not None and c.margin < 1e-6]
    _ = units
    return s, {"evaluations": evals[0], "objective_si": f if minimise else -f, "active_constraints": active,
               "x": dict(zip(names, map(float, r.x), strict=True))}


def as_law(ls: LawSet, output: str, unit: str, quantity: str, fidelity="analytic", cost: float = 1.0, id: str | None = None):
    """Expose a LawSet as an Evidence-Calculus Law. Model-form error = sum of relation bounds (None if any
    participating relation's bound is unknown) — composition propagates ignorance honestly."""
    from ..evidence.core import Est, Law

    def run(p):
        s = solve(ls, {k: v for k, v in p.items() if k in ls.vars})
        if output not in s.values or s.status not in ("solved", "underdetermined"):
            raise LawError(f"{ls.name}: {output} not determined ({s.status}): {s.diagnostics}")
        bad = [c for c in s.checks if c.kind == "validity" and c.status != "pass"]
        if bad:
            raise LawError(f"outside validity: {[(c.id, c.detail) for c in bad]}")
        used = {r for blk, _ in s.plan for r in blk}
        bounds = [r.model_form_rel for r in ls.relations if r.id in used]
        y = s.values[output].to(unit)
        mf = None if any(b is None for b in bounds) else abs(y) * sum(bounds)
        return Est(y, unit, 0.0, mf, note=f"law network {ls.name}: plan {[b for b, _ in s.plan]}")
    # validity predicates may involve solved quantities, so they are checked after solving (inside run)
    return Law(id or f"laws:{ls.name}", quantity, unit, fidelity, cost, run, [], {},
               sorted({x for r in ls.relations for x in r.references}))
