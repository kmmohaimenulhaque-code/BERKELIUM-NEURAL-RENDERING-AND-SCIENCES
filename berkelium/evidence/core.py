"""The Evidence Calculus (ADR-019): one mechanism for multi-fidelity reasoning in every physical domain.

Hypothesis. Analytic formulas, reduced-order correlations, FEM/CFD solvers, learned surrogates and lab
measurements are the same kind of object: an *evidence source* that maps a design point to an interval-valued
estimate of a quantity, valid only inside a declared domain, at a declared cost. Engineering questions are
*claims* (predicates on quantities). Deciding a claim is search over evidence sources.

Primitives
  Law          executable, typed model: inputs (dimension-checked), output quantity, validity predicates
               (safe-expression language, dimension-checked), fidelity, cost, and a model-form bound.
  Claim        quantity  comparator  target   (with units).
  Evidence     one Law (or Measurement) evaluated at one point: Estimate + validity verdict + provenance.
  Resolution   decided pass / fail, or indeterminate / insufficient_evidence, or CONFLICT — with the full
               evidence chain. Never a bare yes/no.

Algorithm (resolve)
  1. Candidate laws producing the quantity, cheapest first.
  2. Skip (and record why) any law whose validity predicates fail at the point, or that cannot be evaluated.
  3. Evaluate. Only an estimate with a KNOWN interval can decide; one with an unknown model-form error is
     kept as corroborating evidence only.
  4. Stop at the first decisive interval (pass or fail) — unless ``verify_with`` demands an escalation.
  5. Falsification: every pair of evaluated sources with known intervals must overlap. Disjoint intervals
     are a CONFLICT — some declared assumption, bound, measurement or implementation is wrong — and the
     claim is not decided, whatever any single source says.

Calibration (engineering memory). A cheap law's model-form bound may be LEARNED from verified high-fidelity
runs: max |relative discrepancy| + the reference's own error, stored with its sample count and the box (in
dimensionless groups) it covers. The bound applies only inside that box; outside, the law cannot decide.
Learned bounds never replace verification: they are derived from it and carry its provenance.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal

from ..expr import ExprError, eval_str
from ..units import Quantity, UnitError

Verdict = Literal["pass", "fail", "indeterminate", "insufficient_evidence", "conflict"]


@dataclass(frozen=True)
class Est:
    """Interval-valued estimate in a fixed unit. Unknown model-form error => cannot decide alone."""
    value: float
    unit: str
    numerical_error: float = 0.0
    model_form_error: float | None = 0.0
    converged: bool = True
    note: str = ""

    def interval(self) -> tuple[float, float] | None:
        if not self.converged or self.model_form_error is None:
            return None
        b = abs(self.numerical_error) + abs(self.model_form_error)
        return (self.value - b, self.value + b)


@dataclass
class Law:
    id: str
    quantity: str                       # what it predicts, e.g. "beam.tip_deflection"
    unit: str
    fidelity: Literal["analytic", "reduced_order", "numerical", "surrogate", "measurement"]
    cost: float                         # relative cost; ordering only
    run: Callable[[Mapping[str, Quantity]], Est]
    validity: list[str] = field(default_factory=list)      # predicates over inputs + derived groups
    groups: dict[str, str] = field(default_factory=dict)   # named dimensionless groups (expressions)
    references: list[str] = field(default_factory=list)

    def env(self, point: Mapping[str, Quantity]) -> dict[str, Quantity]:
        e = dict(point)
        for k, ex in self.groups.items():
            e[k] = _q(eval_str(ex, e))
        return e

    def check(self, point: Mapping[str, Quantity]) -> tuple[bool, str]:
        try:
            e = self.env(point)
            for p in self.validity:
                if eval_str(p, e) is not True:
                    return False, f"outside validity: {p}"
        except (ExprError, UnitError, KeyError, ZeroDivisionError) as err:
            return False, f"validity not evaluable: {err}"
        return True, "valid"


def _q(v) -> Quantity:
    return v if isinstance(v, Quantity) else Quantity.of(float(v))


@dataclass(frozen=True)
class Claim:
    quantity: str
    comparator: Literal["<=", ">=", "<", ">"]
    target: float
    unit: str

    def judge(self, iv: tuple[float, float]) -> Literal["pass", "fail", "indeterminate"]:
        lo, hi = iv
        if self.comparator in ("<=", "<"):
            return "pass" if hi <= self.target else ("fail" if lo > self.target else "indeterminate")
        return "pass" if lo >= self.target else ("fail" if hi < self.target else "indeterminate")


@dataclass
class Evidence:
    source: str
    fidelity: str
    status: Literal["evaluated", "invalid_domain", "error"]
    reason: str
    estimate: Est | None = None
    verdict: str | None = None


@dataclass
class Resolution:
    claim: Claim
    verdict: Verdict
    evidence: list[Evidence]
    decided_by: str | None
    conflicts: list[tuple[str, str]]
    cost: float

    def to_dict(self) -> dict:
        return {"claim": self.claim.__dict__, "verdict": self.verdict, "decided_by": self.decided_by,
                "conflicts": self.conflicts, "cost": self.cost,
                "evidence": [{**{k: v for k, v in e.__dict__.items() if k != "estimate"},
                              "estimate": e.estimate.__dict__ if e.estimate else None} for e in self.evidence]}

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, default=str).encode()).hexdigest()


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def resolve(claim: Claim, point: Mapping[str, Quantity], laws: list[Law],
            measurements: list[tuple[str, Est]] = (), verify_with: str | None = None,
            exhaustive: bool = False) -> Resolution:
    """Decide ``claim`` at ``point`` with the cheapest sufficient evidence (see module docstring)."""
    ev: list[Evidence] = []
    cost = 0.0
    decided_by = None
    verdict: Verdict = "insufficient_evidence"
    for name, m in measurements:
        if m.unit != claim.unit:
            raise UnitError(f"measurement {name} in {m.unit}, claim in {claim.unit}")
        iv = m.interval()
        ev.append(Evidence(name, "measurement", "evaluated", "measurement", m, claim.judge(iv) if iv else None))
    for law in sorted((x for x in laws if x.quantity == claim.quantity), key=lambda x: (x.cost, x.id)):
        ok, why = law.check(point)
        if not ok:
            ev.append(Evidence(law.id, law.fidelity, "invalid_domain", why))
            continue
        try:
            est = law.run(point)
        except Exception as err:  # a failing model is evidence of its own, never silently dropped
            ev.append(Evidence(law.id, law.fidelity, "error", f"{type(err).__name__}: {err}"))
            continue
        cost += law.cost
        if est.unit != claim.unit:
            k = Quantity.of(1.0, est.unit).to(claim.unit)
            est = Est(est.value * k, claim.unit, est.numerical_error * k,
                      None if est.model_form_error is None else est.model_form_error * k, est.converged, est.note)
        iv = est.interval()
        v = claim.judge(iv) if iv else None
        ev.append(Evidence(law.id, law.fidelity, "evaluated", "decisive" if v in ("pass", "fail") else
                           ("interval straddles target" if iv else "interval unknown: corroboration only"), est, v))
        if decided_by is None and v in ("pass", "fail") and (verify_with is None or law.id == verify_with):
            decided_by, verdict = law.id, v
            if not exhaustive:
                break
        elif decided_by is None and v == "indeterminate":
            verdict = "indeterminate"
    # falsification: all known intervals must be mutually consistent
    known = [(e.source, e.estimate.interval()) for e in ev if e.estimate and e.estimate.interval()]
    conflicts = [(a, b) for i, (a, ia) in enumerate(known) for (b, ib) in known[i + 1:] if not _overlap(ia, ib)]
    if conflicts:
        verdict, decided_by = "conflict", None
    return Resolution(claim, verdict, ev, decided_by, conflicts, cost)


# --------------------------------------------------------------------------------------- calibration
@dataclass(frozen=True)
class Calibration:
    """Learned model-form bound for a cheap law, relative to a verified reference law."""
    law: str
    reference: str
    groups: tuple[str, ...]
    box: tuple[tuple[float, float], ...]       # per group [min, max] actually sampled
    rel_bound: float                           # max |rel discrepancy| + max reference relative error
    n: int
    samples: tuple[tuple[float, ...], ...]

    def covers(self, g: Mapping[str, float]) -> bool:
        return all(lo - 1e-12 <= g[k] <= hi + 1e-12 for k, (lo, hi) in zip(self.groups, self.box, strict=True))


def calibrate(cheap: Law, reference: Law, points: list[Mapping[str, Quantity]], groups: tuple[str, ...]) -> Calibration:
    """Run both laws on points; keep only points where the reference is converged with a known interval."""
    rows, worst = [], 0.0
    for p in points:
        r, c = reference.run(p), cheap.run(p)
        iv = r.interval()
        if iv is None or r.value == 0:
            continue
        rel_disc = abs(c.value * Quantity.of(1.0, c.unit).to(r.unit) - r.value) / abs(r.value)
        rel_ref = (iv[1] - iv[0]) / 2 / abs(r.value)
        worst = max(worst, rel_disc + rel_ref)
        e = cheap.env(p)
        rows.append(tuple(_q(e[k]).to("1") for k in groups))
    if len(rows) < 2:
        raise ValueError("calibration needs at least 2 verified reference points")
    box = tuple((min(r[i] for r in rows), max(r[i] for r in rows)) for i in range(len(groups)))
    return Calibration(cheap.id, reference.id, groups, box, worst, len(rows), tuple(rows))


def calibrated(law: Law, cal: Calibration) -> Law:
    """A copy of ``law`` whose model-form bound is the learned one, valid ONLY inside the sampled box."""
    def run(p):
        e = law.run(p)
        return Est(e.value, e.unit, e.numerical_error, cal.rel_bound * abs(e.value), e.converged,
                   f"model-form bound learned from {cal.n} verified {cal.reference} runs")
    box_preds = [f"{g} >= {lo!r}" for g, (lo, _) in zip(cal.groups, cal.box, strict=True)] + \
                [f"{g} <= {hi!r}" for g, (_, hi) in zip(cal.groups, cal.box, strict=True)]
    return Law(f"{law.id}+cal", law.quantity, law.unit, law.fidelity, law.cost, run,
               law.validity + box_preds, law.groups, law.references + [f"calibrated vs {cal.reference} (n={cal.n})"])


_ = math  # reserved
