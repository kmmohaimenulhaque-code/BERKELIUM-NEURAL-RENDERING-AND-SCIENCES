"""L7 numerical validation: turn SimulationResults into ValidationResults (PHYSICS-6).

Two kinds of L7 result:
  * solution verification (one per case): did the solver run and did every QoI converge?
      converged -> pass; non_converged -> indeterminate; failed -> error; unsupported/not_evaluated ->
      not_evaluated. Never collapsed into pass/fail.
  * requirement checks on ``sim.<case>.<qoi>`` quantities, THREE-VALUED on the Estimate's error interval:
      the whole interval satisfies the requirement -> pass; the whole interval violates it -> fail (warn if
      soft); the interval straddles the limit, or an error component is unknown -> indeterminate.
"""

from __future__ import annotations

from ..schema.common import QuantityModel
from ..schema.design import Requirement
from ..schema.evaluation import Fidelity, ValidationResult
from ..units import Quantity, UnitError
from .schema import SimulationResult

V = "l7@0.1"
_CASE_STATUS = {"converged": "pass", "non_converged": "indeterminate", "failed": "error",
                "unsupported": "not_evaluated", "not_evaluated": "not_evaluated"}


def _fid(r: SimulationResult) -> Fidelity:
    kind = "numerical" if r.fidelity == "numerical" else "analytic"
    method = f"{r.physics} [{r.fidelity}]" + (f" via {r.provenance.solver}" if r.provenance else "")
    return Fidelity(kind=kind, method=method, assumptions=list(r.assumptions),
                    references=list(r.provenance.references) if r.provenance else [])


def case_results(r: SimulationResult) -> list[ValidationResult]:
    ev = [m.sha256 for m in r.mesh] + [f.sha256 for f in r.fields]
    if r.provenance:
        ev.append(r.provenance.case_sha256)
    msg = f"{r.case_id}: solver status {r.status} ({r.fidelity}); {r.message}"
    return [ValidationResult(validator=f"simulation.{V}", level=7, status=_CASE_STATUS[r.status], target=r.target,
                             message=msg, evidence=ev, fidelity=_fid(r))]


def requirement_results(reqs: list[Requirement], sims: list[SimulationResult]) -> list[ValidationResult]:
    by = {(s.case_id, e.id): (s, e) for s in sims for e in s.estimates}
    out = []
    for q in reqs:
        if not q.quantity.startswith("sim."):
            continue
        parts = q.quantity.split(".")
        key = (parts[1], parts[2]) if len(parts) == 3 else None
        if key not in by:
            out.append(ValidationResult(validator=f"requirement.{V}", level=7, status="not_evaluated", target=q.id,
                                        message=f"no simulation estimate {q.quantity!r}",
                                        fidelity=Fidelity(kind="numerical")))
            continue
        s, e = by[key]
        try:
            k = Quantity.of(1.0, e.unit).to(q.target.unit)
        except UnitError as err:
            out.append(ValidationResult(validator=f"requirement.{V}", level=7, status="fail", target=q.id,
                                        message=str(err), fidelity=_fid(s)))
            continue
        iv = e.interval()
        val = e.value * k
        tgt = q.target.value
        tol = q.tolerance.q().to(q.target.unit) if q.tolerance else 0.0
        if iv is None:
            status, why = ("error" if e.status == "failed" else "indeterminate"), \
                f"estimate status {e.status}; error interval unknown"
        else:
            lo, hi = sorted((iv[0] * k, iv[1] * k))
            c = q.comparator
            if c in ("<=", "<"):
                ok, bad = hi <= tgt + tol, lo > tgt + tol
            elif c in (">=", ">"):
                ok, bad = lo >= tgt - tol, hi < tgt - tol
            else:
                ok, bad = (lo >= tgt - tol and hi <= tgt + tol), (hi < tgt - tol or lo > tgt + tol)
            status = "pass" if ok else ("fail" if q.strength == "hard" else "warn") if bad else "indeterminate"
            why = f"interval [{lo:.6g}, {hi:.6g}] {q.target.unit} vs {c} {tgt:g}"
        out.append(ValidationResult(
            validator=f"requirement.{V}", level=7, status=status, target=q.id,
            message=f"{q.quantity} = {val:.6g} {q.target.unit}; {why}",
            measured=QuantityModel(value=val, unit=q.target.unit), limit=q.target, comparator=q.comparator,
            evidence=[s.provenance.case_sha256] if s.provenance else [], fidelity=_fid(s)))
    return out
