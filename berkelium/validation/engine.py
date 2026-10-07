"""Validation engine. Every result names its level, fidelity and evidence; ``not_evaluated`` is a
first-class outcome used whenever the information or method needed to decide does not exist."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from ..expr import ExprError, evaluate, parse
from ..schema.common import QuantityModel
from ..schema.design import Constraint, Parameter, Requirement
from ..schema.evaluation import Fidelity, ValidationReport, ValidationResult
from ..units import Quantity, UnitError

V = "core@0.1"
RULE = Fidelity(kind="rule")


def schema_result(ok: bool, message: str) -> ValidationResult:
    return ValidationResult(validator=f"schema.{V}", level=0, status="pass" if ok else "fail", target="$",
                            message=message, fidelity=Fidelity(kind="rule", method="pydantic"))


def parameter_results(params: list[Parameter]) -> list[ValidationResult]:
    out = []
    for p in params:
        if p.kind in ("length", "angle", "real", "count") and (p.lower is not None or p.upper is not None):
            v = p.value
            try:
                if isinstance(v, QuantityModel):
                    val = v.q().to(p.unit) if p.unit else v.value
                elif isinstance(v, int | float) and not isinstance(v, bool):
                    val = float(v)
                else:
                    raise TypeError("non-numeric value")
            except (UnitError, TypeError) as e:
                out.append(ValidationResult(validator=f"param.{V}", level=1, status="fail", target=p.id,
                                            message=f"cannot check bounds: {e}", fidelity=RULE))
                continue
            ok = (p.lower is None or val >= p.lower) and (p.upper is None or val <= p.upper)
            out.append(ValidationResult(validator=f"param.{V}", level=1, status="pass" if ok else "fail",
                                        target=p.id, message=f"{val} within [{p.lower}, {p.upper}] {p.unit or ''}",
                                        fidelity=RULE))
    return out


def compare(measured: Quantity, comparator: str, target: Quantity, tol: Quantity | None) -> bool:
    if measured.dim != target.dim:
        raise UnitError("requirement and measured quantity have different dimensions")
    a, b = measured.si, target.si
    t = tol.si if tol is not None else 1e-9 * max(1.0, abs(b))
    return {"==": abs(a - b) <= t, "approx": abs(a - b) <= t, "<=": a <= b + t, ">=": a >= b - t,
            "<": a < b, ">": a > b}[comparator]


def requirement_results(reqs: list[Requirement], env: Mapping[str, Any],
                        sources: Mapping[str, str]) -> list[ValidationResult]:
    out = []
    for r in reqs:
        key = f"{r.applies_to}.{r.quantity}" if r.applies_to and not r.quantity.startswith(r.applies_to + ".") \
            else r.quantity
        if key not in env:
            out.append(ValidationResult(validator=f"requirement.{V}", level=2, status="not_evaluated", target=r.id,
                                        message=f"no computed or measured quantity named {key!r}",
                                        fidelity=RULE))
            continue
        q = env[key]
        try:
            ok = compare(q, r.comparator, r.target.q(), r.tolerance.q() if r.tolerance else None)
        except UnitError as e:
            out.append(ValidationResult(validator=f"requirement.{V}", level=2, status="fail", target=r.id,
                                        message=str(e), fidelity=RULE))
            continue
        unit = r.target.unit
        status = "pass" if ok else ("fail" if r.strength == "hard" else "warn")
        out.append(ValidationResult(
            validator=f"requirement.{V}", level=2, status=status, target=r.id,
            message=f"{key} {r.comparator} target ({'met' if ok else 'not met'})",
            measured=QuantityModel(value=q.to(unit), unit=unit), limit=r.target, comparator=r.comparator,
            evidence=[sources.get(key, "")], fidelity=Fidelity(kind="rule", method="requirement_compare")))
    return out


def constraint_results(cons: list[Constraint], env: Mapping[str, Any]) -> list[ValidationResult]:
    out = []
    for c in cons:
        node = parse(c.expr)
        try:
            val = evaluate(node, env)
        except ExprError as e:
            status = "not_evaluated" if "unknown name" in str(e) else "error"
            out.append(ValidationResult(validator=f"constraint.{V}", level=2, status=status, target=c.id,
                                        message=f"{c.expr}: {e}", fidelity=RULE))
            continue
        if not isinstance(val, bool):
            out.append(ValidationResult(validator=f"constraint.{V}", level=2, status="error", target=c.id,
                                        message=f"{c.expr}: constraint is not boolean", fidelity=RULE))
            continue
        status = "pass" if val else ("fail" if c.strength == "hard" else "warn")
        out.append(ValidationResult(validator=f"constraint.{V}", level=2, status=status, target=c.id,
                                    message=f"{c.expr} -> {val}", fidelity=Fidelity(kind="rule", method="expr")))
    return out


def geometric_results(target: str, meas, expected_solids: int | None = 1) -> list[ValidationResult]:
    """L3 (valid, closed, non-empty) and L4 (solid count) from backend measurements."""
    geo = Fidelity(kind="geometric", method=f"measure@{meas.backend}")
    ok3 = meas.valid and meas.closed and meas.volume_mm3 > 0
    out = [ValidationResult(validator=f"geometry.{V}", level=3, status="pass" if ok3 else "fail", target=target,
                            message=f"valid={meas.valid} closed={meas.closed} volume={meas.volume_mm3:.6g} mm^3",
                            measured=QuantityModel(value=meas.volume_mm3, unit="mm^3"), fidelity=geo)]
    if expected_solids is not None:
        ok4 = meas.n_solids == expected_solids
        out.append(ValidationResult(validator=f"topology.{V}", level=4, status="pass" if ok4 else "fail",
                                    target=target, message=f"{meas.n_solids} solid(s), expected {expected_solids}",
                                    measured=QuantityModel(value=meas.n_solids, unit="1"), fidelity=geo))
    return out


def summarize(results: list[ValidationResult]) -> ValidationReport:
    counts = Counter(r.status for r in results)
    if counts.get("error"):
        summary = "error"
    elif counts.get("fail"):
        summary = "fail"
    elif counts.get("warn"):
        summary = "warn"
    elif counts.get("pass"):
        summary = "pass"
    else:
        summary = "not_evaluated"
    evaluated = [r.level for r in results if r.status in ("pass", "fail", "warn")]
    return ValidationReport(results=results, summary=summary, counts=dict(sorted(counts.items())),  # type: ignore[arg-type]
                            highest_level_evaluated=max(evaluated) if evaluated else -1)


