"""Deterministic, dimension-checked evaluator for expression ASTs. No eval()."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

from ..units import DIMENSIONLESS, Quantity, UnitError, dim_div, dim_mul, dim_pow, dim_str, parse_unit
from .ast import Binary, Bool, Call, Name, Node, Num, Unary

Value = Quantity | bool


class ExprError(ValueError):
    """Evaluation error: unknown name, dimension mismatch, domain error, non-finite result."""


def _q(v: Value, what: str) -> Quantity:
    if isinstance(v, bool) or not isinstance(v, Quantity):
        raise ExprError(f"{what}: expected a quantity, got {type(v).__name__}")
    return v


def _b(v: Value, what: str) -> bool:
    if not isinstance(v, bool):
        raise ExprError(f"{what}: expected a boolean")
    return v


def _finite(x: float, what: str) -> float:
    if not math.isfinite(x):
        raise ExprError(f"{what}: non-finite result")
    return x


def _dimless(q: Quantity, fn: str) -> float:
    if q.dim != DIMENSIONLESS:
        raise ExprError(f"{fn}() requires a dimensionless argument, got {dim_str(q.dim)}")
    return q.si


def _same(a: Quantity, b: Quantity, what: str) -> None:
    if a.dim != b.dim:
        raise ExprError(f"{what}: dimension mismatch {dim_str(a.dim)} vs {dim_str(b.dim)}")


def _unary_dimless(f: Callable[[float], float], name: str) -> Callable[[list[Value]], Value]:
    def g(args: list[Value]) -> Value:
        if len(args) != 1:
            raise ExprError(f"{name}() takes 1 argument")
        x = _dimless(_q(args[0], name), name)
        try:
            return Quantity(_finite(f(x), name))
        except (ValueError, OverflowError) as e:
            raise ExprError(f"{name}(): math domain error ({e})") from None
    return g


def _sqrt(args: list[Value]) -> Value:
    if len(args) != 1:
        raise ExprError("sqrt() takes 1 argument")
    q = _q(args[0], "sqrt")
    if any(e % 2 for e in q.dim):
        raise ExprError(f"sqrt() of {dim_str(q.dim)} has fractional dimension")
    if q.si < 0:
        raise ExprError("sqrt(): negative argument")
    return Quantity(math.sqrt(q.si), tuple(e // 2 for e in q.dim))  # type: ignore[arg-type]


def _abs(args: list[Value]) -> Value:
    if len(args) != 1:
        raise ExprError("abs() takes 1 argument")
    q = _q(args[0], "abs")
    return Quantity(abs(q.si), q.dim)


def _minmax(fn: Callable[..., float], name: str) -> Callable[[list[Value]], Value]:
    def g(args: list[Value]) -> Value:
        if len(args) < 2:
            raise ExprError(f"{name}() takes at least 2 arguments")
        qs = [_q(a, name) for a in args]
        for q in qs[1:]:
            _same(qs[0], q, name)
        return Quantity(fn(q.si for q in qs), qs[0].dim)
    return g


def _pow(args: list[Value]) -> Value:
    if len(args) != 2:
        raise ExprError("pow() takes 2 arguments")
    return _power(_q(args[0], "pow"), _q(args[1], "pow"))


def _power(base: Quantity, exp: Quantity) -> Quantity:
    e = _dimless(exp, "^")
    if base.dim != DIMENSIONLESS:
        if not float(e).is_integer():
            raise ExprError("non-integer power of a dimensioned quantity")
        dim = dim_pow(base.dim, int(e))
    else:
        dim = DIMENSIONLESS
    try:
        val = math.pow(base.si, e)
    except (ValueError, OverflowError, ZeroDivisionError) as ex:
        raise ExprError(f"power: {ex}") from None
    return Quantity(_finite(val, "^"), dim)


def _atan2(args: list[Value]) -> Value:
    if len(args) != 2:
        raise ExprError("atan2() takes 2 arguments")
    y, x = _q(args[0], "atan2"), _q(args[1], "atan2")
    _same(y, x, "atan2")
    return Quantity(math.atan2(y.si, x.si))


def _clamp(args: list[Value]) -> Value:
    if len(args) != 3:
        raise ExprError("clamp() takes 3 arguments")
    x, lo, hi = (_q(a, "clamp") for a in args)
    _same(x, lo, "clamp")
    _same(x, hi, "clamp")
    return Quantity(min(max(x.si, lo.si), hi.si), x.dim)


def _round_like(f: Callable[[float], float], name: str) -> Callable[[list[Value]], Value]:
    def g(args: list[Value]) -> Value:
        if len(args) != 1:
            raise ExprError(f"{name}() takes 1 argument")
        return Quantity(float(f(_dimless(_q(args[0], name), name))))
    return g


def _involute(args: list[Value]) -> Value:
    """inv(alpha) = tan(alpha) - alpha  (KHK Gear Technical Reference, Eq. 3.2)."""
    if len(args) != 1:
        raise ExprError("inv() takes 1 argument")
    a = _dimless(_q(args[0], "inv"), "inv")
    if not 0.0 <= a < math.pi / 2:
        raise ExprError("inv(): pressure angle must be in [0, pi/2)")
    return Quantity(math.tan(a) - a)


FUNCTIONS: dict[str, Callable[[list[Value]], Value]] = {
    "sin": _unary_dimless(math.sin, "sin"),
    "cos": _unary_dimless(math.cos, "cos"),
    "tan": _unary_dimless(math.tan, "tan"),
    "asin": _unary_dimless(math.asin, "asin"),
    "acos": _unary_dimless(math.acos, "acos"),
    "atan": _unary_dimless(math.atan, "atan"),
    "exp": _unary_dimless(math.exp, "exp"),
    "log": _unary_dimless(math.log, "log"),
    "atan2": _atan2,
    "sqrt": _sqrt,
    "abs": _abs,
    "min": _minmax(min, "min"),
    "max": _minmax(max, "max"),
    "pow": _pow,
    "clamp": _clamp,
    "floor": _round_like(math.floor, "floor"),
    "ceil": _round_like(math.ceil, "ceil"),
    "round": _round_like(round, "round"),
    "inv": _involute,
}
CONSTANTS: dict[str, Quantity] = {"pi": Quantity(math.pi)}


def evaluate(node: Node, env: Mapping[str, Value | float | int] | None = None) -> Value:
    """Evaluate an AST. Plain numbers in ``env`` are treated as dimensionless."""
    env = env or {}

    def ev(n: Node) -> Value:
        match n:
            case Num(value=v, unit=u):
                if u is None:
                    return Quantity(v)
                try:
                    f, d = parse_unit(u)
                except UnitError as e:
                    raise ExprError(str(e)) from None
                return Quantity(v * f, d)
            case Bool(value=v):
                return v
            case Name(id=i):
                if i in env:
                    val = env[i]
                    if isinstance(val, bool | Quantity):
                        return val
                    if isinstance(val, int | float):
                        return Quantity(float(val))
                    raise ExprError(f"unsupported value for {i!r}: {type(val).__name__}")
                if i in CONSTANTS:
                    return CONSTANTS[i]
                raise ExprError(f"unknown name {i!r}")
            case Unary(op="not", operand=o):
                return not _b(ev(o), "not")
            case Unary(op=op, operand=o):
                q = _q(ev(o), op)
                return Quantity(-q.si if op == "-" else q.si, q.dim)
            case Binary(op="and", left=a, right=b):
                return _b(ev(a), "and") and _b(ev(b), "and")  # short-circuit is deterministic
            case Binary(op="or", left=a, right=b):
                return _b(ev(a), "or") or _b(ev(b), "or")
            case Binary(op=op, left=a, right=b):
                x, y = ev(a), ev(b)
                if op in ("==", "!=") and isinstance(x, bool) and isinstance(y, bool):
                    return (x == y) if op == "==" else (x != y)
                qa, qb = _q(x, op), _q(y, op)
                if op in ("+", "-"):
                    _same(qa, qb, op)
                    return Quantity(_finite(qa.si + qb.si if op == "+" else qa.si - qb.si, op), qa.dim)
                if op == "*":
                    return Quantity(_finite(qa.si * qb.si, op), dim_mul(qa.dim, qb.dim))
                if op == "/":
                    if qb.si == 0.0:
                        raise ExprError("division by zero")
                    return Quantity(_finite(qa.si / qb.si, op), dim_div(qa.dim, qb.dim))
                if op == "^":
                    return _power(qa, qb)
                _same(qa, qb, op)
                return {"<": qa.si < qb.si, "<=": qa.si <= qb.si, ">": qa.si > qb.si,
                        ">=": qa.si >= qb.si, "==": qa.si == qb.si, "!=": qa.si != qb.si}[op]
            case Call(func="if", args=args):
                if len(args) != 3:
                    raise ExprError("if() takes 3 arguments")
                return ev(args[1]) if _b(ev(args[0]), "if") else ev(args[2])
            case Call(func=f, args=args):
                if f not in FUNCTIONS:
                    raise ExprError(f"unknown function {f!r}")
                return FUNCTIONS[f]([ev(a) for a in args])
        raise ExprError(f"cannot evaluate node {n!r}")

    return ev(node)
