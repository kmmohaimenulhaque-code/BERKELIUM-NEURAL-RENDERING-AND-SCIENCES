"""Vectorised (NumPy) evaluation of unit-free numeric expressions — used for implicit fields.

Field expressions are plain numbers in mm; unit annotations are rejected so that no implicit
SI conversion can change the meaning of a field. The same AST could later compile to PyTorch-ROCm."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .ast import Binary, Bool, Call, Name, Node, Num, Unary
from .evaluator import ExprError

_NP_FUNCS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs, "floor": np.floor, "ceil": np.ceil,
}


def evaluate_array(node: Node, env: Mapping[str, np.ndarray | float]) -> np.ndarray:
    def ev(n: Node):
        match n:
            case Num(value=v, unit=u):
                if u is not None:
                    raise ExprError("unit annotations are not allowed in field expressions")
                return v
            case Bool():
                raise ExprError("booleans are not allowed in field expressions")
            case Name(id=i):
                if i == "pi":
                    return np.pi
                if i not in env:
                    raise ExprError(f"unknown name {i!r}")
                return env[i]
            case Unary(op="-", operand=o):
                return -ev(o)
            case Unary(op="+", operand=o):
                return ev(o)
            case Binary(op=op, left=a, right=b) if op in "+-*/^":
                x, y = ev(a), ev(b)
                return {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide, "^": np.power}[op](x, y)
            case Call(func="min", args=args) if len(args) >= 2:
                out = ev(args[0])
                for a in args[1:]:
                    out = np.minimum(out, ev(a))
                return out
            case Call(func="max", args=args) if len(args) >= 2:
                out = ev(args[0])
                for a in args[1:]:
                    out = np.maximum(out, ev(a))
                return out
            case Call(func="atan2", args=(a, b)):
                return np.arctan2(ev(a), ev(b))
            case Call(func="clamp", args=(x, lo, hi)):
                return np.clip(ev(x), ev(lo), ev(hi))
            case Call(func=f, args=(a,)) if f in _NP_FUNCS:
                return _NP_FUNCS[f](ev(a))
        raise ExprError(f"unsupported in field expressions: {n!r}")

    with np.errstate(all="ignore"):
        out = np.asarray(ev(node), dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ExprError("field expression produced non-finite values")
    return out


def compile_scalar(node: Node, names: tuple[str, ...]):
    """Compile a unit-free numeric AST into a closure f(*values) -> float (no eval/exec).
    Used for per-point callbacks such as Manifold's level-set sampler."""
    import math as m

    fn1 = {"sin": m.sin, "cos": m.cos, "tan": m.tan, "asin": m.asin, "acos": m.acos, "atan": m.atan,
           "exp": m.exp, "log": m.log, "sqrt": m.sqrt, "abs": abs, "floor": m.floor, "ceil": m.ceil}
    idx = {n: i for i, n in enumerate(names)}

    def c(n: Node):
        match n:
            case Num(value=v, unit=None):
                return lambda a: v
            case Name(id="pi"):
                return lambda a: m.pi
            case Name(id=i) if i in idx:
                k = idx[i]
                return lambda a: a[k]
            case Unary(op="-", operand=o):
                f = c(o)
                return lambda a: -f(a)
            case Unary(op="+", operand=o):
                return c(o)
            case Binary(op=op, left=l, right=r) if op in "+-*/^":
                fl, fr = c(l), c(r)
                return {"+": lambda a: fl(a) + fr(a), "-": lambda a: fl(a) - fr(a),
                        "*": lambda a: fl(a) * fr(a), "/": lambda a: fl(a) / fr(a),
                        "^": lambda a: fl(a) ** fr(a)}[op]
            case Call(func="min", args=args) if len(args) >= 2:
                fs = [c(x) for x in args]
                return lambda a: min(f(a) for f in fs)
            case Call(func="max", args=args) if len(args) >= 2:
                fs = [c(x) for x in args]
                return lambda a: max(f(a) for f in fs)
            case Call(func="atan2", args=(y, x)):
                fy, fx = c(y), c(x)
                return lambda a: m.atan2(fy(a), fx(a))
            case Call(func="clamp", args=(x, lo, hi)):
                f, flo, fhi = c(x), c(lo), c(hi)
                return lambda a: min(max(f(a), flo(a)), fhi(a))
            case Call(func=fname, args=(x,)) if fname in fn1:
                g, f = fn1[fname], c(x)
                return lambda a: g(f(a))
        raise ExprError(f"unsupported in field expressions: {n!r}")

    f = c(node)
    return lambda *vals: float(f(vals))
