"""Typed AST for the Berkelium expression language (ADR-003). Nodes are immutable."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Num:
    value: float
    unit: str | None = None  # literal unit annotation, e.g. 2[mm]


@dataclass(frozen=True, slots=True)
class Bool:
    value: bool


@dataclass(frozen=True, slots=True)
class Name:
    id: str


@dataclass(frozen=True, slots=True)
class Unary:
    op: str  # "-", "+", "not"
    operand: Node


@dataclass(frozen=True, slots=True)
class Binary:
    op: str  # + - * / ^ < <= > >= == != and or
    left: Node
    right: Node


@dataclass(frozen=True, slots=True)
class Call:
    func: str
    args: tuple[Node, ...]


Node = Num | Bool | Name | Unary | Binary | Call

_PREC = {"or": 1, "and": 2, "<": 4, "<=": 4, ">": 4, ">=": 4, "==": 4, "!=": 4,
         "+": 5, "-": 5, "*": 6, "/": 6, "^": 8}


def free_names(node: Node) -> frozenset[str]:
    """Identifiers referenced by an expression (constants like ``pi`` excluded)."""
    match node:
        case Name(id=i):
            return frozenset() if i in ("pi",) else frozenset({i})
        case Unary(operand=o):
            return free_names(o)
        case Binary(left=a, right=b):
            return free_names(a) | free_names(b)
        case Call(args=args):
            out: frozenset[str] = frozenset()
            for a in args:
                out |= free_names(a)
            return out
        case _:
            return frozenset()


def to_source(node: Node) -> str:
    """Canonical, fully parenthesised source text. parse(to_source(n)) == n."""
    match node:
        case Num(value=v, unit=u):
            s = repr(float(v))
            return f"{s}[{u}]" if u else s
        case Bool(value=v):
            return "true" if v else "false"
        case Name(id=i):
            return i
        case Unary(op=op, operand=o):
            return f"(not {to_source(o)})" if op == "not" else f"({op}{to_source(o)})"
        case Binary(op=op, left=a, right=b):
            return f"({to_source(a)} {op} {to_source(b)})"
        case Call(func=f, args=args):
            return f"{f}({', '.join(to_source(a) for a in args)})"
    raise TypeError(f"not an expression node: {node!r}")
