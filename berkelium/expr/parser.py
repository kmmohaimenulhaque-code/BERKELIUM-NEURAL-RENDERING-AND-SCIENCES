"""Pratt parser for the Berkelium expression language. No eval(), bounded size and depth."""

from __future__ import annotations

import re

from .ast import _PREC, Binary, Bool, Call, Name, Node, Num, Unary

MAX_SOURCE = 4096
MAX_DEPTH = 64


class ExprSyntaxError(ValueError):
    def __init__(self, msg: str, pos: int, source: str):
        super().__init__(f"{msg} at {pos}: {source!r}")
        self.pos = pos
        self.source = source


_TOKEN_RE = re.compile(
    r"""\s*(?:
      (?P<num>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
    | (?P<unit>\[[^\[\]]{1,64}\])
    | (?P<name>[A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)*)
    | (?P<op><=|>=|==|!=|[-+*/^(),<>])
    )""",
    re.VERBOSE,
)
_KEYWORDS = {"and", "or", "not", "true", "false"}


def _tokenize(src: str) -> list[tuple[str, str, int]]:
    out: list[tuple[str, str, int]] = []
    pos = 0
    while pos < len(src):
        if src[pos:].strip() == "":
            break
        m = _TOKEN_RE.match(src, pos)
        if not m or m.end() == pos:
            raise ExprSyntaxError("unexpected character", pos, src)
        kind = m.lastgroup
        assert kind is not None
        val = m.group(kind)
        start = m.start(kind)
        if kind == "name" and val in _KEYWORDS:
            kind = "kw"
        out.append((kind, val, start))
        pos = m.end()
    out.append(("end", "", len(src)))
    return out


class _Parser:
    def __init__(self, src: str):
        self.src = src
        self.toks = _tokenize(src)
        self.i = 0
        self.depth = 0

    def peek(self) -> tuple[str, str, int]:
        return self.toks[self.i]

    def take(self) -> tuple[str, str, int]:
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect(self, val: str) -> None:
        k, v, p = self.take()
        if v != val or k not in ("op",):
            raise ExprSyntaxError(f"expected {val!r}", p, self.src)

    def _infix_prec(self) -> int:
        k, v, _ = self.peek()
        if k == "op" and v in _PREC:
            return _PREC[v]
        if k == "kw" and v in ("and", "or"):
            return _PREC[v]
        return 0

    def expr(self, min_prec: int = 0) -> Node:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise ExprSyntaxError("expression nested too deeply", self.peek()[2], self.src)
        left = self.prefix()
        while True:
            prec = self._infix_prec()
            if prec <= min_prec:
                break
            _, op, pos = self.take()
            # '^' is right-associative; every other binary operator is left-associative.
            right = self.expr(prec - 1 if op == "^" else prec)
            if prec == 4 and isinstance(left, Binary) and _PREC.get(left.op) == 4:
                raise ExprSyntaxError("chained comparisons are not allowed; use 'and'", pos, self.src)
            left = Binary(op, left, right)
        self.depth -= 1
        return left

    def prefix(self) -> Node:
        k, v, p = self.take()
        if k == "num":
            node: Node = Num(float(v))
            if self.peek()[0] == "unit":
                u = self.take()[1][1:-1].strip()
                node = Num(float(v), u)
            return node
        if k == "kw" and v in ("true", "false"):
            return Bool(v == "true")
        if k == "kw" and v == "not":
            return Unary("not", self.expr(3))
        if k == "op" and v in ("-", "+"):
            return Unary(v, self.expr(7))
        if k == "op" and v == "(":
            inner = self.expr(0)
            self.expect(")")
            return inner
        if k == "name":
            if self.peek()[1] == "(" and self.peek()[0] == "op":
                self.take()
                args: list[Node] = []
                if self.peek()[1] != ")":
                    args.append(self.expr(0))
                    while self.peek()[1] == ",":
                        self.take()
                        args.append(self.expr(0))
                self.expect(")")
                return Call(v, tuple(args))
            return Name(v)
        raise ExprSyntaxError(f"unexpected token {v!r}", p, self.src)


def parse(src: str) -> Node:
    """Parse expression source into an AST."""
    if not isinstance(src, str):
        raise TypeError("expression source must be a string")
    if len(src) > MAX_SOURCE:
        raise ExprSyntaxError("expression too long", MAX_SOURCE, src[:64] + "...")
    p = _Parser(src)
    node = p.expr(0)
    k, v, pos = p.peek()
    if k != "end":
        raise ExprSyntaxError(f"unexpected trailing token {v!r}", pos, src)
    return node
