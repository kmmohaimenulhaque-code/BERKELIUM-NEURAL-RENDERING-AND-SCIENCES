"""Units and dimensions for Berkelium.

Design (ADR-012): Berkelium owns a small, deterministic unit system instead of
using pint inside the core. Every quantity is stored as a float magnitude in SI
base units plus an integer dimension vector. pint (or anything else) may be used
at I/O boundaries; nothing in the engineering path depends on it.

Conventions
-----------
* Plane angle is dimensionless (SI convention); ``deg`` is a scale factor of
  pi/180 onto radians. Semantic typing of angles happens at the Parameter level
  (``kind="angle"``), not in the dimension vector.
* Geometry kernels receive millimetres; conversion happens at one boundary
  (``Quantity.to("mm")``).
* Unit strings use ``*``, ``/`` and ``^`` (e.g. ``N*mm``, ``N/mm^2``, ``rev/min``).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache

# Base dimensions: length, mass, time, current, temperature, amount, luminous intensity
DIM_NAMES = ("L", "M", "T", "I", "Th", "N", "J")
Dim = tuple[int, int, int, int, int, int, int]
DIMENSIONLESS: Dim = (0, 0, 0, 0, 0, 0, 0)


class UnitError(ValueError):
    """Unknown unit, malformed unit string, or dimension mismatch."""


def _dim(L: int = 0, M: int = 0, T: int = 0, I: int = 0, Th: int = 0, N: int = 0, J: int = 0) -> Dim:  # noqa: E741
    return (L, M, T, I, Th, N, J)


def dim_mul(a: Dim, b: Dim) -> Dim:
    return tuple(x + y for x, y in zip(a, b, strict=True))  # type: ignore[return-value]


def dim_div(a: Dim, b: Dim) -> Dim:
    return tuple(x - y for x, y in zip(a, b, strict=True))  # type: ignore[return-value]


def dim_pow(a: Dim, p: int) -> Dim:
    return tuple(x * p for x in a)  # type: ignore[return-value]


def dim_str(d: Dim) -> str:
    parts = [f"{n}^{e}" if e != 1 else n for n, e in zip(DIM_NAMES, d, strict=True) if e]
    return "*".join(parts) if parts else "1"


# name -> (factor to SI, dimension). Offsets (degC/degF) are intentionally unsupported.
_LENGTH = _dim(L=1)
_MASS = _dim(M=1)
_TIME = _dim(T=1)
_FORCE = _dim(L=1, M=1, T=-2)
_PRESSURE = _dim(L=-1, M=1, T=-2)
_ENERGY = _dim(L=2, M=1, T=-2)
_POWER = _dim(L=2, M=1, T=-3)

UNITS: dict[str, tuple[float, Dim]] = {
    "1": (1.0, DIMENSIONLESS),
    # length
    "m": (1.0, _LENGTH),
    "cm": (1e-2, _LENGTH),
    "mm": (1e-3, _LENGTH),
    "um": (1e-6, _LENGTH),
    "in": (0.0254, _LENGTH),
    "ft": (0.3048, _LENGTH),
    # mass
    "kg": (1.0, _MASS),
    "g": (1e-3, _MASS),
    "lb": (0.45359237, _MASS),
    # time
    "s": (1.0, _TIME),
    "min": (60.0, _TIME),
    "h": (3600.0, _TIME),
    # angle (dimensionless, SI convention)
    "rad": (1.0, DIMENSIONLESS),
    "deg": (math.pi / 180.0, DIMENSIONLESS),
    "rev": (2.0 * math.pi, DIMENSIONLESS),
    "percent": (0.01, DIMENSIONLESS),
    # force
    "N": (1.0, _FORCE),
    "kN": (1e3, _FORCE),
    "lbf": (4.4482216152605, _FORCE),
    # pressure / stress
    "Pa": (1.0, _PRESSURE),
    "kPa": (1e3, _PRESSURE),
    "MPa": (1e6, _PRESSURE),
    "GPa": (1e9, _PRESSURE),
    "psi": (6894.757293168361, _PRESSURE),
    "kpsi": (6894757.293168361, _PRESSURE),
    # energy / power
    "J": (1.0, _ENERGY),
    "W": (1.0, _POWER),
    "kW": (1e3, _POWER),
    "hp": (745.6998715822702, _POWER),  # mechanical horsepower = 550 ft*lbf/s
    # temperature (absolute only)
    "K": (1.0, _dim(Th=1)),
}

_TOKEN = re.compile(r"\s*(?:(?P<name>[A-Za-z_][A-Za-z_0-9]*|1)|(?P<op>[*/^()])|(?P<num>-?\d+))")


@lru_cache(maxsize=1024)
def parse_unit(text: str) -> tuple[float, Dim]:
    """Parse a unit string such as ``N*mm``, ``N/mm^2`` or ``rev/min`` into (factor, dim)."""
    if not isinstance(text, str) or not text.strip():
        raise UnitError("empty unit string")
    tokens: list[tuple[str, str]] = []
    pos = 0
    s = text.strip()
    while pos < len(s):
        m = _TOKEN.match(s, pos)
        if not m or m.end() == pos:
            raise UnitError(f"malformed unit string {text!r} at {pos}")
        pos = m.end()
        kind = m.lastgroup
        assert kind is not None
        tokens.append((kind, m.group(kind)))

    idx = 0

    def peek() -> tuple[str, str] | None:
        return tokens[idx] if idx < len(tokens) else None

    def take() -> tuple[str, str]:
        nonlocal idx
        if idx >= len(tokens):
            raise UnitError(f"unexpected end of unit string {text!r}")
        t = tokens[idx]
        idx += 1
        return t

    def atom() -> tuple[float, Dim]:
        kind, val = take()
        if kind == "op" and val == "(":
            r = product()
            k2, v2 = take()
            if (k2, v2) != ("op", ")"):
                raise UnitError(f"missing ')' in {text!r}")
            base = r
        elif kind == "name":
            if val not in UNITS:
                raise UnitError(f"unknown unit {val!r} in {text!r}")
            base = UNITS[val]
        else:
            raise UnitError(f"unexpected token {val!r} in {text!r}")
        nxt = peek()
        if nxt == ("op", "^"):
            take()
            k3, v3 = take()
            if k3 != "num":
                raise UnitError(f"exponent must be an integer in {text!r}")
            p = int(v3)
            return base[0] ** p, dim_pow(base[1], p)
        return base

    def product() -> tuple[float, Dim]:
        f, d = atom()
        while (nxt := peek()) is not None and nxt[0] == "op" and nxt[1] in "*/":
            op = take()[1]
            f2, d2 = atom()
            if op == "*":
                f, d = f * f2, dim_mul(d, d2)
            else:
                f, d = f / f2, dim_div(d, d2)
        return f, d

    result = product()
    if idx != len(tokens):
        raise UnitError(f"trailing tokens in unit string {text!r}")
    return result


@dataclass(frozen=True, slots=True)
class Quantity:
    """A magnitude in SI base units with a dimension vector. Immutable."""

    si: float
    dim: Dim = DIMENSIONLESS

    @classmethod
    def of(cls, value: float, unit: str = "1") -> Quantity:
        f, d = parse_unit(unit)
        return cls(float(value) * f, d)

    def to(self, unit: str) -> float:
        f, d = parse_unit(unit)
        if d != self.dim:
            raise UnitError(f"cannot convert {dim_str(self.dim)} to {unit!r} ({dim_str(d)})")
        return self.si / f

    @property
    def dimensionless(self) -> bool:
        return self.dim == DIMENSIONLESS

    def __repr__(self) -> str:
        return f"Quantity({self.si!r}, {dim_str(self.dim)})"


def check_dim(unit: str, expected: str) -> None:
    """Raise UnitError if ``unit`` is not dimensionally compatible with ``expected``."""
    if parse_unit(unit)[1] != parse_unit(expected)[1]:
        raise UnitError(f"unit {unit!r} is not compatible with {expected!r}")
